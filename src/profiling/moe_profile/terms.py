"""Shared profiling terms and timing-scope definitions."""

from __future__ import annotations

import argparse


# Public timing-scope names used in CSV/JSONL output. Keep these terms stable:
# they are presentation-facing, not internal implementation names.
TIMING_SCOPE_AUTO = "auto"
TIMING_SCOPE_MOE_LAYER = "moe_layer"
TIMING_SCOPE_EXPERT_PATH = "expert_path"
TIMING_SCOPE_CHOICES = (
    TIMING_SCOPE_AUTO,
    TIMING_SCOPE_MOE_LAYER,
    TIMING_SCOPE_EXPERT_PATH,
)

TIMING_SCOPE_HELP = (
    "auto uses expert_path for MegaBlocks and moe_layer for the reference backend. "
    "moe_layer times the full Nano-compatible MoE layer: layout conversion, router "
    "projection, full row-wise softmax for router probabilities, top-k, row-wise "
    "selected-logit softmax/gating, MegaBlocks expert block, weighted scatter/combine, "
    "and output layout. expert_path times only the prepared MegaBlocks expert block: "
    "dispatch/sort/binning, gather, expert MLP, weighted scatter/combine, and shared "
    "expert combine if configured."
)


def resolve_timing_scope(args: argparse.Namespace) -> str:
    """Resolve ``auto`` into the actual timed boundary.

    ``moe_layer`` is the full MoE-layer boundary used for presentation and
    hardware-style accounting. ``expert_path`` isolates the MegaBlocks
    dispatch/sort/gather, expert MLP, and weighted scatter/combine implementation
    after Nano-compatible routing has already been prepared.
    """

    if args.timing_scope != TIMING_SCOPE_AUTO:
        return args.timing_scope
    if args.backend == "megablocks":
        return TIMING_SCOPE_EXPERT_PATH
    return TIMING_SCOPE_MOE_LAYER


def require_valid_timing_scope(args: argparse.Namespace, timing_scope: str) -> None:
    """Validate timing-scope/backend combinations before allocating tensors."""

    if timing_scope == TIMING_SCOPE_EXPERT_PATH and args.backend != "megablocks":
        raise SystemExit("--timing-scope expert_path is only valid with --backend megablocks.")


def moe_semantic_record(
    *,
    expert_type: str,
    activation: str,
) -> dict[str, str]:
    """Return presentation terms for the MoE layer implemented by the profiler.

    These strings are intentionally plain-language and stable because they are
    written to CSV/JSONL outputs. They describe the profiled Nano-compatible
    adapter path, not an abstract MoE layer and not stock MegaBlocks training
    semantics.
    """

    if expert_type == "glu":
        expert_mlp = (
            "GLU expert MLP: gate projection, up projection, element-wise "
            f"{activation} on the gate projection, element-wise gate*up, down projection."
        )
    else:
        expert_mlp = (
            "FFN expert MLP: W1 matmul, optional expert bias, element-wise "
            f"{activation}, W2 matmul, optional output bias."
        )

    return {
        "routing_semantics": (
            "Nano-compatible adapter routing: router projection matmul; full row-wise "
            "softmax over expert logits for aux/load-balance probabilities; top-k over "
            "raw logits; row-wise softmax over selected top-k logits for gate weights."
        ),
        "softmax_location": (
            "In current Nano-compatible MegaBlocks runs, softmax is explicit adapter-side "
            "routing work. It is not inside the timed MegaBlocks expert path."
        ),
        "topk_location": (
            "In current Nano-compatible MegaBlocks runs, top-k selection is explicit "
            "adapter-side routing work over raw router logits."
        ),
        "gate_weight_semantics": (
            "Gate weights are row-wise softmax over the selected top-k router logits."
        ),
        "gate_multiply_location": (
            "Gate multiply and reduce back to token rows are folded into MegaBlocks "
            "weighted scatter/combine: ops.binned_scatter for standard MoE and "
            "ops.scatter for dMoE."
        ),
        "expert_mlp_semantics": expert_mlp,
        "expert_path_semantics": (
            "MegaBlocks expert path means dispatch metadata, sort/binning, gather, "
            "expert MLP compute, and weighted scatter/combine. Lower-level phase "
            "profiling splits this into sort, histogram, cumsum, gather, expert MLP, "
            "and scatter/combine."
        ),
    }
