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
    "projection, top-k, selected-logit softmax/gating, MegaBlocks expert path, and "
    "output layout. expert_path times only the prepared MegaBlocks expert path: "
    "dispatch/sort/binning, gather, expert MLP, weighted combine/scatter, and shared "
    "expert combine if configured."
)


def resolve_timing_scope(args: argparse.Namespace) -> str:
    """Resolve ``auto`` into the actual timed boundary.

    ``moe_layer`` is the full MoE-layer boundary used for presentation and
    hardware-style accounting. ``expert_path`` isolates the MegaBlocks
    dispatch/compute/combine implementation after Nano-compatible routing has
    already been prepared.
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

