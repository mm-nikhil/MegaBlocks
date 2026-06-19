"""Model-shape and dtype helpers used by profiling entrypoints."""

from __future__ import annotations

import argparse
import json
from functools import partial

import torch
import torch.nn.functional as F


# The local grouped_gemm extension used by MegaBlocks dMoE validates BF16 input,
# weight, and output tensors in C++. Keep the fallback named and visible so a
# mixed FP32/BF16 run does not look like one homogeneous dtype experiment.
DMOE_BF16_ONLY_DTYPE_POLICY = "dmoe_bf16_only_local_grouped_gemm"


def parse_dtype(name: str) -> torch.dtype:
    """Map the CLI dtype name to the torch dtype used for tensors and modules."""

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Return activation/weight bytes per element for memory preflight estimates."""

    return {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }[dtype]


def resolve_backend_dtype(backend: str, requested_dtype: str) -> tuple[str, str]:
    """Return the effective dtype and the policy that selected it.

    NanoJAX runs should request FP32 by default. The exception is
    ``megablocks_dmoe`` in this checkout: its grouped GEMM backend is BF16-only,
    so dMoE rows are deliberately recorded as BF16 with an explicit policy.
    """

    if backend == "megablocks_dmoe" and requested_dtype != "bfloat16":
        return "bfloat16", DMOE_BF16_ONLY_DTYPE_POLICY
    return requested_dtype, "requested"


def activation_fn_from_name(name: str):
    """Resolve catalog activation names to torch callables."""

    if name == "gelu_tanh":
        return partial(F.gelu, approximate="tanh")
    if name == "silu":
        return F.silu
    raise RuntimeError(f"Unsupported activation={name!r}.")


def load_model_shape(args: argparse.Namespace) -> dict[str, object]:
    """Load and validate a named MoE shape from the shape catalog.

    The profiler CLI still accepts explicit shape values so focused axis sweeps
    can override catalog dimensions with ``--allow-shape-override``.
    """

    if not args.model_shape_name:
        return {}

    try:
        with args.model_shapes_config.open(encoding="utf-8") as handle:
            shapes = json.load(handle)
    except OSError as exc:
        raise RuntimeError(f"Could not read model shape catalog: {args.model_shapes_config}") from exc

    try:
        shape = shapes[args.model_shape_name]
    except KeyError as exc:
        available = ", ".join(sorted(shapes))
        raise RuntimeError(
            f"Unknown model shape {args.model_shape_name!r}. Available shapes: {available}",
        ) from exc

    expected = {
        "d_model": int(shape["hidden_size"]),
        "d_ff": int(shape["expert_intermediate_size"]),
        "n_experts": int(shape["num_routed_experts"]),
        "top_k": int(shape["num_experts_per_token"]),
    }
    mismatches = [
        f"{key}: arg={getattr(args, key)} catalog={value}"
        for key, value in expected.items()
        if getattr(args, key) != value
    ]
    max_t = int(shape.get("max_position_embeddings", 0) or 0)
    if max_t and args.seq_len > max_t:
        mismatches.append(f"seq_len: arg={args.seq_len} catalog_max={max_t}")
    if mismatches and not args.allow_shape_override:
        raise RuntimeError(
            "Profiler arguments do not match the selected model shape. "
            "Use --allow-shape-override for axis sweeps. Mismatches: "
            + "; ".join(mismatches),
        )
    return dict(shape)
