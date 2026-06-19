"""Derived MoE profiling metrics and FLOP accounting."""

from __future__ import annotations

import argparse
from typing import Optional

import torch


def tokens_per_expert_metrics(tokens_per_expert: Optional[torch.Tensor]) -> dict[str, float | int]:
    """Summarize routed-token imbalance across experts."""

    if tokens_per_expert is None:
        return {}
    counts = tokens_per_expert.detach().float().cpu()
    mean = float(counts.mean().item())
    return {
        "tokens_per_expert_min": int(counts.min().item()),
        "tokens_per_expert_max": int(counts.max().item()),
        "tokens_per_expert_mean": mean,
        "tokens_per_expert_std": float(counts.std(unbiased=False).item()),
        "expert_imbalance": float(counts.max().item() / mean) if mean else 0.0,
    }


def expert_flop_multiplier(expert_type: str) -> int:
    """FLOP multiplier for one selected expert row.

    FFN uses two matmuls: ``4 * D * H`` FLOPs per selected row. GLU/SwiGLU uses
    three matmuls: ``6 * D * H`` FLOPs per selected row.
    """

    if expert_type == "ffn":
        return 4
    if expert_type == "glu":
        return 6
    raise ValueError(f"Unsupported expert_type={expert_type!r}; expected 'ffn' or 'glu'.")


def flops_metrics(
    args: argparse.Namespace,
    *,
    model_shape: dict[str, object],
    tokens_per_expert: Optional[torch.Tensor],
    mean_forward_ms: float,
) -> dict[str, float | int]:
    """Compute model-level useful FLOPs and backend FLOP estimates.

    ``active_expert_flops`` counts useful selected expert math. Backend estimates
    additionally reflect dense-reference all-expert work or standard-MoE padding.
    Router/top-k/softmax/layout work is timed but not included in expert FLOPs.
    """

    tokens = args.batch_size * args.seq_len
    assignments = tokens * args.top_k
    expert_type = str(model_shape.get("expert_type", "ffn") or "ffn")
    multiplier = expert_flop_multiplier(expert_type)
    shared_experts = int(model_shape.get("num_shared_experts", 0) or 0)
    shared_hidden = int(model_shape.get("shared_expert_intermediate_size", 0) or 0)
    router_flops = 2 * tokens * args.d_model * args.n_experts
    routed_active_per_token = multiplier * args.top_k * args.d_model * args.d_ff
    shared_per_token = multiplier * shared_experts * args.d_model * shared_hidden
    active_expert_flops_per_token = routed_active_per_token + shared_per_token
    dense_all_expert_flops_per_token = (
        multiplier * args.n_experts * args.d_model * args.d_ff + shared_per_token
    )
    active_expert_flops = tokens * active_expert_flops_per_token

    backend_expert_rows = assignments
    if args.backend == "reference":
        backend_expert_rows = tokens * args.n_experts
        backend_expert_flops = tokens * dense_all_expert_flops_per_token
    elif args.megablocks_layer == "moe" and tokens_per_expert is not None:
        backend_expert_rows = int(args.n_experts * tokens_per_expert.max().item())
        backend_expert_flops = (
            multiplier * backend_expert_rows * args.d_model * args.d_ff
            + tokens * shared_per_token
        )
    else:
        backend_expert_flops = active_expert_flops

    backend_total_flops = router_flops + backend_expert_flops
    seconds = mean_forward_ms / 1000.0
    padding_factor = float(backend_expert_rows / assignments) if assignments else 0.0
    return {
        "assignments": int(assignments),
        "router_flops": int(router_flops),
        "active_expert_flops": int(active_expert_flops),
        "backend_expert_flops_estimate": int(backend_expert_flops),
        "backend_total_flops_estimate": int(backend_total_flops),
        "megablocks_padded_expert_rows": int(backend_expert_rows if args.megablocks_layer == "moe" else 0),
        "backend_expert_rows": int(backend_expert_rows),
        "padding_factor": padding_factor,
        "routed_active_expert_flops_per_token": int(routed_active_per_token),
        "shared_expert_flops_per_token": int(shared_per_token),
        "active_expert_flops_per_token": int(active_expert_flops_per_token),
        "dense_all_expert_flops_per_token": int(dense_all_expert_flops_per_token),
        "backend_expert_flops_per_token": float(backend_expert_flops / tokens) if tokens else 0.0,
        "ms_per_input_token": float(mean_forward_ms / tokens) if tokens else 0.0,
        "ms_per_assignment": float(mean_forward_ms / assignments) if assignments else 0.0,
        "tokens_per_second": float(tokens / seconds),
        "active_expert_tflops_per_second": float(active_expert_flops / seconds / 1e12),
        "backend_estimated_tflops_per_second": float(backend_total_flops / seconds / 1e12),
    }


def bias_semantics(args: argparse.Namespace) -> str:
    """Human-readable bias policy recorded with every profiler row."""

    if args.use_expert_biases:
        return "matched_expert_biases"
    if args.zero_expert_biases:
        return "zero_expert_biases"
    if args.allow_bias_mismatch:
        return "intentional_bias_mismatch"
    return "nano_expert_biases"

