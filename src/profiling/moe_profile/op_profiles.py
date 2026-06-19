"""Diagnostic operation profiles for MegaBlocks-backed MoE runs."""

from __future__ import annotations

import argparse

import torch

from moe_profile.megablocks_adapter import (
    MegaBlocksRouting,
    megablocks_expert_dispatch,
    nano_aux_loss_from_router,
)
from moe_profile.runtime import cuda_time_ms, wall_time_ms


def promote_scalar_tensor(x: torch.Tensor) -> torch.Tensor:
    """MegaBlocks cumsum can return a scalar tensor for tiny inputs; normalize it."""

    return x.view(1) if not len(x.size()) else x


def measure_megablocks_phases(
    layer: torch.nn.Module,
    routing: MegaBlocksRouting,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Replay lower-level MegaBlocks implementation phases.

    This is an implementation view of the expert path: sort/bin metadata,
    gather, expert MLP, and scatter/combine. The timings are independent
    replays, not an exact additive breakdown of the measured forward call.
    """

    from megablocks import ops

    experts = layer.experts
    x_flat = routing.x_mb.view(-1, routing.x_mb.shape[-1])
    top_experts = routing.indices.flatten().int()
    expert_weights = routing.gates.flatten()
    top_k = args.top_k
    phase_warmup = args.phase_warmup
    phase_iters = args.phase_iters

    if phase_iters < 1:
        raise RuntimeError("--phase-iters must be >= 1.")

    def time_cuda(fn) -> float:
        return cuda_time_ms(
            fn,
            warmup=phase_warmup,
            iters=phase_iters,
            device=device,
        )

    def sort_phase():
        return ops.sort(top_experts, experts.sort_end_bit)

    bin_ids, indices = sort_phase()
    tokens_per_expert = ops.histogram(top_experts, experts.num_experts)
    bins = promote_scalar_tensor(ops.inclusive_cumsum(tokens_per_expert, 0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    metrics: dict[str, float | int | str] = {
        "phase_profile": True,
        "phase_profile_scope": "expert_path_implementation_replay",
        "phase_profile_warmup": phase_warmup,
        "phase_profile_iters": phase_iters,
        "phase_sort_ms": time_cuda(sort_phase),
        "phase_histogram_ms": time_cuda(
            lambda: ops.histogram(top_experts, experts.num_experts),
        ),
        "phase_cumsum_ms": time_cuda(
            lambda: promote_scalar_tensor(ops.inclusive_cumsum(tokens_per_expert, 0)),
        ),
    }

    if args.megablocks_layer == "moe":
        expert_capacity = experts.expert_capacity(args.batch_size * args.seq_len)
        if expert_capacity == 0:
            expert_capacity = int(torch.max(tokens_per_expert).item())
        gathered = ops.binned_gather(x_flat, indices, bins, expert_capacity, top_k)
        expert_out = experts.mlp(gathered)

        metrics.update({
            "phase_path": "standard_moe_binned",
            "phase_expert_capacity": int(expert_capacity),
            "phase_capacity_decision_wall_ms": wall_time_ms(
                lambda: int(torch.max(tokens_per_expert).item()),
                warmup=phase_warmup,
                iters=phase_iters,
                device=device,
            ),
            "phase_gather_ms": time_cuda(
                lambda: ops.binned_gather(x_flat, indices, bins, expert_capacity, top_k),
            ),
            "phase_expert_mlp_ms": time_cuda(lambda: experts.mlp(gathered)),
            "phase_scatter_ms": time_cuda(
                lambda: ops.binned_scatter(expert_out, indices, expert_weights, bins, top_k),
            ),
        })
    elif args.megablocks_layer == "dmoe":
        gathered = ops.gather(x_flat, indices, bin_ids, bins, top_k)
        expert_out = experts.mlp(gathered, tokens_per_expert)

        metrics.update({
            "phase_path": "dmoe_grouped",
            "phase_expert_capacity": 0,
            "phase_capacity_decision_wall_ms": 0.0,
            "phase_gather_ms": time_cuda(
                lambda: ops.gather(x_flat, indices, bin_ids, bins, top_k),
            ),
            "phase_expert_mlp_ms": time_cuda(lambda: experts.mlp(gathered, tokens_per_expert)),
            "phase_scatter_ms": time_cuda(
                lambda: ops.scatter(expert_out, indices, bin_ids, expert_weights, bins, top_k),
            ),
        })
    else:
        raise RuntimeError(f"Unsupported MegaBlocks layer for phase profiling: {args.megablocks_layer}")

    gpu_phase_keys = (
        "phase_sort_ms",
        "phase_histogram_ms",
        "phase_cumsum_ms",
        "phase_gather_ms",
        "phase_expert_mlp_ms",
        "phase_scatter_ms",
    )
    metrics["phase_gpu_sum_ms"] = float(sum(float(metrics[key]) for key in gpu_phase_keys))
    return metrics


def measure_megablocks_moe_ops(
    layer: torch.nn.Module,
    x: torch.Tensor,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float | int | str | bool]:
    """Replay logical MoE-layer operations for presentation-level diagnostics.

    This view follows the MoE layer boundary: router projection, top-k,
    selected-logit softmax/gating, MegaBlocks expert path, and final layout. The
    gate/combine field is a subset timing from the expert path, so it should not
    be added to the expert-block timing.
    """

    from megablocks import ops

    if args.moe_op_iters < 1:
        raise RuntimeError("--moe-op-iters must be >= 1.")

    warmup = args.moe_op_warmup
    iters = args.moe_op_iters

    def time_cuda(fn) -> float:
        return cuda_time_ms(fn, warmup=warmup, iters=iters, device=device)

    x_mb = x.transpose(0, 1).contiguous()
    flat_x = x_mb.view(-1, x_mb.shape[-1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    router_projection_ms = time_cuda(lambda: layer.router.layer(flat_x))
    logits = layer.router.layer(flat_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    topk_selection_ms = time_cuda(lambda: torch.topk(logits, args.top_k, dim=-1))
    top_values, top_indices = torch.topk(logits, args.top_k, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    selected_softmax_gating_ms = time_cuda(lambda: torch.softmax(top_values, dim=-1))
    gates = torch.softmax(top_values, dim=-1)
    router_probs = torch.softmax(logits, dim=-1)
    aux_loss = nano_aux_loss_from_router(router_probs, top_indices, args.n_experts)
    routing = MegaBlocksRouting(
        x_mb=x_mb,
        router_probs=router_probs,
        gates=gates,
        indices=top_indices,
        aux_loss=aux_loss,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    expert_block_ms = time_cuda(lambda: megablocks_expert_dispatch(layer, routing))
    expert_block_out = megablocks_expert_dispatch(layer, routing)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    experts = layer.experts
    top_experts = routing.indices.flatten().int()
    expert_weights = routing.gates.flatten()
    top_k = args.top_k

    bin_ids, indices = ops.sort(top_experts, experts.sort_end_bit)
    tokens_per_expert = ops.histogram(top_experts, experts.num_experts)
    bins = promote_scalar_tensor(ops.inclusive_cumsum(tokens_per_expert, 0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if args.megablocks_layer == "moe":
        expert_capacity = experts.expert_capacity(args.batch_size * args.seq_len)
        if expert_capacity == 0:
            expert_capacity = int(torch.max(tokens_per_expert).item())
        gathered = ops.binned_gather(flat_x, indices, bins, expert_capacity, top_k)
        expert_out = experts.mlp(gathered)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gate_combine_ms = time_cuda(
            lambda: ops.binned_scatter(expert_out, indices, expert_weights, bins, top_k),
        )
        op_path = "standard_moe_binned"
    elif args.megablocks_layer == "dmoe":
        expert_capacity = 0
        gathered = ops.gather(flat_x, indices, bin_ids, bins, top_k)
        expert_out = experts.mlp(gathered, tokens_per_expert)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gate_combine_ms = time_cuda(
            lambda: ops.scatter(expert_out, indices, bin_ids, expert_weights, bins, top_k),
        )
        op_path = "dmoe_grouped"
    else:
        raise RuntimeError(f"Unsupported MegaBlocks layer for MoE op profiling: {args.megablocks_layer}")

    output_layout_ms = time_cuda(lambda: expert_block_out.transpose(0, 1).contiguous())
    non_additive_sum_ms = (
        router_projection_ms
        + topk_selection_ms
        + selected_softmax_gating_ms
        + expert_block_ms
        + output_layout_ms
    )

    return {
        "moe_op_profile": True,
        "moe_op_profile_scope": "moe_layer_logical_replay",
        "moe_op_profile_warmup": warmup,
        "moe_op_profile_iters": iters,
        "moe_op_path": op_path,
        "moe_op_expert_capacity": int(expert_capacity),
        "moe_op_router_projection_matmul_ms": float(router_projection_ms),
        "moe_op_topk_selection_ms": float(topk_selection_ms),
        "moe_op_selected_softmax_gating_ms": float(selected_softmax_gating_ms),
        "moe_op_expert_block_dispatch_compute_combine_ms": float(expert_block_ms),
        "moe_op_gate_multiply_combine_ms": float(gate_combine_ms),
        "moe_op_output_layout_to_nano_ms": float(output_layout_ms),
        "moe_op_main_replay_sum_ms": float(non_additive_sum_ms),
        "moe_op_profile_note": (
            "Independent diagnostic timings. expert_block is the whole "
            "MegaBlocks expert path; gate_multiply_combine is the weighted "
            "scatter/combine subset and is not additive with expert_block."
        ),
    }

