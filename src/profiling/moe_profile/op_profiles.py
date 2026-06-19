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

    This view follows the MoE layer boundary using disjoint logical blocks:
    input layout, router projection matmul, full row-wise router softmax,
    top-k expert selection, row-wise selected-logit softmax for gate weights,
    aux/router bookkeeping, MegaBlocks expert block, and output layout. The
    expert block contains dispatch/sort/binning, gather, expert MLP compute,
    and weighted scatter/combine. The component sum is useful for explanation,
    but it is still a replay diagnostic: splitting kernels can change launch
    behavior, so the authoritative latency remains ``mean_forward_ms`` from
    the timed whole layer.
    """

    if args.moe_op_iters < 1:
        raise RuntimeError("--moe-op-iters must be >= 1.")

    warmup = args.moe_op_warmup
    iters = args.moe_op_iters

    def time_cuda(fn) -> float:
        return cuda_time_ms(fn, warmup=warmup, iters=iters, device=device)

    def input_layout():
        return x.transpose(0, 1).contiguous()

    input_layout_ms = time_cuda(input_layout)
    x_mb = input_layout()
    flat_x = x_mb.view(-1, x_mb.shape[-1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    router_projection_ms = time_cuda(lambda: layer.router.layer(flat_x))
    logits = layer.router.layer(flat_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    router_full_softmax_ms = time_cuda(lambda: torch.softmax(logits, dim=-1))
    router_probs = torch.softmax(logits, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    topk_selection_ms = time_cuda(lambda: torch.topk(logits, args.top_k, dim=-1))
    top_values, top_indices = torch.topk(logits, args.top_k, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    selected_softmax_gating_ms = time_cuda(lambda: torch.softmax(top_values, dim=-1))
    gates = torch.softmax(top_values, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    router_aux_loss_ms = time_cuda(
        lambda: nano_aux_loss_from_router(router_probs, top_indices, args.n_experts),
    )
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

    expert_path_ms = time_cuda(lambda: megablocks_expert_dispatch(layer, routing))
    expert_path_out = megablocks_expert_dispatch(layer, routing)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    output_layout_ms = time_cuda(lambda: expert_path_out.transpose(0, 1).contiguous())
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def whole_moe_layer_replay():
        replay_x_mb = x.transpose(0, 1).contiguous()
        replay_flat_x = replay_x_mb.view(-1, replay_x_mb.shape[-1])
        replay_logits = layer.router.layer(replay_flat_x)
        replay_router_probs = torch.softmax(replay_logits, dim=-1)
        replay_top_values, replay_top_indices = torch.topk(replay_logits, args.top_k, dim=-1)
        replay_gates = torch.softmax(replay_top_values, dim=-1)
        replay_aux_loss = nano_aux_loss_from_router(
            replay_router_probs,
            replay_top_indices,
            args.n_experts,
        )
        replay_routing = MegaBlocksRouting(
            x_mb=replay_x_mb,
            router_probs=replay_router_probs,
            gates=replay_gates,
            indices=replay_top_indices,
            aux_loss=replay_aux_loss,
        )
        replay_out = megablocks_expert_dispatch(layer, replay_routing)
        return replay_out.transpose(0, 1).contiguous(), replay_aux_loss

    whole_moe_layer_replay_ms = time_cuda(whole_moe_layer_replay)

    tokens_per_expert = torch.bincount(routing.indices.flatten().to(torch.long), minlength=args.n_experts)
    if args.megablocks_layer == "moe":
        expert_capacity = layer.experts.expert_capacity(args.batch_size * args.seq_len)
        if expert_capacity == 0:
            expert_capacity = int(torch.max(tokens_per_expert).item())
        op_path = "standard_moe_binned"
    elif args.megablocks_layer == "dmoe":
        expert_capacity = 0
        op_path = "dmoe_grouped"
    else:
        raise RuntimeError(f"Unsupported MegaBlocks layer for MoE op profiling: {args.megablocks_layer}")

    disjoint_replay_sum_ms = (
        input_layout_ms
        + router_projection_ms
        + router_full_softmax_ms
        + topk_selection_ms
        + selected_softmax_gating_ms
        + router_aux_loss_ms
        + expert_path_ms
        + output_layout_ms
    )
    replay_sum_minus_whole_ms = disjoint_replay_sum_ms - whole_moe_layer_replay_ms

    return {
        "moe_op_profile": True,
        "moe_op_profile_scope": "moe_layer_disjoint_replay",
        "moe_op_profile_warmup": warmup,
        "moe_op_profile_iters": iters,
        "moe_op_path": op_path,
        "moe_op_expert_capacity": int(expert_capacity),
        "moe_op_input_layout_to_megablocks_ms": float(input_layout_ms),
        "moe_op_router_projection_matmul_ms": float(router_projection_ms),
        "moe_op_router_full_softmax_ms": float(router_full_softmax_ms),
        "moe_op_topk_selection_ms": float(topk_selection_ms),
        "moe_op_selected_softmax_gating_ms": float(selected_softmax_gating_ms),
        "moe_op_router_aux_loss_ms": float(router_aux_loss_ms),
        "moe_op_expert_path_dispatch_compute_combine_ms": float(expert_path_ms),
        "moe_op_expert_block_dispatch_compute_combine_ms": float(expert_path_ms),
        "moe_op_output_layout_to_nano_ms": float(output_layout_ms),
        "moe_op_disjoint_replay_sum_ms": float(disjoint_replay_sum_ms),
        "moe_op_whole_moe_layer_replay_ms": float(whole_moe_layer_replay_ms),
        "moe_op_replay_sum_minus_whole_ms": float(replay_sum_minus_whole_ms),
        "moe_op_main_replay_sum_ms": float(disjoint_replay_sum_ms),
        "moe_op_profile_note": (
            "Disjoint diagnostic replay. Components are non-overlapping logical "
            "MoE blocks and their sum is reported, but split replay is not the "
            "authoritative production latency. Use mean_forward_ms for whole-layer "
            "latency. expert_path is one real MegaBlocks dispatch/sort/gather, "
            "expert MLP, and weighted scatter/combine unit. Gate multiply is folded "
            "into weighted scatter/combine. Lower-level gather/MLP/scatter timings "
            "require --phase-profile."
        ),
    }
