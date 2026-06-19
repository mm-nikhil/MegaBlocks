"""Correctness checks for Nano reference versus MegaBlocks adapter output."""

from __future__ import annotations

from typing import Optional

import torch

from moe_profile.megablocks_adapter import MegaBlocksForward


def router_check_metrics(
    *,
    reference_indices: torch.Tensor,
    reference_gates: torch.Tensor,
    megablocks_indices: torch.Tensor,
    megablocks_gates: torch.Tensor,
    output_diff: Optional[torch.Tensor],
    outlier_abs_threshold: float,
) -> dict[str, object]:
    """Compare router choices and gates.

    Position-wise index mismatch is recorded, but expert-set mismatch is the
    semantic signal because a top-k order swap does not change the weighted set.
    """

    ref_idx = reference_indices.detach().cpu()
    mb_idx = megablocks_indices.detach().cpu()
    idx_mismatch = ref_idx != mb_idx
    expert_set_mismatch = (
        torch.sort(ref_idx, dim=-1).values != torch.sort(mb_idx, dim=-1).values
    )
    token_mismatch = expert_set_mismatch.any(dim=-1)

    ref_gates = reference_gates.detach().float().cpu()
    mb_gates = megablocks_gates.detach().float().cpu()
    gate_diff = (ref_gates - mb_gates).abs()
    gate_scale = ref_gates.abs().max().clamp_min(1e-12)

    gate_matches = ref_idx.unsqueeze(-1) == mb_idx.unsqueeze(-2)
    common_gate_mask = gate_matches.any(dim=-1)
    aligned_mb_gates = (gate_matches.float() * mb_gates.unsqueeze(-2)).sum(dim=-1)
    aligned_gate_diff = (ref_gates - aligned_mb_gates).abs()
    if common_gate_mask.any().item():
        common_aligned_gate_diff = aligned_gate_diff[common_gate_mask]
        aligned_gate_max_abs = float(common_aligned_gate_diff.max().item())
        aligned_gate_mean_abs = float(common_aligned_gate_diff.mean().item())
    else:
        aligned_gate_max_abs = 0.0
        aligned_gate_mean_abs = 0.0

    metrics: dict[str, object] = {
        "router_indices_equal": bool(not idx_mismatch.any().item()),
        "router_index_mismatch_count": int(idx_mismatch.sum().item()),
        "router_index_mismatch_fraction": float(idx_mismatch.float().mean().item()),
        "router_token_mismatch_count": int(token_mismatch.sum().item()),
        "router_token_mismatch_fraction": float(token_mismatch.float().mean().item()),
        "router_expert_set_mismatch_count": int(token_mismatch.sum().item()),
        "router_expert_set_mismatch_fraction": float(token_mismatch.float().mean().item()),
        "router_gate_max_abs": float(gate_diff.max().item()),
        "router_gate_mean_abs": float(gate_diff.mean().item()),
        "router_gate_max_rel": float((gate_diff.max() / gate_scale).item()),
        "router_gate_aligned_max_abs": aligned_gate_max_abs,
        "router_gate_aligned_mean_abs": aligned_gate_mean_abs,
    }

    if output_diff is None:
        return metrics

    diff_cpu = output_diff.detach().cpu()
    max_abs = float(diff_cpu.max().item())
    flat_index = int(diff_cpu.reshape(-1).argmax().item())
    _, seq_len, d_model = diff_cpu.shape
    token_index = flat_index // d_model
    batch_index = token_index // seq_len
    seq_index = token_index % seq_len
    max_error_token_router_mismatch = bool(token_mismatch[batch_index, seq_index].item())
    max_error_token_gate_max_abs = float(aligned_gate_diff[batch_index, seq_index].max().item())
    token_abs = diff_cpu.amax(dim=-1)
    outlier_tokens = token_abs > outlier_abs_threshold
    outlier_elements = diff_cpu > outlier_abs_threshold
    if token_mismatch.any().item():
        max_abs_on_router_mismatch_tokens = float(token_abs[token_mismatch].max().item())
    else:
        max_abs_on_router_mismatch_tokens = 0.0
    router_match_mask = ~token_mismatch
    if router_match_mask.any().item():
        max_abs_on_router_match_tokens = float(token_abs[router_match_mask].max().item())
    else:
        max_abs_on_router_match_tokens = 0.0

    if max_abs <= outlier_abs_threshold:
        diagnosis = "within_threshold"
    elif max_error_token_router_mismatch:
        diagnosis = "router_choice_flip_at_max_error"
    elif token_mismatch.any().item():
        diagnosis = "router_choice_flips_elsewhere"
    else:
        diagnosis = "numeric_or_expert_path"

    metrics.update({
        "max_error_batch_index": batch_index,
        "max_error_seq_index": seq_index,
        "max_error_hidden_index": flat_index % d_model,
        "max_error_token_router_mismatch": max_error_token_router_mismatch,
        "max_error_token_gate_max_abs": max_error_token_gate_max_abs,
        "outlier_abs_threshold": outlier_abs_threshold,
        "output_outlier_token_count": int(outlier_tokens.sum().item()),
        "output_outlier_token_fraction": float(outlier_tokens.float().mean().item()),
        "output_outlier_element_count": int(outlier_elements.sum().item()),
        "output_outlier_element_fraction": float(outlier_elements.float().mean().item()),
        "max_abs_on_router_match_tokens": max_abs_on_router_match_tokens,
        "max_abs_on_router_mismatch_tokens": max_abs_on_router_mismatch_tokens,
        "outlier_diagnosis": diagnosis,
    })
    return metrics


def compare_moe_outputs(
    *,
    reference_for_checks,
    actual_forward: MegaBlocksForward,
    outlier_abs_threshold: float,
) -> dict[str, object]:
    """Compare full MoE output, aux loss, router choices, and gates."""

    expected = reference_for_checks.output.float()
    actual = actual_forward.output.float()
    diff = (expected - actual).abs()
    ref_scale = expected.abs().max().clamp_min(1e-12)
    aux_diff = (reference_for_checks.aux_loss.float() - actual_forward.aux_loss.float()).abs()
    metrics: dict[str, object] = {
        "max_abs_vs_reference": diff.max().item(),
        "mean_abs_vs_reference": diff.mean().item(),
        "max_rel_vs_reference": (diff.max() / ref_scale).item(),
        "max_abs_reference": expected.abs().max().item(),
        "aux_loss_reference": reference_for_checks.aux_loss.float().item(),
        "aux_loss_actual": actual_forward.aux_loss.float().item(),
        "aux_loss_abs_diff": aux_diff.item(),
    }
    metrics.update(router_check_metrics(
        reference_indices=reference_for_checks.indices,
        reference_gates=reference_for_checks.gates,
        megablocks_indices=actual_forward.indices,
        megablocks_gates=actual_forward.gates,
        output_diff=diff,
        outlier_abs_threshold=outlier_abs_threshold,
    ))
    metrics["correctness_passed"] = bool(
        metrics["max_abs_vs_reference"] <= outlier_abs_threshold
        and metrics["aux_loss_abs_diff"] <= outlier_abs_threshold
        and metrics["router_expert_set_mismatch_count"] == 0
    )
    return metrics


def print_check_summary(metrics: dict[str, object], *, verbose: bool) -> None:
    """Print the compact correctness summary used during sweep runs."""

    status = "passed" if metrics["correctness_passed"] else "failed"
    print(f"correctness: {status}")
    print(f"output_max_abs_error: {metrics['max_abs_vs_reference']:.6g}")
    print(f"output_mean_abs_error: {metrics['mean_abs_vs_reference']:.6g}")
    print(f"aux_loss_abs_error: {metrics['aux_loss_abs_diff']:.6g}")
    print(f"router_expert_set_mismatches: {metrics['router_expert_set_mismatch_count']}")
    print(f"router_gate_max_abs_error: {metrics['router_gate_max_abs']:.6g}")
    if not verbose:
        return

    print("verbose_check_metrics:")
    for key, value in metrics.items():
        if key in {
            "correctness_passed",
            "max_abs_vs_reference",
            "mean_abs_vs_reference",
            "aux_loss_abs_diff",
            "router_expert_set_mismatch_count",
            "router_gate_max_abs",
        }:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")

