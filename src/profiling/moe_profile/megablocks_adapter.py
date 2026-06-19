"""Nano-compatible adapter around MegaBlocks MoE and dMoE layers.

The benchmark keeps NanoJAX MoE math fixed: router probabilities, selected
experts, and selected-expert gates are computed with NanoJAX semantics, then
MegaBlocks performs sparse dispatch, expert MLP execution, and weighted combine.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from nano_moe_torch import NanoMoEWeights
from moe_profile.config import activation_fn_from_name
from moe_profile.weights import make_synthetic_glu_up_weight


class NanoMoEBiasedBatchedMLP(torch.nn.Module):
    """Bias-aware expert MLP for MegaBlocks' standard padded MoE layout."""

    def __init__(self, weights: NanoMoEWeights):
        super().__init__()
        self.w1 = torch.nn.Parameter(weights.w1.contiguous())
        self.b1 = torch.nn.Parameter(weights.b1.contiguous())
        self.w2 = torch.nn.Parameter(weights.w2.contiguous())
        self.b2 = torch.nn.Parameter(weights.b2.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(x, self.w1)
        x = x + self.b1[:, None, :]
        x = F.gelu(x, approximate="tanh")
        x = torch.bmm(x, self.w2)
        return x + self.b2[:, None, :]


class NanoMoEBiasedGroupedMLP(torch.nn.Module):
    """Bias-aware expert MLP for MegaBlocks dMoE grouped layout."""

    def __init__(self, weights: NanoMoEWeights):
        super().__init__()
        self.n_experts = weights.n_experts
        self.d_model = weights.d_model
        self.d_ff = weights.d_ff
        self.w1 = torch.nn.Parameter(
            weights.w1.transpose(1, 2).contiguous().view(-1, weights.d_model),
        )
        self.b1 = torch.nn.Parameter(weights.b1.contiguous())
        self.w2 = torch.nn.Parameter(weights.w2.contiguous().view(-1, weights.d_model))
        self.b2 = torch.nn.Parameter(weights.b2.contiguous())

    def _expert_ids(
        self,
        tokens_per_expert: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        experts = torch.arange(self.n_experts, device=tokens_per_expert.device, dtype=torch.long)
        return torch.repeat_interleave(
            experts,
            tokens_per_expert.to(torch.long),
            output_size=total_rows,
        )

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        from megablocks import grouped_gemm_util as gg

        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        expert_ids = self._expert_ids(tokens_per_expert, x.shape[0])
        w1 = self.w1.view(self.n_experts, self.d_ff, self.d_model)
        w2 = self.w2.view(self.n_experts, self.d_ff, self.d_model)

        assert gg.ops is not None
        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = x + self.b1.index_select(0, expert_ids)
        x = F.gelu(x, approximate="tanh")
        x = gg.ops.gmm(x, w2, batch_sizes)
        return x + self.b2.index_select(0, expert_ids)


class SyntheticGLUBatchedMLP(torch.nn.Module):
    """GLU expert MLP for standard MoE's padded expert layout."""

    def __init__(
        self,
        weights: NanoMoEWeights,
        v1: torch.Tensor,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.w_gate = torch.nn.Parameter(weights.w1.contiguous())
        self.w_up = torch.nn.Parameter(v1.contiguous())
        self.w_down = torch.nn.Parameter(weights.w2.contiguous())
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.bmm(x, self.w_gate)
        up = torch.bmm(x, self.w_up)
        return torch.bmm(self.activation_fn(gate) * up, self.w_down)


@dataclass(frozen=True)
class MegaBlocksForward:
    """Nano-layout output and diagnostics returned by the adapter."""

    output: torch.Tensor
    aux_loss: torch.Tensor
    router_probs: torch.Tensor
    gates: torch.Tensor
    indices: torch.Tensor
    tokens_per_expert: torch.Tensor


@dataclass(frozen=True)
class MegaBlocksRouting:
    """Prepared Nano-compatible routing tensors consumed by MegaBlocks experts."""

    x_mb: torch.Tensor
    router_probs: torch.Tensor
    gates: torch.Tensor
    indices: torch.Tensor
    aux_loss: torch.Tensor


def require_megablocks_runtime() -> None:
    """Fail early when MegaBlocks Python imports exist but CUDA ops are missing."""

    if importlib.util.find_spec("megablocks_ops") is None:
        raise RuntimeError(
            "MegaBlocks is importable, but megablocks_ops is not built. "
            "Install a CUDA toolkit with nvcc, rebuild grouped_gemm if using dMoE, "
            "then reinstall MegaBlocks from the local checkout."
        )


def build_megablocks_layer(
    args: argparse.Namespace,
    weights: NanoMoEWeights,
    model_shape: dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Build a MegaBlocks layer and copy Nano-compatible weights into it."""

    require_megablocks_runtime()
    if args.megablocks_layer == "dmoe" and dtype != torch.bfloat16:
        raise RuntimeError(
            "MegaBlocks dMoE uses the grouped_gemm extension in this checkout, "
            "and that extension currently requires bfloat16 inputs. Use "
            "--dtype bfloat16 for --megablocks-layer dmoe."
        )
    if weights.max_abs_bias() != 0.0 and not args.use_expert_biases and not args.allow_bias_mismatch:
        raise RuntimeError(
            "Nano-MoE-JAX experts include per-expert Dense biases, but MegaBlocks "
            "expert MLPs are bias-free. Use --use-expert-biases for the standard "
            "MoE bias-aware adapter, --zero-expert-biases for a biasless benchmark, "
            "or --allow-bias-mismatch to time a known non-equivalent layer."
        )

    from megablocks.layers.arguments import Arguments
    from megablocks.layers.dmoe import dMoE
    from megablocks.layers.moe import MoE

    expert_type = str(model_shape.get("expert_type", "ffn") or "ffn")
    activation = str(model_shape.get("activation", "gelu_tanh") or "gelu_tanh")
    shared_experts = int(model_shape.get("num_shared_experts", 0) or 0)
    shared_hidden = int(model_shape.get("shared_expert_intermediate_size", 0) or 0)
    if expert_type not in {"ffn", "glu"}:
        raise RuntimeError(f"Unsupported expert_type={expert_type!r}.")
    if args.use_expert_biases and expert_type != "ffn":
        raise RuntimeError("--use-expert-biases is only implemented for Nano FFN adapters.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    mb_args = Arguments(
        hidden_size=args.d_model,
        ffn_hidden_size=args.d_ff,
        num_layers=1,
        bias=False,
        return_bias=False,
        activation_fn=activation_fn_from_name(activation),
        moe_num_experts=args.n_experts,
        moe_top_k=args.top_k,
        moe_capacity_factor=0,
        moe_normalize_expert_weights=1,
        moe_loss_weight=0.0,
        memory_optimized_mlp=False,
        mlp_type="glu" if expert_type == "glu" else "mlp",
        mlp_impl="grouped",
        shared_expert=shared_experts > 0,
        shared_expert_hidden_size=shared_hidden or None,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        device=device,
    )

    layer = dMoE(mb_args) if args.megablocks_layer == "dmoe" else MoE(mb_args)
    layer.eval()

    has_nonzero_bias = weights.max_abs_bias() != 0.0
    with torch.no_grad():
        layer.router.layer.weight.copy_(weights.router_kernel.t().contiguous())
        if args.use_expert_biases and has_nonzero_bias:
            # Only replace the stock MLP when Nano biases are nonzero.
            if args.megablocks_layer == "dmoe":
                layer.experts.mlp = NanoMoEBiasedGroupedMLP(weights)
            else:
                layer.experts.mlp = NanoMoEBiasedBatchedMLP(weights)
            layer.experts.mlp.eval()
        elif expert_type == "glu":
            glu_v1 = make_synthetic_glu_up_weight(args, dtype, device)
            activation_fn = activation_fn_from_name(activation)
            if args.megablocks_layer == "moe":
                # Standard MoE expects a padded batched expert module.
                layer.experts.mlp = SyntheticGLUBatchedMLP(weights, glu_v1, activation_fn)
                layer.experts.mlp.eval()
            else:
                layer.experts.mlp.w1.view(args.n_experts, args.d_ff, args.d_model).copy_(
                    weights.w1.transpose(1, 2).contiguous(),
                )
                layer.experts.mlp.v1.view(args.n_experts, args.d_ff, args.d_model).copy_(
                    glu_v1.transpose(1, 2).contiguous(),
                )
                layer.experts.mlp.w2.view(args.n_experts, args.d_ff, args.d_model).copy_(
                    weights.w2.contiguous(),
                )
        elif args.megablocks_layer == "dmoe":
            layer.experts.mlp.w1.view(args.n_experts, args.d_ff, args.d_model).copy_(
                weights.w1.transpose(1, 2).contiguous(),
            )
            layer.experts.mlp.w2.view(args.n_experts, args.d_ff, args.d_model).copy_(
                weights.w2.contiguous(),
            )
        else:
            layer.experts.mlp.w1.copy_(weights.w1.contiguous())
            layer.experts.mlp.w2.copy_(weights.w2.contiguous())
    return layer


def nano_aux_loss_from_router(router_probs: torch.Tensor, top_indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """NanoMoE Switch-style load-balancing penalty from router probabilities."""

    top1 = top_indices[:, 0]
    dispatch_mask = F.one_hot(top1, num_classes=n_experts).to(router_probs.dtype)
    token_fraction = dispatch_mask.mean(dim=0)
    prob_mean = router_probs.mean(dim=0)
    return n_experts * torch.sum(token_fraction * prob_mean)


def dense_glu_reference_forward(
    x: torch.Tensor,
    weights: NanoMoEWeights,
    v1: torch.Tensor,
    *,
    top_k: int,
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Dense all-expert GLU reference for synthetic shape runs."""

    router_logits = torch.matmul(x, weights.router_kernel)
    top_k_values, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
    gates = torch.softmax(top_k_values, dim=-1)

    gate_proj = torch.einsum("btd,edh->ebth", x, weights.w1)
    up_proj = torch.einsum("btd,edh->ebth", x, v1)
    hidden = activation_fn(gate_proj) * up_proj
    expert_outputs = torch.einsum("ebth,ehd->ebtd", hidden, weights.w2)

    batch_size, seq_len, _ = x.shape
    batch_idx = torch.arange(batch_size, device=x.device)[:, None, None]
    seq_idx = torch.arange(seq_len, device=x.device)[None, :, None]
    selected = expert_outputs[top_k_indices, batch_idx, seq_idx, :]
    return torch.sum(gates[..., None] * selected, dim=2)


def megablocks_prepare_routing(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    n_experts: int,
    top_k: int,
) -> MegaBlocksRouting:
    """Compute Nano-compatible router logits, top-k choices, gates, and aux loss.

    This is not the stock MegaBlocks router call. It intentionally exposes the
    NanoJAX routing convention so correctness checks can compare router indices,
    gates, auxiliary loss, and final output against the PyTorch reference.
    """

    x_mb = x.transpose(0, 1).contiguous()
    flat_x = x_mb.view(-1, x_mb.shape[-1])
    logits = layer.router.layer(flat_x)
    router_probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, top_k, dim=-1)
    gates = torch.softmax(top_values, dim=-1)
    aux_loss = nano_aux_loss_from_router(router_probs, top_indices, n_experts)
    return MegaBlocksRouting(
        x_mb=x_mb,
        router_probs=router_probs,
        gates=gates,
        indices=top_indices,
        aux_loss=aux_loss,
    )


def megablocks_expert_dispatch(layer: torch.nn.Module, routing: MegaBlocksRouting) -> torch.Tensor:
    """Run the prepared MegaBlocks expert path.

    This includes the MegaBlocks dispatch/sort/gather work, expert MLP execution,
    weighted combine/scatter, and any configured shared-expert combine.
    """

    out = layer.experts(routing.x_mb, routing.router_probs, routing.gates, routing.indices)
    if isinstance(out, tuple):
        out = out[0]
    if getattr(layer, "shared_expert", None) is not None:
        shared_expert_out = layer.shared_expert(routing.x_mb)
        out = layer.shared_expert.add_experts_sharedexpert(shared_expert_out, out)
    return out


def megablocks_forward(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    n_experts: int,
    top_k: int,
    collect_diagnostics: bool = False,
) -> MegaBlocksForward:
    """Run the full Nano-compatible MoE layer through MegaBlocks experts."""

    routing = megablocks_prepare_routing(layer, x, n_experts=n_experts, top_k=top_k)
    out = megablocks_expert_dispatch(layer, routing)

    if collect_diagnostics:
        gates_btd = routing.gates.view(seq_len, batch_size, top_k).transpose(0, 1).contiguous()
        indices_btd = routing.indices.view(seq_len, batch_size, top_k).transpose(0, 1).contiguous()
        router_probs_out = routing.router_probs.view(seq_len, batch_size, n_experts).transpose(0, 1).contiguous()
        tokens_per_expert = torch.bincount(routing.indices.flatten().to(torch.long), minlength=n_experts)
    else:
        # Diagnostics are collected once after timing so small-N measurements are
        # not dominated by histogram and layout bookkeeping.
        gates_btd = routing.gates
        indices_btd = routing.indices
        router_probs_out = routing.router_probs
        tokens_per_expert = torch.empty(0, dtype=torch.int64, device=x.device)

    return MegaBlocksForward(
        output=out.transpose(0, 1).contiguous(),
        aux_loss=routing.aux_loss,
        router_probs=router_probs_out,
        gates=gates_btd,
        indices=indices_btd,
        tokens_per_expert=tokens_per_expert,
    )
