"""Profile the Nano-compatible MegaBlocks MoE adapter.

Backends:
    reference:
        Exact PyTorch implementation of Nano-MoE-JAX semantics. This computes all
        experts and is useful for correctness and shape sanity checks.

    megablocks:
        MegaBlocks MoE/dMoE adapter. This requires compiled MegaBlocks kernels.
        It refuses nonzero Nano expert biases by default. Use --use-expert-biases
        to install a Nano-compatible expert MLP wrapper for standard MoE or dMoE.

Example:
    python src/profiling/profile_moe_layer.py --backend reference --device cuda

After installing a CUDA toolkit and rebuilding MegaBlocks/grouped_gemm:
    python src/profiling/profile_moe_layer.py --backend megablocks --megablocks-layer moe --use-expert-biases
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nano_moe_torch import NanoMoEWeights, from_flax_moe_params, nano_moe_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("reference", "megablocks"), default="reference")
    parser.add_argument("--megablocks-layer", choices=("moe", "dmoe"), default="dmoe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing-scope",
        choices=("auto", "megablocks_core", "adapter_boundary"),
        default="auto",
        help=(
            "auto times MegaBlocks dispatch/expert/combine for the megablocks backend "
            "and the full reference boundary for the reference backend."
        ),
    )
    parser.add_argument("--weight-source", choices=("nano_jax_init", "synthetic"), default="nano_jax_init")
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--zero-expert-biases", action="store_true")
    parser.add_argument("--use-expert-biases", action="store_true")
    parser.add_argument("--allow-bias-mismatch", action="store_true")
    parser.add_argument("--check-output", action="store_true")
    parser.add_argument("--verbose-checks", action="store_true")
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_synthetic_weights(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> NanoMoEWeights:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(*shape: int) -> torch.Tensor:
        return (0.02 * torch.randn(*shape, generator=generator)).to(device=device, dtype=dtype)

    b1 = torch.zeros(args.n_experts, args.d_ff, device=device, dtype=dtype)
    b2 = torch.zeros(args.n_experts, args.d_model, device=device, dtype=dtype)
    if not args.zero_expert_biases:
        b1 = randn(args.n_experts, args.d_ff)
        b2 = randn(args.n_experts, args.d_model)

    return NanoMoEWeights(
        router_kernel=randn(args.d_model, args.n_experts),
        w1=randn(args.n_experts, args.d_model, args.d_ff),
        b1=b1,
        w2=randn(args.n_experts, args.d_ff, args.d_model),
        b2=b2,
    )


def zero_expert_biases(weights: NanoMoEWeights) -> NanoMoEWeights:
    return NanoMoEWeights(
        router_kernel=weights.router_kernel,
        w1=weights.w1,
        b1=torch.zeros_like(weights.b1),
        w2=weights.w2,
        b2=torch.zeros_like(weights.b2),
    )


def make_nano_jax_initialized_weights(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> NanoMoEWeights:
    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise RuntimeError(f"Nano-MoE-JAX checkout not found: {nano_dir}")

    # JAX is used only to reproduce Nano-MoE-JAX initializers. Keep that tiny
    # setup on CPU so profiling runs do not allocate JAX GPU memory or warn when
    # CUDA-enabled jaxlib is absent.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(nano_dir))
    try:
        import jax
        import jax.numpy as jnp
        from nano_moe.config import NanoMoEConfig
        from nano_moe.layers import MoELayer
    except ImportError as exc:
        raise RuntimeError(
            "weight_source=nano_jax_init requires JAX, Flax, and the Nano-MoE-JAX checkout. "
            "Use --weight-source synthetic for a dependency-light synthetic benchmark."
        ) from exc

    config = NanoMoEConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        top_k=args.top_k,
        dropout_rate=0.0,
    )
    rng = jax.random.PRNGKey(args.seed)
    init_x = jnp.zeros((args.batch_size, args.seq_len, args.d_model), dtype=jnp.float32)
    params = MoELayer(config=config).init(rng, init_x, deterministic=True)["params"]
    return from_flax_moe_params(params, device=device, dtype=dtype)


def make_weights(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> NanoMoEWeights:
    if args.weight_source == "nano_jax_init":
        weights = make_nano_jax_initialized_weights(args, dtype, device)
    else:
        weights = make_synthetic_weights(args, dtype, device)

    if args.zero_expert_biases:
        weights = zero_expert_biases(weights)
    return weights


def make_input(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, generator=generator)
    return x.to(device=device, dtype=dtype)


def cuda_time_ms(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    after_warmup: Optional[Callable[[], None]] = None,
) -> float:
    if device.type != "cuda":
        for _ in range(warmup):
            fn()
        if after_warmup is not None:
            after_warmup()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return 1000.0 * (time.perf_counter() - start) / iters

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    if after_warmup is not None:
        after_warmup()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def measure_forward(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    trials: int,
    device: torch.device,
) -> dict[str, float | int]:
    trial_ms = []
    peak_allocated = 0
    peak_reserved = 0
    peak_allocated_delta = 0
    baseline_allocated_max = 0

    for _ in range(trials):
        baseline_allocated = 0

        def after_warmup() -> None:
            nonlocal baseline_allocated, baseline_allocated_max
            if device.type != "cuda":
                return
            torch.cuda.synchronize(device)
            # Reset after warmup so lazy kernel setup and allocator priming do not
            # get reported as steady-state forward memory.
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_allocated_max = max(baseline_allocated_max, baseline_allocated)
            torch.cuda.reset_peak_memory_stats(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        trial_ms.append(cuda_time_ms(
            fn,
            warmup=warmup,
            iters=iters,
            device=device,
            after_warmup=after_warmup,
        ))

        if device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            peak_allocated_delta = max(
                peak_allocated_delta,
                max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated),
            )

    ms_tensor = torch.tensor(trial_ms, dtype=torch.float64)
    std_ms = 0.0 if trials == 1 else float(ms_tensor.std(unbiased=False).item())
    return {
        "mean_forward_ms": float(ms_tensor.mean().item()),
        "std_forward_ms": std_ms,
        "min_forward_ms": float(ms_tensor.min().item()),
        "max_forward_ms": float(ms_tensor.max().item()),
        "trials": trials,
        "baseline_memory_allocated_bytes": int(baseline_allocated_max),
        "peak_memory_allocated_bytes": int(peak_allocated),
        "peak_memory_reserved_bytes": int(peak_reserved),
        "peak_memory_allocated_delta_bytes": int(peak_allocated_delta),
    }


def require_megablocks_runtime() -> None:
    if importlib.util.find_spec("megablocks_ops") is None:
        raise RuntimeError(
            "MegaBlocks is importable, but megablocks_ops is not built. "
            "Install a CUDA toolkit with nvcc, rebuild grouped_gemm if using dMoE, "
            "then reinstall MegaBlocks from the local checkout."
        )


def gpu_metadata(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_capability": list(torch.cuda.get_device_capability(index)),
        "torch_cuda": torch.version.cuda,
    }


class NanoMoEBiasedBatchedMLP(torch.nn.Module):
    """Bias-aware expert MLP for MegaBlocks' standard MoE layout.

    MegaBlocks standard MoE gathers tokens into a 3D tensor with shape
    ``(n_experts, expert_capacity, d_model)``. Its stock MLP applies two
    bias-free batched matmuls. Nano-MoE-JAX uses Dense biases, so this module
    preserves the same gathered layout while adding ``b1`` before GELU and
    ``b2`` after the down projection.
    """

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
    """Bias-aware expert MLP for MegaBlocks dMoE grouped layout.

    dMoE gathers all routed token/expert assignments into one 2D tensor sorted by
    expert, then runs grouped GEMMs using ``tokens_per_expert``. This wrapper
    preserves that path and injects Nano-MoE-JAX's per-expert Dense biases after
    the first grouped GEMM and after the second grouped GEMM.
    """

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


@dataclass(frozen=True)
class MegaBlocksForward:
    output: torch.Tensor
    aux_loss: torch.Tensor
    router_probs: torch.Tensor
    gates: torch.Tensor
    indices: torch.Tensor
    tokens_per_expert: torch.Tensor


@dataclass(frozen=True)
class MegaBlocksRouting:
    x_mb: torch.Tensor
    router_probs: torch.Tensor
    gates: torch.Tensor
    indices: torch.Tensor
    aux_loss: torch.Tensor


def build_megablocks_layer(
    args: argparse.Namespace,
    weights: NanoMoEWeights,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
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

    mb_args = Arguments(
        hidden_size=args.d_model,
        ffn_hidden_size=args.d_ff,
        num_layers=1,
        bias=False,
        return_bias=False,
        activation_fn=partial(F.gelu, approximate="tanh"),
        moe_num_experts=args.n_experts,
        moe_top_k=args.top_k,
        moe_capacity_factor=0,
        moe_normalize_expert_weights=1,
        moe_loss_weight=0.0,
        memory_optimized_mlp=False,
        mlp_type="mlp",
        mlp_impl="grouped",
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
            # Keep the stock MegaBlocks expert MLP when biases are zero. The
            # bias-aware replacement is only needed for trained/synthetic weights
            # with nonzero Nano Dense biases.
            if args.megablocks_layer == "dmoe":
                layer.experts.mlp = NanoMoEBiasedGroupedMLP(weights)
            else:
                layer.experts.mlp = NanoMoEBiasedBatchedMLP(weights)
            layer.experts.mlp.eval()
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
    top1 = top_indices[:, 0]
    dispatch_mask = F.one_hot(top1, num_classes=n_experts).to(router_probs.dtype)
    token_fraction = dispatch_mask.mean(dim=0)
    prob_mean = router_probs.mean(dim=0)
    return n_experts * torch.sum(token_fraction * prob_mean)


def megablocks_prepare_routing(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    n_experts: int,
    top_k: int,
) -> MegaBlocksRouting:
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
    # Public profiling inputs are Nano-style (B, T, D). MegaBlocks consumes
    # (T, B, D), so the adapter boundary includes both layout conversions.
    routing = megablocks_prepare_routing(layer, x, n_experts=n_experts, top_k=top_k)

    # MegaBlocks' stock router takes top-k over probabilities. Nano-MoE-JAX takes
    # top-k over logits, so we feed Nano-compatible assignments into MegaBlocks'
    # dispatch/expert/combine path instead of calling layer(x_mb) directly.
    out = megablocks_expert_dispatch(layer, routing)

    if collect_diagnostics:
        gates_btd = routing.gates.view(seq_len, batch_size, top_k).transpose(0, 1).contiguous()
        indices_btd = routing.indices.view(seq_len, batch_size, top_k).transpose(0, 1).contiguous()
        router_probs_out = routing.router_probs.view(seq_len, batch_size, n_experts).transpose(0, 1).contiguous()
        tokens_per_expert = torch.bincount(routing.indices.flatten().to(torch.long), minlength=n_experts)
    else:
        # Keep diagnostics outside timing unless adapter_boundary timing is requested.
        # Router diagnostics and expert-count histograms are collected once after
        # timing so they do not dominate nano-scale benchmark timings.
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


def router_check_metrics(
    *,
    reference_indices: torch.Tensor,
    reference_gates: torch.Tensor,
    megablocks_indices: torch.Tensor,
    megablocks_gates: torch.Tensor,
    output_diff: Optional[torch.Tensor],
    outlier_abs_threshold: float,
) -> dict[str, object]:
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


def tokens_per_expert_metrics(tokens_per_expert: Optional[torch.Tensor]) -> dict[str, float | int]:
    if tokens_per_expert is None:
        return {}
    counts = tokens_per_expert.detach().float().cpu()
    return {
        "tokens_per_expert_min": int(counts.min().item()),
        "tokens_per_expert_max": int(counts.max().item()),
        "tokens_per_expert_mean": float(counts.mean().item()),
        "tokens_per_expert_std": float(counts.std(unbiased=False).item()),
    }


def flops_metrics(
    args: argparse.Namespace,
    *,
    tokens_per_expert: Optional[torch.Tensor],
    mean_forward_ms: float,
) -> dict[str, float | int]:
    tokens = args.batch_size * args.seq_len
    router_flops = 2 * tokens * args.d_model * args.n_experts
    active_expert_flops = 4 * tokens * args.top_k * args.d_model * args.d_ff

    padded_expert_rows = 0
    if args.backend == "reference":
        backend_expert_flops = 4 * tokens * args.n_experts * args.d_model * args.d_ff
    elif args.megablocks_layer == "moe" and tokens_per_expert is not None:
        padded_expert_rows = int(args.n_experts * tokens_per_expert.max().item())
        backend_expert_flops = 4 * padded_expert_rows * args.d_model * args.d_ff
    else:
        backend_expert_flops = active_expert_flops

    backend_total_flops = router_flops + backend_expert_flops
    seconds = mean_forward_ms / 1000.0
    return {
        "router_flops": int(router_flops),
        "active_expert_flops": int(active_expert_flops),
        "backend_expert_flops_estimate": int(backend_expert_flops),
        "backend_total_flops_estimate": int(backend_total_flops),
        "megablocks_padded_expert_rows": padded_expert_rows,
        "tokens_per_second": float(tokens / seconds),
        "active_expert_tflops_per_second": float(active_expert_flops / seconds / 1e12),
        "backend_estimated_tflops_per_second": float(backend_total_flops / seconds / 1e12),
    }


def bias_semantics(args: argparse.Namespace) -> str:
    if args.use_expert_biases:
        return "matched_expert_biases"
    if args.zero_expert_biases:
        return "zero_expert_biases"
    if args.allow_bias_mismatch:
        return "intentional_bias_mismatch"
    return "nano_expert_biases"


def resolve_timing_scope(args: argparse.Namespace) -> str:
    if args.timing_scope != "auto":
        return args.timing_scope
    if args.backend == "megablocks":
        return "megablocks_core"
    return "adapter_boundary"


def main() -> None:
    args = parse_args()
    if args.zero_expert_biases and args.use_expert_biases:
        raise SystemExit("Use only one of --zero-expert-biases or --use-expert-biases.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1.")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    timing_scope = resolve_timing_scope(args)
    if timing_scope == "megablocks_core" and args.backend != "megablocks":
        raise SystemExit("--timing-scope megablocks_core is only valid with --backend megablocks.")
    weights = make_weights(args, dtype, device)
    x = make_input(args, dtype, device)
    check_metrics = {}
    tokens_per_expert = None

    if args.backend == "reference":
        def run():
            return nano_moe_forward(
                x,
                weights,
                top_k=args.top_k,
                deterministic=True,
                dropout_p=0.0,
            ).output
    else:
        layer = build_megablocks_layer(args, weights, dtype, device)
        if timing_scope == "megablocks_core":
            timed_routing = megablocks_prepare_routing(
                layer,
                x,
                n_experts=args.n_experts,
                top_k=args.top_k,
            )

            def run():
                return megablocks_expert_dispatch(layer, timed_routing)
        else:
            def run():
                return megablocks_forward(
                    layer,
                    x,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    n_experts=args.n_experts,
                    top_k=args.top_k,
                ).output

    with torch.inference_mode():
        timing_metrics = measure_forward(
            run,
            warmup=args.warmup,
            iters=args.iters,
            trials=args.trials,
            device=device,
        )

    if args.backend == "megablocks":
        with torch.inference_mode():
            mb_forward = megablocks_forward(
                layer,
                x,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                n_experts=args.n_experts,
                top_k=args.top_k,
                collect_diagnostics=True,
            )
        tokens_per_expert = mb_forward.tokens_per_expert

        if args.check_output:
            with torch.no_grad():
                reference_for_checks = nano_moe_forward(
                    x,
                    weights,
                    top_k=args.top_k,
                    deterministic=True,
                    dropout_p=0.0,
                )
                check_metrics = compare_moe_outputs(
                    reference_for_checks=reference_for_checks,
                    actual_forward=mb_forward,
                    outlier_abs_threshold=args.outlier_abs_threshold,
                )
                print_check_summary(check_metrics, verbose=args.verbose_checks)

    tokens = args.batch_size * args.seq_len
    flops = flops_metrics(
        args,
        tokens_per_expert=tokens_per_expert,
        mean_forward_ms=timing_metrics["mean_forward_ms"],
    )
    expert_count_metrics = tokens_per_expert_metrics(tokens_per_expert)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "backend_variant": (
            f"megablocks_{args.megablocks_layer}" if args.backend == "megablocks" else "reference"
        ),
        "backend": args.backend,
        "megablocks_layer": args.megablocks_layer if args.backend == "megablocks" else None,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens": tokens,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "warmup": args.warmup,
        "iters": args.iters,
        "trials": args.trials,
        "timing_scope": timing_scope,
        "weight_source": args.weight_source,
        "zero_expert_biases": args.zero_expert_biases,
        "use_expert_biases": args.use_expert_biases,
        "allow_bias_mismatch": args.allow_bias_mismatch,
        "bias_semantics": bias_semantics(args),
        "expert_bias_max_abs": weights.max_abs_bias(),
        "check_output": args.check_output,
        "verbose_checks": args.verbose_checks,
        "torch": torch.__version__,
        **gpu_metadata(device),
        **timing_metrics,
        **expert_count_metrics,
        **flops,
        **check_metrics,
    }

    print(f"backend: {args.backend}")
    if args.backend == "megablocks":
        print(f"megablocks_layer: {args.megablocks_layer}")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"shape: batch={args.batch_size} seq={args.seq_len} tokens={tokens} d_model={args.d_model}")
    print(f"experts: n={args.n_experts} top_k={args.top_k} d_ff={args.d_ff}")
    print(f"timing_scope: {timing_scope}")
    print(f"mean_forward_ms: {timing_metrics['mean_forward_ms']:.4f}")
    print(f"std_forward_ms: {timing_metrics['std_forward_ms']:.4f}")
    print(f"peak_memory_allocated_bytes: {timing_metrics['peak_memory_allocated_bytes']}")
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"wrote_jsonl: {args.jsonl_out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
