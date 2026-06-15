"""Profile the NanoMoE MoE boundary.

Backends:
    reference:
        Exact PyTorch implementation of Nano-MoE-JAX semantics. This computes all
        experts and is useful for correctness and shape sanity checks.

    megablocks:
        MegaBlocks MoE/dMoE adapter. This requires compiled MegaBlocks kernels.
        It refuses nonzero Nano expert biases by default because MegaBlocks expert
        MLPs do not model the per-expert Dense biases used by Nano-MoE-JAX.

Example:
    python src/profiling/profile_moe_layer.py --backend reference --device cuda

After installing a CUDA toolkit and rebuilding MegaBlocks/grouped_gemm:
    python src/profiling/profile_moe_layer.py --backend megablocks --zero-expert-biases
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from nano_moe_torch import NanoMoEWeights, nano_moe_forward


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero-expert-biases", action="store_true")
    parser.add_argument("--allow-bias-mismatch", action="store_true")
    parser.add_argument("--check-output", action="store_true")
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_random_weights(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> NanoMoEWeights:
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


def make_input(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, generator=generator)
    return x.to(device=device, dtype=dtype)


def cuda_time_ms(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    if device.type != "cuda":
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return 1000.0 * (time.perf_counter() - start) / iters

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


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


def build_megablocks_layer(
    args: argparse.Namespace,
    weights: NanoMoEWeights,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    require_megablocks_runtime()
    if weights.max_abs_bias() != 0.0 and not args.allow_bias_mismatch:
        raise RuntimeError(
            "Nano-MoE-JAX experts include per-expert Dense biases, but MegaBlocks "
            "expert MLPs are bias-free. Use --zero-expert-biases for a "
            "MegaBlocks-compatible benchmark or --allow-bias-mismatch to time a "
            "known non-equivalent layer."
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

    with torch.no_grad():
        layer.router.layer.weight.copy_(weights.router_kernel.t().contiguous())
        if args.megablocks_layer == "dmoe":
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


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    weights = make_random_weights(args, dtype, device)
    x = make_input(args, dtype, device)
    check_metrics = {}

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
        x_mb = x.transpose(0, 1).contiguous()

        def run():
            out = layer(x_mb)
            if isinstance(out, tuple):
                out = out[0]
            return out.transpose(0, 1)

        if args.check_output:
            with torch.no_grad():
                expected = nano_moe_forward(
                    x,
                    weights,
                    top_k=args.top_k,
                    deterministic=True,
                    dropout_p=0.0,
                ).output.float()
                actual = run().float()
                diff = (expected - actual).abs()
                ref_scale = expected.abs().max().clamp_min(1e-12)
                check_metrics = {
                    "max_abs_vs_reference": diff.max().item(),
                    "mean_abs_vs_reference": diff.mean().item(),
                    "max_rel_vs_reference": (diff.max() / ref_scale).item(),
                    "max_abs_reference": expected.abs().max().item(),
                }
                for key, value in check_metrics.items():
                    print(f"{key}: {value:.6g}")

    with torch.inference_mode():
        mean_ms = cuda_time_ms(run, warmup=args.warmup, iters=args.iters, device=device)

    tokens = args.batch_size * args.seq_len
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
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
        "zero_expert_biases": args.zero_expert_biases,
        "allow_bias_mismatch": args.allow_bias_mismatch,
        "check_output": args.check_output,
        "mean_forward_ms": mean_ms,
        "torch": torch.__version__,
        **gpu_metadata(device),
        **check_metrics,
    }

    print(f"backend: {args.backend}")
    if args.backend == "megablocks":
        print(f"megablocks_layer: {args.megablocks_layer}")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"shape: batch={args.batch_size} seq={args.seq_len} tokens={tokens} d_model={args.d_model}")
    print(f"experts: n={args.n_experts} top_k={args.top_k} d_ff={args.d_ff}")
    print(f"mean_forward_ms: {mean_ms:.4f}")
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
