"""Profile one Nano-compatible MoE layer.

This file is intentionally a thin command-line entrypoint. The semantics,
MegaBlocks adapter, timing helpers, diagnostics, and metric definitions live in
``moe_profile/`` so sweep runners can compose the same building blocks.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from nano_moe_torch import nano_moe_forward
from moe_profile.config import activation_fn_from_name, load_model_shape, parse_dtype
from moe_profile.correctness import compare_moe_outputs, print_check_summary
from moe_profile.megablocks_adapter import (
    build_megablocks_layer,
    dense_glu_reference_forward,
    megablocks_expert_dispatch,
    megablocks_forward,
    megablocks_prepare_routing,
)
from moe_profile.metrics import bias_semantics, flops_metrics, tokens_per_expert_metrics
from moe_profile.op_profiles import measure_megablocks_moe_ops, measure_megablocks_phases
from moe_profile.runtime import gpu_metadata, measure_forward, memory_preflight
from moe_profile.terms import (
    TIMING_SCOPE_CHOICES,
    TIMING_SCOPE_EXPERT_PATH,
    TIMING_SCOPE_HELP,
    resolve_timing_scope,
    require_valid_timing_scope,
)
from moe_profile.weights import make_input, make_synthetic_glu_up_weight, make_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("reference", "megablocks"), default="reference")
    parser.add_argument("--megablocks-layer", choices=("moe", "dmoe"), default="dmoe")
    parser.add_argument("--batch-size", type=int, default=32, help="B in the Nano-layout input shape (B, T, D).")
    parser.add_argument("--seq-len", type=int, default=128, help="T in the Nano-layout input shape (B, T, D).")
    parser.add_argument("--d-model", type=int, default=128, help="Hidden width D for one MoE layer.")
    parser.add_argument("--d-ff", type=int, default=512, help="Expert hidden width H.")
    parser.add_argument("--n-experts", type=int, default=4, help="Number of routed experts E.")
    parser.add_argument("--top-k", type=int, default=2, help="Selected experts per token K.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
        help="Activation and weight dtype used by the profiled torch/MegaBlocks layer.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for timing, usually cuda.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warmup iterations before each trial.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations averaged within each trial.")
    parser.add_argument("--trials", type=int, default=1, help="Independent timing trials for mean/std reporting.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for synthetic inputs/weights.")
    parser.add_argument("--timing-scope", choices=TIMING_SCOPE_CHOICES, default="auto", help=TIMING_SCOPE_HELP)
    parser.add_argument(
        "--weight-source",
        choices=("nano_jax_init", "synthetic", "trained_nano_checkpoint"),
        default="nano_jax_init",
        help="Where MoE weights come from before conversion into torch/MegaBlocks tensors.",
    )
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/trained_nano_moe_checkpoint"))
    parser.add_argument("--checkpoint-block-index", type=int, default=0)
    parser.add_argument("--zero-expert-biases", action="store_true", help="Force Nano expert Dense biases to zero.")
    parser.add_argument(
        "--use-expert-biases",
        action="store_true",
        help="Install a bias-aware expert MLP adapter when Nano weights have nonzero expert biases.",
    )
    parser.add_argument(
        "--allow-bias-mismatch",
        action="store_true",
        help="Allow timing a known non-equivalent bias-free MegaBlocks path with nonzero Nano biases.",
    )
    parser.add_argument("--check-output", action="store_true", help="Compare MegaBlocks output against reference.")
    parser.add_argument("--verbose-checks", action="store_true", help="Print detailed correctness diagnostics.")
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--jsonl-out", type=Path, help="Append the profiler record to this JSONL file.")
    parser.add_argument("--label", default="", help="Stable run label used by sweep de-duplication.")
    parser.add_argument("--model-shape-name", default="", help="Optional shape catalog key.")
    parser.add_argument("--model-shapes-config", type=Path, default=Path("configs/moe_model_shapes.json"))
    parser.add_argument("--allow-shape-override", action="store_true")
    parser.add_argument(
        "--phase-profile",
        action="store_true",
        help="Collect lower-level MegaBlocks expert-path implementation phase timings.",
    )
    parser.add_argument("--phase-warmup", type=int, default=5)
    parser.add_argument("--phase-iters", type=int, default=20)
    parser.add_argument(
        "--moe-op-profile",
        action="store_true",
        help="Collect logical MoE-layer op timings: router, top-k, gating, expert path, combine/layout.",
    )
    parser.add_argument("--moe-op-warmup", type=int, default=5)
    parser.add_argument("--moe-op-iters", type=int, default=20)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap the measured loop in cudaProfilerStart/Stop for Nsight range capture.",
    )
    parser.add_argument("--skip-memory-preflight", action="store_true")
    parser.add_argument("--memory-preflight-fraction", type=float, default=0.90)
    parser.add_argument("--memory-preflight-safety-multiplier", type=float, default=1.35)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, timing_scope: str) -> None:
    """Validate cross-option constraints before expensive allocations."""

    if args.zero_expert_biases and args.use_expert_biases:
        raise SystemExit("Use only one of --zero-expert-biases or --use-expert-biases.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1.")
    require_valid_timing_scope(args, timing_scope)
    if args.phase_profile and args.backend != "megablocks":
        raise SystemExit("--phase-profile is only valid with --backend megablocks.")
    if args.moe_op_profile and args.backend != "megablocks":
        raise SystemExit("--moe-op-profile is only valid with --backend megablocks.")


def main() -> None:
    args = parse_args()
    timing_scope = resolve_timing_scope(args)
    validate_args(args, timing_scope)

    model_shape = load_model_shape(args)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    preflight_metrics = memory_preflight(
        args,
        model_shape=model_shape,
        dtype=dtype,
        device=device,
    )
    weights = make_weights(args, dtype, device)
    x = make_input(args, dtype, device)
    check_metrics = {}
    phase_metrics = {}
    moe_op_metrics = {}
    tokens_per_expert = None
    expert_type = str(model_shape.get("expert_type", "ffn") or "ffn")
    activation = str(model_shape.get("activation", "gelu_tanh") or "gelu_tanh")

    if args.backend == "reference":
        if expert_type == "glu":
            if args.weight_source != "synthetic":
                raise RuntimeError("Dense GLU reference is only supported with --weight-source synthetic.")
            glu_v1 = make_synthetic_glu_up_weight(args, dtype, device)
            activation_fn = activation_fn_from_name(activation)

            def run():
                return dense_glu_reference_forward(
                    x,
                    weights,
                    glu_v1,
                    top_k=args.top_k,
                    activation_fn=activation_fn,
                )
        else:
            def run():
                return nano_moe_forward(
                    x,
                    weights,
                    top_k=args.top_k,
                    deterministic=True,
                    dropout_p=0.0,
                ).output
    else:
        layer = build_megablocks_layer(args, weights, model_shape, dtype, device)
        if timing_scope == TIMING_SCOPE_EXPERT_PATH:
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
            cuda_profiler_range=args.cuda_profiler_range,
        )

    if args.backend == "megablocks":
        if args.phase_profile:
            with torch.inference_mode():
                phase_routing = megablocks_prepare_routing(
                    layer,
                    x,
                    n_experts=args.n_experts,
                    top_k=args.top_k,
                )
                phase_metrics = measure_megablocks_phases(
                    layer,
                    phase_routing,
                    args,
                    device=device,
                )

        if args.moe_op_profile:
            with torch.inference_mode():
                moe_op_metrics = measure_megablocks_moe_ops(
                    layer,
                    x,
                    args,
                    device=device,
                )

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
        model_shape=model_shape,
        tokens_per_expert=tokens_per_expert,
        mean_forward_ms=timing_metrics["mean_forward_ms"],
    )
    expert_count_metrics = tokens_per_expert_metrics(tokens_per_expert)
    model_shape_name = args.model_shape_name or "custom"
    simulation_level = str(model_shape.get("simulation_level", "custom"))
    shared_experts = int(model_shape.get("num_shared_experts", 0) or 0)
    shared_hidden = int(model_shape.get("shared_expert_intermediate_size", 0) or 0)
    max_position_embeddings = model_shape.get("max_position_embeddings")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model_shape_name": model_shape_name,
        "simulation_level": simulation_level,
        "backend_variant": (
            f"megablocks_{args.megablocks_layer}" if args.backend == "megablocks" else f"reference_dense_{expert_type}"
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
        "num_shared_experts": shared_experts,
        "shared_expert_intermediate_size": shared_hidden,
        "expert_type": expert_type,
        "activation": activation,
        "max_position_embeddings": max_position_embeddings,
        "warmup": args.warmup,
        "iters": args.iters,
        "trials": args.trials,
        "timing_scope": timing_scope,
        "weight_source": args.weight_source,
        "checkpoint_dir": str(args.checkpoint_dir) if args.weight_source == "trained_nano_checkpoint" else None,
        "checkpoint_block_index": (
            args.checkpoint_block_index if args.weight_source == "trained_nano_checkpoint" else None
        ),
        "zero_expert_biases": args.zero_expert_biases,
        "use_expert_biases": args.use_expert_biases,
        "allow_bias_mismatch": args.allow_bias_mismatch,
        "bias_semantics": bias_semantics(args),
        "expert_bias_max_abs": weights.max_abs_bias(),
        "check_output": args.check_output,
        "verbose_checks": args.verbose_checks,
        "torch": torch.__version__,
        **gpu_metadata(device),
        **preflight_metrics,
        **timing_metrics,
        **expert_count_metrics,
        **flops,
        **phase_metrics,
        **moe_op_metrics,
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
    print(f"ms_per_input_token: {flops['ms_per_input_token']:.8f}")
    print(f"active_expert_tflops_per_second: {flops['active_expert_tflops_per_second']:.4f}")
    print(f"std_forward_ms: {timing_metrics['std_forward_ms']:.4f}")
    print(f"peak_memory_allocated_bytes: {timing_metrics['peak_memory_allocated_bytes']}")
    if phase_metrics:
        print(f"phase_path: {phase_metrics['phase_path']}")
        print(f"phase_gpu_sum_ms: {phase_metrics['phase_gpu_sum_ms']:.4f}")
        print(f"phase_gather_ms: {phase_metrics['phase_gather_ms']:.4f}")
        print(f"phase_expert_mlp_ms: {phase_metrics['phase_expert_mlp_ms']:.4f}")
        print(f"phase_scatter_ms: {phase_metrics['phase_scatter_ms']:.4f}")
    if moe_op_metrics:
        print(f"moe_op_profile_scope: {moe_op_metrics['moe_op_profile_scope']}")
        print(f"moe_op_router_projection_matmul_ms: {moe_op_metrics['moe_op_router_projection_matmul_ms']:.4f}")
        print(f"moe_op_topk_selection_ms: {moe_op_metrics['moe_op_topk_selection_ms']:.4f}")
        print(f"moe_op_selected_softmax_gating_ms: {moe_op_metrics['moe_op_selected_softmax_gating_ms']:.4f}")
        print(
            "moe_op_expert_block_dispatch_compute_combine_ms: "
            f"{moe_op_metrics['moe_op_expert_block_dispatch_compute_combine_ms']:.4f}",
        )
        print(f"moe_op_gate_multiply_combine_ms: {moe_op_metrics['moe_op_gate_multiply_combine_ms']:.4f}")
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
