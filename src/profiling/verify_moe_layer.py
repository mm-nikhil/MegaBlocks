"""Verify the Nano-compatible MegaBlocks MoE adapter against the PyTorch reference."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import torch

from nano_moe_torch import nano_moe_forward
from profile_moe_layer import (
    build_megablocks_layer,
    compare_moe_outputs,
    make_input,
    make_weights,
    megablocks_forward,
    parse_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("smoke", "single"),
        default="smoke",
        help="smoke verifies the current trusted working set; single verifies explicit args.",
    )
    parser.add_argument("--megablocks-layer", choices=("moe", "dmoe"), default="moe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-source", choices=("nano_jax_init", "synthetic"), default="nano_jax_init")
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--zero-expert-biases", action="store_true")
    parser.add_argument("--use-expert-biases", action="store_true")
    parser.add_argument("--allow-bias-mismatch", action="store_true")
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def case_namespace(args: argparse.Namespace, **overrides) -> Namespace:
    values = {
        "backend": "megablocks",
        "megablocks_layer": args.megablocks_layer,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "dtype": args.dtype,
        "device": args.device,
        "warmup": 0,
        "iters": 0,
        "trials": 1,
        "seed": args.seed,
        "timing_scope": "moe_layer",
        "weight_source": args.weight_source,
        "nano_jax_dir": args.nano_jax_dir,
        "zero_expert_biases": args.zero_expert_biases,
        "use_expert_biases": args.use_expert_biases,
        "allow_bias_mismatch": args.allow_bias_mismatch,
        "check_output": True,
        "verbose_checks": args.verbose,
        "outlier_abs_threshold": args.outlier_abs_threshold,
        "jsonl_out": None,
        "label": "",
    }
    values.update(overrides)
    if values["megablocks_layer"] == "moe" and not values["zero_expert_biases"]:
        values["use_expert_biases"] = True
    return Namespace(**values)


def smoke_cases(args: argparse.Namespace) -> list[tuple[str, Namespace]]:
    base = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "seed": args.seed,
        "nano_jax_dir": args.nano_jax_dir,
    }
    return [
        ("moe_fp32_nano_init", case_namespace(
            args,
            **base,
            megablocks_layer="moe",
            dtype="float32",
            weight_source="nano_jax_init",
            use_expert_biases=True,
            zero_expert_biases=False,
        )),
        ("moe_fp16_nano_init", case_namespace(
            args,
            **base,
            megablocks_layer="moe",
            dtype="float16",
            weight_source="nano_jax_init",
            use_expert_biases=True,
            zero_expert_biases=False,
        )),
        ("moe_fp32_synthetic_bias", case_namespace(
            args,
            **base,
            megablocks_layer="moe",
            dtype="float32",
            weight_source="synthetic",
            use_expert_biases=True,
            zero_expert_biases=False,
        )),
        ("dmoe_bf16_zero_bias", case_namespace(
            args,
            **base,
            megablocks_layer="dmoe",
            dtype="bfloat16",
            weight_source="nano_jax_init",
            use_expert_biases=False,
            zero_expert_biases=True,
        )),
    ]


def single_case(args: argparse.Namespace) -> list[tuple[str, Namespace]]:
    return [("single", case_namespace(args))]


def run_case(case_args: Namespace) -> dict[str, object]:
    device = torch.device(case_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    dtype = parse_dtype(case_args.dtype)
    weights = make_weights(case_args, dtype, device)
    x = make_input(case_args, dtype, device)
    model_shape = {
        "expert_type": "ffn",
        "activation": "gelu_tanh",
        "num_shared_experts": 0,
        "shared_expert_intermediate_size": 0,
    }
    layer = build_megablocks_layer(case_args, weights, model_shape, dtype, device)

    with torch.inference_mode():
        actual = megablocks_forward(
            layer,
            x,
            batch_size=case_args.batch_size,
            seq_len=case_args.seq_len,
            n_experts=case_args.n_experts,
            top_k=case_args.top_k,
            collect_diagnostics=True,
        )
        reference = nano_moe_forward(
            x,
            weights,
            top_k=case_args.top_k,
            deterministic=True,
            dropout_p=0.0,
        )

    return compare_moe_outputs(
        reference_for_checks=reference,
        actual_forward=actual,
        outlier_abs_threshold=case_args.outlier_abs_threshold,
    )


def print_case_summary(index: int, total: int, name: str, case_args: Namespace, metrics: dict[str, object]) -> None:
    status = "PASS" if metrics["correctness_passed"] else "FAIL"
    print(
        f"[{index}/{total}] {name}: {status} "
        f"layer={case_args.megablocks_layer} dtype={case_args.dtype} "
        f"weights={case_args.weight_source} "
        f"output_max_abs_error={metrics['max_abs_vs_reference']:.6g} "
        f"aux_loss_abs_error={metrics['aux_loss_abs_diff']:.6g} "
        f"router_set_mismatches={metrics['router_expert_set_mismatch_count']} "
        f"gate_max_abs_error={metrics['router_gate_max_abs']:.6g}"
    )


def print_verbose_metrics(metrics: dict[str, object]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    cases = smoke_cases(args) if args.preset == "smoke" else single_case(args)
    print("MegaBlocks adapter vs PyTorch NanoMoE reference")
    print(f"preset: {args.preset} cases={len(cases)}")

    failed = []
    for index, (name, case_args) in enumerate(cases, start=1):
        metrics = run_case(case_args)
        print_case_summary(index, len(cases), name, case_args, metrics)
        if args.verbose:
            print_verbose_metrics(metrics)
        if not metrics["correctness_passed"]:
            failed.append(name)

    if failed:
        raise SystemExit("FAILED cases: " + ", ".join(failed))
    print(f"all {len(cases)} checks passed")


if __name__ == "__main__":
    main()
