"""Run a structured sweep over MoE-layer profiling configs.

The sweep is intentionally a thin orchestrator around ``profile_moe_layer.py``.
Each run appends one JSONL record, so partial sweeps are still useful if a later
configuration fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


BACKENDS = ("reference", "megablocks_moe", "megablocks_dmoe")
Config = tuple[int, int, int, int, int, int, str, str]
FOCUSED_BASELINE = {
    "tokens": 4096,
    "d_model": 128,
    "d_ff": 512,
    "n_experts": 4,
    "top_k": 2,
    "dtype": "float32",
}


def focused_baseline_from_axes(
    *,
    tokens_grid: list[int],
    d_model_grid: list[int],
    d_ff_grid: list[int],
    n_expert_grid: list[int],
    top_k_grid: list[int],
    dtype_grid: list[str],
) -> dict[str, int | str]:
    baseline = dict(FOCUSED_BASELINE)
    axes = [
        ("tokens", tokens_grid),
        ("d_model", d_model_grid),
        ("d_ff", d_ff_grid),
        ("n_experts", n_expert_grid),
        ("top_k", top_k_grid),
        ("dtype", dtype_grid),
    ]
    for axis_name, values in axes:
        if len(values) == 1:
            baseline[axis_name] = values[0]
    return baseline


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def label_value(value: str) -> str:
    return value.replace(":", "").replace("/", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("focused", "grid"),
        default="focused",
        help="focused varies one axis at a time; grid runs the full Cartesian product.",
    )
    parser.add_argument("--tokens", default="512,2048,4096", help="Comma-separated B*T token counts.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length used to derive batch size.")
    parser.add_argument("--d-models", default="128,256")
    parser.add_argument("--d-ffs", default="512,1024")
    parser.add_argument("--n-experts", default="4,8")
    parser.add_argument("--top-ks", default="1,2")
    parser.add_argument("--dtypes", default="float32,float16,bfloat16")
    parser.add_argument("--backends", default="reference,megablocks_moe")
    parser.add_argument(
        "--dmoe-bias-mode",
        choices=("zero", "matched", "mismatch"),
        default="zero",
        help=(
            "Bias handling for megablocks_dmoe rows: zero is the current exact "
            "biasless dMoE benchmark, matched uses the bias-aware dMoE adapter, "
            "and mismatch times known-non-equivalent bias-free dMoE."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing-scope",
        choices=("auto", "megablocks_core", "adapter_boundary"),
        default="auto",
    )
    parser.add_argument("--weight-source", choices=("nano_jax_init", "synthetic"), default="nano_jax_init")
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--jsonl-out", type=Path, default=Path("results/raw/sweep.jsonl"))
    parser.add_argument("--limit", type=int, help="Run only the first N generated configs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print full profile output for every run.")
    return parser.parse_args()


def backend_args(backend: str, *, dmoe_bias_mode: str = "zero") -> list[str]:
    if backend == "reference":
        return ["--backend", "reference"]
    if backend == "megablocks_moe":
        return [
            "--backend",
            "megablocks",
            "--megablocks-layer",
            "moe",
            "--use-expert-biases",
            "--check-output",
        ]
    if backend == "megablocks_dmoe":
        args = [
            "--backend",
            "megablocks",
            "--megablocks-layer",
            "dmoe",
            "--check-output",
        ]
        if dmoe_bias_mode == "zero":
            args.append("--zero-expert-biases")
        elif dmoe_bias_mode == "matched":
            args.append("--use-expert-biases")
        elif dmoe_bias_mode == "mismatch":
            args.append("--allow-bias-mismatch")
        else:
            raise ValueError(f"Unsupported dMoE bias mode: {dmoe_bias_mode}")
        return args
    raise ValueError(f"Unsupported backend: {backend}")


def valid_combo(*, backend: str, dtype: str, top_k: int, n_experts: int, tokens: int, seq_len: int) -> bool:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Expected one of {BACKENDS}.")
    if top_k > n_experts:
        return False
    if tokens % seq_len != 0:
        return False
    # Current grouped dMoE path is only known to run with bf16 in this environment.
    if backend == "megablocks_dmoe" and dtype != "bfloat16":
        return False
    return True


def append_config(
    configs: list[Config],
    seen: set[Config],
    *,
    tokens: int,
    d_model: int,
    d_ff: int,
    n_experts: int,
    top_k: int,
    dtype: str,
    backend: str,
    seq_len: int,
) -> None:
    if not valid_combo(
        backend=backend,
        dtype=dtype,
        top_k=top_k,
        n_experts=n_experts,
        tokens=tokens,
        seq_len=seq_len,
    ):
        return
    batch_size = tokens // seq_len
    config = (tokens, batch_size, d_model, d_ff, n_experts, top_k, dtype, backend)
    if config in seen:
        return
    seen.add(config)
    configs.append(config)


def focused_configs(
    *,
    tokens_grid: list[int],
    d_model_grid: list[int],
    d_ff_grid: list[int],
    n_expert_grid: list[int],
    top_k_grid: list[int],
    dtype_grid: list[str],
    backend_grid: list[str],
    seq_len: int,
) -> list[Config]:
    configs = []
    seen = set()
    baseline = focused_baseline_from_axes(
        tokens_grid=tokens_grid,
        d_model_grid=d_model_grid,
        d_ff_grid=d_ff_grid,
        n_expert_grid=n_expert_grid,
        top_k_grid=top_k_grid,
        dtype_grid=dtype_grid,
    )

    axes = [
        ("tokens", tokens_grid),
        ("d_model", d_model_grid),
        ("d_ff", d_ff_grid),
        ("n_experts", n_expert_grid),
        ("top_k", top_k_grid),
        ("dtype", dtype_grid),
    ]

    for axis_name, values in axes:
        for value in values:
            params = dict(baseline)
            params[axis_name] = value
            for backend in backend_grid:
                append_config(
                    configs,
                    seen,
                    tokens=int(params["tokens"]),
                    d_model=int(params["d_model"]),
                    d_ff=int(params["d_ff"]),
                    n_experts=int(params["n_experts"]),
                    top_k=int(params["top_k"]),
                    dtype=str(params["dtype"]),
                    backend=backend,
                    seq_len=seq_len,
                )
    return configs


def grid_configs(
    *,
    tokens_grid: list[int],
    d_model_grid: list[int],
    d_ff_grid: list[int],
    n_expert_grid: list[int],
    top_k_grid: list[int],
    dtype_grid: list[str],
    backend_grid: list[str],
    seq_len: int,
) -> list[Config]:
    configs = []
    seen = set()
    for tokens, d_model, d_ff, n_experts, top_k, dtype, backend in itertools.product(
        tokens_grid,
        d_model_grid,
        d_ff_grid,
        n_expert_grid,
        top_k_grid,
        dtype_grid,
        backend_grid,
    ):
        append_config(
            configs,
            seen,
            tokens=tokens,
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            top_k=top_k,
            dtype=dtype,
            backend=backend,
            seq_len=seq_len,
        )
    return configs


def last_jsonl_record(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            if position == 0:
                return None
            buffer = bytearray()
            position -= 1
            while position >= 0:
                handle.seek(position)
                char = handle.read(1)
                if char == b"\n" and buffer:
                    break
                buffer.extend(char)
                position -= 1
        return json.loads(bytes(reversed(buffer)).decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def print_compact_result(record: dict | None) -> None:
    if record is None:
        print("done: no JSONL record found")
        return
    max_abs = record.get("max_abs_vs_reference")
    max_abs_text = "-" if max_abs is None else f"{max_abs:.3g}"
    flips = record.get("router_expert_set_mismatch_count", "-")
    diagnosis = record.get("outlier_diagnosis", "-")
    print(
        "done: "
        f"{record['label']} "
        f"ms={record['mean_forward_ms']:.4f} "
        f"max_abs={max_abs_text} "
        f"router_flips={flips} "
        f"diagnosis={diagnosis}",
    )


def main() -> None:
    args = parse_args()
    tokens_grid = parse_csv_ints(args.tokens)
    d_model_grid = parse_csv_ints(args.d_models)
    d_ff_grid = parse_csv_ints(args.d_ffs)
    n_expert_grid = parse_csv_ints(args.n_experts)
    top_k_grid = parse_csv_ints(args.top_ks)
    dtype_grid = parse_csv_strings(args.dtypes)
    backend_grid = parse_csv_strings(args.backends)

    config_builder = focused_configs if args.preset == "focused" else grid_configs
    configs = config_builder(
        tokens_grid=tokens_grid,
        d_model_grid=d_model_grid,
        d_ff_grid=d_ff_grid,
        n_expert_grid=n_expert_grid,
        top_k_grid=top_k_grid,
        dtype_grid=dtype_grid,
        backend_grid=backend_grid,
        seq_len=args.seq_len,
    )

    if args.limit is not None:
        configs = configs[:args.limit]

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"preset={args.preset}")
    print(f"sweep_configs={len(configs)}")
    print(f"jsonl_out={args.jsonl_out}")
    if args.preset == "focused":
        baseline = focused_baseline_from_axes(
            tokens_grid=tokens_grid,
            d_model_grid=d_model_grid,
            d_ff_grid=d_ff_grid,
            n_expert_grid=n_expert_grid,
            top_k_grid=top_k_grid,
            dtype_grid=dtype_grid,
        )
        print("focused_baseline=" + ",".join(f"{key}={value}" for key, value in baseline.items()))

    script = Path(__file__).with_name("profile_moe_layer.py")
    for index, (tokens, batch_size, d_model, d_ff, n_experts, top_k, dtype, backend) in enumerate(configs, start=1):
        label = (
            f"{backend}_{label_value(args.device)}_tok{tokens}_d{d_model}_ff{d_ff}_"
            f"e{n_experts}_k{top_k}_{dtype}_{args.weight_source}"
        )
        cmd = [
            sys.executable,
            str(script),
            *backend_args(backend, dmoe_bias_mode=args.dmoe_bias_mode),
            "--batch-size",
            str(batch_size),
            "--seq-len",
            str(args.seq_len),
            "--d-model",
            str(d_model),
            "--d-ff",
            str(d_ff),
            "--n-experts",
            str(n_experts),
            "--top-k",
            str(top_k),
            "--dtype",
            dtype,
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--trials",
            str(args.trials),
            "--seed",
            str(args.seed),
            "--timing-scope",
            args.timing_scope,
            "--weight-source",
            args.weight_source,
            "--nano-jax-dir",
            str(args.nano_jax_dir),
            "--outlier-abs-threshold",
            str(args.outlier_abs_threshold),
            "--jsonl-out",
            str(args.jsonl_out),
            "--label",
            label,
        ]
        if args.dry_run or args.verbose:
            print(f"[{index}/{len(configs)}] {' '.join(cmd)}")
        else:
            print(f"[{index}/{len(configs)}] {label}")
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=not args.verbose,
            text=True,
        )
        if result.returncode != 0:
            if args.continue_on_error:
                if not args.verbose:
                    print(result.stdout, end="")
                    print(result.stderr, end="", file=sys.stderr)
                print(f"failed_config={label} returncode={result.returncode}", file=sys.stderr)
                continue
            raise SystemExit(result.returncode)
        if not args.verbose:
            print_compact_result(last_jsonl_record(args.jsonl_out))


if __name__ == "__main__":
    main()
