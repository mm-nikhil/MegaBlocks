"""Run the first-cut model token-capacity sweep.

This is intentionally narrower than the general sweep script. It reads a model
shape from ``configs/moe_model_shapes.json``, varies N = B*T, appends profiler
records to one result folder, and writes one dashboard image for the first-cut
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_TOKENS = "512,1024,2048,4096,8192,16384,32768,65536,131072"
DEFAULT_BACKENDS = "reference,megablocks_moe,megablocks_dmoe"
SUMMARY_FIELDS = (
    "model_shape_name",
    "simulation_level",
    "backend_variant",
    "timing_scope",
    "weight_source",
    "checkpoint_dir",
    "checkpoint_block_index",
    "device",
    "dtype",
    "batch_size",
    "seq_len",
    "tokens",
    "d_model",
    "d_ff",
    "n_experts",
    "top_k",
    "num_shared_experts",
    "shared_expert_intermediate_size",
    "expert_type",
    "activation",
    "bias_semantics",
    "expert_bias_max_abs",
    "mean_forward_ms",
    "std_forward_ms",
    "memory_preflight_enabled",
    "memory_preflight_estimated_bytes",
    "memory_preflight_base_estimated_bytes",
    "memory_preflight_cuda_free_bytes",
    "memory_preflight_cuda_total_bytes",
    "memory_preflight_allowed_bytes",
    "memory_preflight_fraction",
    "memory_preflight_safety_multiplier",
    "ms_per_input_token",
    "assignments",
    "router_flops",
    "active_expert_flops_per_token",
    "active_expert_tflops_per_second",
    "tokens_per_second",
    "ms_per_assignment",
    "padding_factor",
    "phase_profile",
    "phase_path",
    "phase_profile_warmup",
    "phase_profile_iters",
    "phase_sort_ms",
    "phase_histogram_ms",
    "phase_cumsum_ms",
    "phase_capacity_decision_wall_ms",
    "phase_gather_ms",
    "phase_expert_mlp_ms",
    "phase_scatter_ms",
    "phase_gpu_sum_ms",
    "phase_expert_capacity",
    "tokens_per_expert_min",
    "tokens_per_expert_max",
    "tokens_per_expert_mean",
    "tokens_per_expert_std",
    "expert_imbalance",
    "check_output",
    "correctness_passed",
    "outlier_abs_threshold",
    "max_abs_vs_reference",
    "mean_abs_vs_reference",
    "max_rel_vs_reference",
    "aux_loss_abs_diff",
    "router_expert_set_mismatch_count",
    "router_gate_max_abs",
    "outlier_diagnosis",
    "label",
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-shape-name", default="nano_moe_jax")
    parser.add_argument("--model-shapes-config", type=Path, default=Path("configs/moe_model_shapes.json"))
    parser.add_argument("--result-root", type=Path, default=Path("results/model-token-capacity"))
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--seq-len", type=int, default=0, help="0 means use the shape catalog max_position_embeddings.")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--dtype", help="Defaults to the shape catalog dtype.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-scope", choices=("auto", "megablocks_core", "adapter_boundary"), default="auto")
    parser.add_argument(
        "--weight-source",
        choices=("auto", "nano_jax_init", "synthetic", "trained_nano_checkpoint"),
        default="auto",
    )
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/trained_nano_moe_checkpoint"))
    parser.add_argument("--checkpoint-block-index", type=int, default=0)
    parser.add_argument("--outlier-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-check-output", action="store_true")
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--phase-warmup", type=int, default=5)
    parser.add_argument("--phase-iters", type=int, default=20)
    parser.add_argument("--plot-mode", choices=("token_capacity", "phase_profile"), default="token_capacity")
    parser.add_argument("--skip-memory-preflight", action="store_true")
    parser.add_argument("--memory-preflight-fraction", type=float, default=0.90)
    parser.add_argument("--memory-preflight-safety-multiplier", type=float, default=1.35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_shape(path: Path, name: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        shapes = json.load(handle)
    if name not in shapes:
        available = ", ".join(sorted(shapes))
        raise SystemExit(f"Unknown model shape {name!r}. Available shapes: {available}")
    return shapes[name]


def label_value(value: str) -> str:
    return value.replace(":", "").replace("/", "_")


def backend_args(backend: str, *, check_output: bool, shape: dict) -> list[str]:
    simulation_level = str(shape.get("simulation_level", "custom"))
    if backend == "reference":
        return ["--backend", "reference"]
    if backend == "megablocks_moe":
        args = [
            "--backend",
            "megablocks",
            "--megablocks-layer",
            "moe",
        ]
        args.append("--use-expert-biases" if simulation_level == "exact_adapter" else "--zero-expert-biases")
        if check_output:
            args.append("--check-output")
        return args
    if backend == "megablocks_dmoe":
        args = [
            "--backend",
            "megablocks",
            "--megablocks-layer",
            "dmoe",
        ]
        args.append("--use-expert-biases" if simulation_level == "exact_adapter" else "--zero-expert-biases")
        if check_output:
            args.append("--check-output")
        return args
    raise ValueError(f"Unsupported backend: {backend}")


def latest_jsonl_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else None


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows_by_label: OrderedDict[str, dict] = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = str(row.get("label", ""))
        if label in rows_by_label:
            del rows_by_label[label]
        rows_by_label[label] = row
    return list(rows_by_label.values())


def write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.get("tokens", 0), item.get("backend_variant", ""))):
            writer.writerow(row)


def group_points(rows: Iterable[dict], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is None:
            continue
        grouped[str(row.get("backend_variant", row.get("backend", "unknown")))].append(
            (float(row["tokens"]), float(row[metric])),
        )
    return {name: sorted(points) for name, points in sorted(grouped.items())}


def simulation_caveat(simulation_level: object) -> str:
    if str(simulation_level) == "exact_adapter":
        return ""
    return "Caveat: synthetic shape/expert simulation; not exact checkpoint/router semantics."


def save_dashboard(rows: list[dict], shape: dict, out_path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("mean_forward_ms", "mean forward time (ms)"),
        ("ms_per_input_token", "ms per input-token row"),
        ("active_expert_tflops_per_second", "active expert TFLOP/s"),
        ("padding_factor", "backend rows / routed assignments"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    axes = axes.flatten()

    for ax, (metric, ylabel) in zip(axes, metrics):
        for name, points in group_points(rows, metric).items():
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(x_values, y_values, marker="o", linewidth=2, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N input-token rows = batch_size * seq_len")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9")
        ax.legend()

    model_name = rows[0].get("model_shape_name", "unknown")
    timing_by_backend = {
        str(row.get("backend_variant", row.get("backend", "unknown"))): str(row.get("timing_scope", "unknown"))
        for row in rows
    }
    timing_text = ", ".join(
        f"{backend}={timing}" for backend, timing in sorted(timing_by_backend.items())
    )
    subtitle = (
        f"{model_name} | {rows[0].get('simulation_level')} | "
        f"D={shape['hidden_size']} H={shape['expert_intermediate_size']} "
        f"E={shape['num_routed_experts']} K={shape['num_experts_per_token']} "
        f"S={shape.get('num_shared_experts', 0)} "
        f"{shape['expert_type']}/{shape['activation']} "
        f"dtype={rows[0].get('dtype')} "
        f"weights={rows[0].get('weight_source', 'unknown')}\n"
        f"timing: {timing_text}"
    )
    caveat = simulation_caveat(rows[0].get("simulation_level"))
    if caveat:
        subtitle = f"{subtitle}\n{caveat}"
    fig.suptitle(f"Model Token-Capacity Sweep\n{subtitle}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87 if caveat else 0.90))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_phase_dashboard(rows: list[dict], out_path: Path) -> None:
    phase_rows = [row for row in rows if row.get("phase_profile")]
    if not phase_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        (
            "Metadata Phases",
            (
                ("phase_sort_ms", "sort"),
                ("phase_histogram_ms", "histogram"),
                ("phase_cumsum_ms", "cumsum"),
                ("phase_capacity_decision_wall_ms", "capacity sync"),
            ),
        ),
        (
            "Dispatch / Compute Phases",
            (
                ("phase_gather_ms", "gather"),
                ("phase_expert_mlp_ms", "expert MLP"),
                ("phase_scatter_ms", "scatter"),
            ),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (title, metrics) in zip(axes, panels):
        for metric, short_name in metrics:
            for backend, points in group_points(phase_rows, metric).items():
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                ax.plot(x_values, y_values, marker="o", linewidth=2, label=f"{backend}: {short_name}")
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N input-token rows = batch_size * seq_len")
        ax.set_ylabel("phase time (ms)")
        ax.grid(True, color="0.9")
        ax.legend(fontsize=8)

    model_name = phase_rows[0].get("model_shape_name", "unknown")
    caveat = simulation_caveat(phase_rows[0].get("simulation_level"))
    title_lines = [
        f"MoE Phase Profile: {model_name}",
        "Independent diagnostic timings; phase sums are not exact whole-call decomposition.",
    ]
    if caveat:
        title_lines.append(caveat)
    fig.suptitle("\n".join(title_lines), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.84 if caveat else 0.88))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_notes(
    *,
    path: Path,
    args: argparse.Namespace,
    shape: dict,
    rows: list[dict],
    failures: list[str],
) -> None:
    max_by_backend: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("backend_variant", row.get("backend", "unknown")))
        max_by_backend[backend] = max(max_by_backend.get(backend, 0), int(row["tokens"]))

    lines = [
        "# Phase Profile" if args.plot_mode == "phase_profile" else "# Model Token-Capacity Sweep",
        "",
        f"model_shape_name: `{args.model_shape_name}`",
        f"simulation_level: `{shape.get('simulation_level')}`",
        f"weight_source: `{args.weight_source}`",
        f"dtype: `{args.dtype or shape['dtype']}`",
        "",
    ]
    if args.weight_source == "trained_nano_checkpoint":
        lines.extend([
            f"checkpoint_dir: `{args.checkpoint_dir}`",
            f"checkpoint_block_index: `{args.checkpoint_block_index}`",
            "",
        ])
    lines.extend([
        f"check_output: `{not args.skip_check_output and shape.get('simulation_level') == 'exact_adapter'}`",
        f"outlier_abs_threshold: `{args.outlier_abs_threshold}`",
        "",
    ])
    caveat = simulation_caveat(shape.get("simulation_level"))
    if caveat:
        lines.extend([
            "## Simulation Caveat",
            "",
            caveat,
            "",
            "This means the run uses the catalog's MoE layer geometry and expert type",
            "such as `D`, `H`, `E`, `K`, GLU/SwiGLU, activation, and dtype, with",
            "synthetic weights. It does not load the real model checkpoint and does",
            "not implement the model-specific router exactly.",
            "",
            "For OLMoE-shaped runs, `reference_dense_glu` is a PyTorch CUDA dense",
            "all-expert GLU baseline for this synthetic shape. It is useful as a",
            "dense-vs-sparse comparison point, but it is not exact OLMoE execution.",
            "",
        ])
    lines.extend([
        "Max successful `N` by backend:",
        "",
        *[f"- `{backend}`: `{tokens}`" for backend, tokens in sorted(max_by_backend.items())],
        "",
        "This sweep varies `N = B*T`, the number of input-token hidden rows at one MoE layer.",
        "It is not generated output tokens per second.",
        "",
        "The dashboard shows:" if args.plot_mode == "phase_profile" else "The first-cut dashboard shows:",
        "",
    ])
    if args.plot_mode == "phase_profile":
        lines.extend([
            "- phase timings for MegaBlocks routing metadata, gather, expert MLP, and scatter.",
            "",
            "Phase timings are independent diagnostic replays of MegaBlocks operations.",
            "Use them to explain bottlenecks, not as an exact additive breakdown of",
            "`mean_forward_ms`.",
        ])
    else:
        lines.extend([
            "- `mean_forward_ms`: average timed forward call for the selected timing scope.",
            "- `ms_per_input_token`: `mean_forward_ms / N`.",
            "- `active_expert_tflops_per_second`: useful active expert math normalized by runtime.",
            "- `padding_factor`: backend expert rows divided by routed token-expert pairs.",
        ])
    if args.phase_profile and args.plot_mode != "phase_profile":
        lines.extend([
            "",
            "This result also includes `phase_dashboard.png` and phase columns in",
            "`summary.csv`. Phase timings are independent diagnostic replays of",
            "MegaBlocks operations, so use them to explain bottlenecks rather than",
            "as an exact additive breakdown of `mean_forward_ms`.",
        ])
    lines.extend([
        "",
        "Backend success, failure, and unsupported status is recorded in `backend_status.md`.",
        "",
        "Shape:",
        "",
        "```json",
        json.dumps(shape, indent=2, sort_keys=True),
        "```",
    ])
    if failures:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_backend_status(path: Path, rows: list[dict], failures: list[str], requested_backends: list[str]) -> None:
    max_by_backend: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("backend_variant", row.get("backend", "unknown")))
        max_by_backend[backend] = max(max_by_backend.get(backend, 0), int(row["tokens"]))

    lines = [
        "# Backend Status",
        "",
        "Requested backend families:",
        "",
        *[f"- `{backend}`" for backend in requested_backends],
        "",
        "Successful plotted backend variants:",
        "",
    ]
    if max_by_backend:
        lines.extend(f"- `{backend}`: max successful `N={tokens}`" for backend, tokens in sorted(max_by_backend.items()))
    else:
        lines.append("- none")

    if failures:
        lines.extend(["", "Failures / unsupported rows:", ""])
        lines.extend(f"- {failure}" for failure in failures)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shape = load_shape(args.model_shapes_config, args.model_shape_name)
    simulation_level = str(shape.get("simulation_level", "custom"))
    seq_len = args.seq_len or int(shape["max_position_embeddings"])
    dtype = args.dtype or str(shape["dtype"])
    weight_source = args.weight_source
    if weight_source == "auto":
        weight_source = "nano_jax_init" if simulation_level == "exact_adapter" else "synthetic"
    tokens_grid = parse_csv_ints(args.tokens)
    backends = parse_csv_strings(args.backends)
    result_dir = args.result_root / args.model_shape_name
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_path = result_dir / "raw.jsonl"
    summary_path = result_dir / "summary.csv"
    dashboard_path = result_dir / "dashboard.png"
    phase_dashboard_path = result_dir / "phase_dashboard.png"
    config_path = result_dir / "config.json"
    notes_path = result_dir / "notes.md"
    backend_status_path = result_dir / "backend_status.md"

    if raw_path.exists() and not args.append and not args.dry_run:
        raw_path.unlink()

    run_config = {
        "model_shape_name": args.model_shape_name,
        "model_shapes_config": str(args.model_shapes_config),
        "tokens": tokens_grid,
        "seq_len": seq_len,
        "backends": backends,
        "dtype": dtype,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "trials": args.trials,
        "timing_scope": args.timing_scope,
        "weight_source": weight_source,
        "checkpoint_dir": str(args.checkpoint_dir) if weight_source == "trained_nano_checkpoint" else None,
        "checkpoint_block_index": args.checkpoint_block_index if weight_source == "trained_nano_checkpoint" else None,
        "outlier_abs_threshold": args.outlier_abs_threshold,
        "skip_check_output": args.skip_check_output,
        "phase_profile": args.phase_profile,
        "plot_mode": args.plot_mode,
        "phase_warmup": args.phase_warmup,
        "phase_iters": args.phase_iters,
        "skip_memory_preflight": args.skip_memory_preflight,
        "memory_preflight_fraction": args.memory_preflight_fraction,
        "memory_preflight_safety_multiplier": args.memory_preflight_safety_multiplier,
        "shape": shape,
    }
    if args.append and config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            run_config["tokens"] = sorted(set(existing_config.get("tokens", [])) | set(tokens_grid))
            run_config["backends"] = sorted(set(existing_config.get("backends", [])) | set(backends))
        except json.JSONDecodeError:
            pass
    if args.append and raw_path.exists():
        existing_rows = load_rows(raw_path)
        run_config["tokens"] = sorted(
            set(run_config["tokens"]) | {int(row["tokens"]) for row in existing_rows},
        )
        run_config["backends"] = sorted(
            set(run_config["backends"])
            | {str(row.get("backend_variant", row.get("backend", "unknown"))) for row in existing_rows},
        )
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"result_dir={result_dir}")
    print(f"raw_jsonl={raw_path}")
    print(f"tokens={tokens_grid}")
    print(f"backends={backends}")

    profile_script = Path(__file__).with_name("profile_moe_layer.py")
    failures: list[str] = []
    for tokens in tokens_grid:
        if tokens % seq_len != 0:
            failures.append(f"N={tokens}: skipped because it is not divisible by T={seq_len}")
            continue
        batch_size = tokens // seq_len
        for backend in backends:
            label = (
                f"{args.model_shape_name}_{backend}_{label_value(args.device)}_"
                f"tok{tokens}_t{seq_len}_{dtype}_{weight_source}"
            )
            check_output = not args.skip_check_output and simulation_level == "exact_adapter"
            cmd = [
                sys.executable,
                str(profile_script),
                *backend_args(backend, check_output=check_output, shape=shape),
                "--model-shape-name",
                args.model_shape_name,
                "--model-shapes-config",
                str(args.model_shapes_config),
                "--batch-size",
                str(batch_size),
                "--seq-len",
                str(seq_len),
                "--d-model",
                str(shape["hidden_size"]),
                "--d-ff",
                str(shape["expert_intermediate_size"]),
                "--n-experts",
                str(shape["num_routed_experts"]),
                "--top-k",
                str(shape["num_experts_per_token"]),
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
                weight_source,
                "--nano-jax-dir",
                str(args.nano_jax_dir),
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--checkpoint-block-index",
                str(args.checkpoint_block_index),
                "--outlier-abs-threshold",
                str(args.outlier_abs_threshold),
                "--jsonl-out",
                str(raw_path),
                "--label",
                label,
            ]
            if args.phase_profile:
                cmd.extend([
                    "--phase-profile",
                    "--phase-warmup",
                    str(args.phase_warmup),
                    "--phase-iters",
                    str(args.phase_iters),
                ])
            if args.skip_memory_preflight:
                cmd.append("--skip-memory-preflight")
            cmd.extend([
                "--memory-preflight-fraction",
                str(args.memory_preflight_fraction),
                "--memory-preflight-safety-multiplier",
                str(args.memory_preflight_safety_multiplier),
            ])
            print(f"N={tokens} B={batch_size} backend={backend}")
            if args.dry_run or args.verbose:
                print(" ".join(cmd))
            if args.dry_run:
                continue

            result = subprocess.run(cmd, check=False, capture_output=not args.verbose, text=True)
            if result.returncode != 0:
                if not args.verbose:
                    print(result.stdout, end="")
                    print(result.stderr, end="", file=sys.stderr)
                reason = ""
                stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
                if stderr_lines:
                    reason = f" reason={stderr_lines[0]}"
                failure = f"N={tokens} backend={backend}: returncode={result.returncode}{reason}"
                failures.append(failure)
                print(f"failed: {failure}", file=sys.stderr)
                continue

            row = latest_jsonl_record(raw_path)
            if row is not None:
                print(
                    "  done "
                    f"ms={row['mean_forward_ms']:.4f} "
                    f"ms/token={row['ms_per_input_token']:.8f} "
                    f"active_TF/s={row['active_expert_tflops_per_second']:.3f}",
                )

    rows = load_rows(raw_path)
    if not args.dry_run:
        write_summary_csv(rows, summary_path)
        if args.plot_mode == "phase_profile":
            save_phase_dashboard(rows, dashboard_path)
            if phase_dashboard_path.exists():
                phase_dashboard_path.unlink()
        else:
            save_dashboard(rows, shape, dashboard_path)
            if phase_dashboard_path.exists():
                phase_dashboard_path.unlink()
        write_notes(path=notes_path, args=args, shape=shape, rows=rows, failures=failures)
        write_backend_status(backend_status_path, rows, failures, backends)
        print(f"summary_csv={summary_path}")
        print(f"dashboard={dashboard_path}")
        print(f"notes={notes_path}")
        print(f"backend_status={backend_status_path}")


if __name__ == "__main__":
    main()
