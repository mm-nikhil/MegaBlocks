"""Collect clock-derived NanoJAX MoE hardware-utilization rows.

This runner intentionally does not use Nsight Compute or NVIDIA hardware
counters. It measures MoE runtime with ``profile_moe_layer.py`` and computes a
roofline-style estimate:

    clock_compute_utilization = W / (t * f * P * R)

where W is useful selected-expert FLOPs, t is measured MoE time, f is SM clock,
P is SM count, and R is assumed peak FLOPs per SM cycle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from moe_profile.run_fields import MOE_OP_FIELDS
from moe_profile.run_plots import save_moe_op_dashboard


DEFAULT_TOKENS = "512,1024,2048,4096,8192,16384"
DEFAULT_BACKENDS = "megablocks_moe,megablocks_dmoe"
DEFAULT_FALLBACK_SM_CLOCK_MHZ = 2115.0
DEFAULT_FALLBACK_SM_COUNT = 68
DEFAULT_PEAK_FLOPS_PER_SM_CYCLE = 256.0

SUMMARY_FIELDS = (
    "model_shape_name",
    "backend",
    "megablocks_layer",
    "timing_scope",
    "weight_source",
    "dtype",
    "N",
    "batch_size",
    "seq_len",
    "d_model",
    "d_ff",
    "n_experts",
    "top_k",
    "mean_forward_ms",
    "active_expert_flops",
    "active_expert_flops_per_token",
    "sm_clock_mhz",
    "sm_clock_source",
    "sm_count",
    "sm_count_source",
    "peak_flops_per_sm_cycle",
    "clock_elapsed_cycles_per_sm",
    "clock_elapsed_sm_cycle_slots",
    "clock_ideal_expert_cycles_per_sm",
    "clock_estimated_slack_cycles_per_sm",
    "clock_compute_util_pct",
    "clock_estimated_unused_compute_pct",
    "clock_equivalent_unused_sms",
    "clock_peak_tflops_per_second",
    "clock_achieved_expert_tflops_per_second",
    *MOE_OP_FIELDS,
    "timing_json",
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-shape-name", default="nano_moe_jax")
    parser.add_argument("--model-shapes-config", type=Path, default=Path("configs/moe_model_shapes.json"))
    parser.add_argument("--result-root", type=Path, default=Path("results/hardware"))
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-source", default="trained_nano_checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/trained_nano_moe_checkpoint"))
    parser.add_argument("--checkpoint-block-index", type=int, default=0)
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument(
        "--timing-scope",
        choices=("auto", "moe_layer", "expert_path"),
        default="moe_layer",
        help=(
            "moe_layer is the full Nano-compatible MoE layer: router projection, "
            "top-k, selected-logit softmax/gating, expert path, and output combine. "
            "expert_path isolates the prepared MegaBlocks dispatch/expert/combine path."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--clock-source",
        choices=("max_sm_clock", "current_sm_clock"),
        default="max_sm_clock",
        help="Clock source for clock-derived cycle estimates when --sm-clock-mhz is not set.",
    )
    parser.add_argument(
        "--sm-clock-mhz",
        type=float,
        default=0.0,
        help="Override SM clock in MHz for clock-derived estimates.",
    )
    parser.add_argument(
        "--fallback-sm-clock-mhz",
        type=float,
        default=DEFAULT_FALLBACK_SM_CLOCK_MHZ,
        help="Fallback SM clock MHz if nvidia-smi cannot report the requested clock.",
    )
    parser.add_argument(
        "--sm-count",
        type=int,
        default=0,
        help="Override SM count / PE count for clock-derived estimates.",
    )
    parser.add_argument(
        "--fallback-sm-count",
        type=int,
        default=DEFAULT_FALLBACK_SM_COUNT,
        help="Fallback SM count if torch cannot report CUDA device properties.",
    )
    parser.add_argument(
        "--peak-flops-per-sm-cycle",
        type=float,
        default=DEFAULT_PEAK_FLOPS_PER_SM_CYCLE,
        help="Assumed peak FLOPs per SM cycle. RTX 3080 CUDA-core FP32/FMA roof uses 256.",
    )
    parser.add_argument(
        "--skip-moe-op-profile",
        action="store_true",
        help="Skip diagram-level per-op timing diagnostics.",
    )
    parser.add_argument("--moe-op-warmup", type=int, default=5)
    parser.add_argument("--moe-op-iters", type=int, default=20)
    parser.add_argument("--allow-bias-mismatch", action="store_true")
    parser.add_argument("--skip-memory-preflight", action="store_true")
    parser.add_argument("--memory-preflight-fraction", type=float, default=0.90)
    parser.add_argument("--memory-preflight-safety-multiplier", type=float, default=1.35)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_shape(path: Path, name: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        shapes = json.load(handle)
    if name not in shapes:
        available = ", ".join(sorted(shapes))
        raise SystemExit(f"Unknown model shape {name!r}. Available shapes: {available}")
    return shapes[name]


def resolved_timing_scope(value: str) -> str:
    return value


def query_nvidia_smi_clocks() -> dict[str, float | str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.max.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        return {"clock_query_status": f"nvidia-smi unavailable: {exc}"}
    if result.returncode != 0:
        reason = (result.stderr.strip() or result.stdout.strip()).splitlines()
        return {"clock_query_status": reason[0] if reason else "nvidia-smi clock query failed"}
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        return {"clock_query_status": f"could not parse nvidia-smi clock output: {first_line!r}"}
    try:
        current_sm_clock_mhz = float(parts[0])
        max_sm_clock_mhz = float(parts[1])
    except ValueError:
        return {"clock_query_status": f"could not parse nvidia-smi clock output: {first_line!r}"}
    return {
        "clock_query_status": "ok",
        "current_sm_clock_mhz": current_sm_clock_mhz,
        "max_sm_clock_mhz": max_sm_clock_mhz,
    }


def select_sm_clock_mhz(
    *,
    override_mhz: float,
    clock_source: str,
    clock_metadata: dict[str, float | str],
    fallback_mhz: float,
) -> tuple[float | None, str]:
    if override_mhz > 0:
        return override_mhz, "override"
    key = "max_sm_clock_mhz" if clock_source == "max_sm_clock" else "current_sm_clock_mhz"
    value = clock_metadata.get(key)
    if isinstance(value, (float, int)) and value > 0:
        return float(value), clock_source
    if fallback_mhz > 0:
        return fallback_mhz, f"fallback_{clock_source}"
    return None, f"{clock_source}_unavailable"


def select_sm_count(
    *,
    override_count: int,
    metadata: dict[str, object],
    fallback_count: int,
) -> tuple[int, str]:
    if override_count > 0:
        return int(override_count), "override"
    value = metadata.get("sm_count")
    if isinstance(value, int) and value > 0:
        return int(value), "torch_cuda_properties"
    if fallback_count > 0:
        return int(fallback_count), "fallback"
    return 0, "unavailable"


def clean_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float):
        return None
    return value_float


def active_expert_flops_per_token(shape: dict) -> int:
    expert_type = str(shape.get("expert_type", "ffn") or "ffn")
    if expert_type == "ffn":
        multiplier = 4
    elif expert_type == "glu":
        multiplier = 6
    else:
        raise SystemExit(f"Unsupported expert_type for useful FLOP estimate: {expert_type!r}")

    d_model = int(shape["hidden_size"])
    d_ff = int(shape["expert_intermediate_size"])
    top_k = int(shape["num_experts_per_token"])
    shared_experts = int(shape.get("num_shared_experts", 0) or 0)
    shared_hidden = int(shape.get("shared_expert_intermediate_size", 0) or 0)
    routed = multiplier * top_k * d_model * d_ff
    shared = multiplier * shared_experts * d_model * shared_hidden
    return int(routed + shared)


def clock_derived_metrics(
    *,
    mean_forward_ms: object,
    active_expert_flops: int,
    sm_count: int,
    sm_clock_mhz: float | None,
    sm_clock_source: str,
    peak_flops_per_sm_cycle: float,
) -> dict[str, float | str | None]:
    metrics: dict[str, float | str | None] = {
        "sm_clock_mhz": sm_clock_mhz,
        "sm_clock_source": sm_clock_source,
        "peak_flops_per_sm_cycle": peak_flops_per_sm_cycle,
        "clock_elapsed_cycles_per_sm": None,
        "clock_elapsed_sm_cycle_slots": None,
        "clock_ideal_expert_cycles_per_sm": None,
        "clock_estimated_slack_cycles_per_sm": None,
        "clock_compute_util_pct": None,
        "clock_estimated_unused_compute_pct": None,
        "clock_equivalent_unused_sms": None,
        "clock_peak_tflops_per_second": None,
        "clock_achieved_expert_tflops_per_second": None,
    }
    mean_ms = clean_float(mean_forward_ms)
    if (
        mean_ms is None
        or mean_ms <= 0
        or active_expert_flops <= 0
        or sm_count <= 0
        or sm_clock_mhz is None
        or sm_clock_mhz <= 0
        or peak_flops_per_sm_cycle <= 0
    ):
        return metrics

    elapsed_cycles_per_sm = mean_ms * sm_clock_mhz * 1000.0
    elapsed_sm_cycle_slots = elapsed_cycles_per_sm * sm_count
    theoretical_compute_capacity = elapsed_sm_cycle_slots * peak_flops_per_sm_cycle
    ideal_expert_cycles_per_sm = active_expert_flops / (sm_count * peak_flops_per_sm_cycle)
    compute_util_pct = 100.0 * active_expert_flops / theoretical_compute_capacity
    unused_compute_pct = 100.0 - compute_util_pct
    peak_tflops = sm_count * peak_flops_per_sm_cycle * sm_clock_mhz * 1e6 / 1e12
    achieved_tflops = active_expert_flops / (mean_ms / 1000.0) / 1e12

    metrics.update({
        "clock_elapsed_cycles_per_sm": elapsed_cycles_per_sm,
        "clock_elapsed_sm_cycle_slots": elapsed_sm_cycle_slots,
        "clock_ideal_expert_cycles_per_sm": ideal_expert_cycles_per_sm,
        "clock_estimated_slack_cycles_per_sm": elapsed_cycles_per_sm - ideal_expert_cycles_per_sm,
        "clock_compute_util_pct": compute_util_pct,
        "clock_estimated_unused_compute_pct": unused_compute_pct,
        "clock_equivalent_unused_sms": sm_count * unused_compute_pct / 100.0,
        "clock_peak_tflops_per_second": peak_tflops,
        "clock_achieved_expert_tflops_per_second": achieved_tflops,
    })
    return metrics


def backend_args(backend: str, *, allow_bias_mismatch: bool) -> tuple[list[str], str, str]:
    if backend == "megablocks_moe":
        args = ["--backend", "megablocks", "--megablocks-layer", "moe"]
        megablocks_layer = "moe"
    elif backend == "megablocks_dmoe":
        args = ["--backend", "megablocks", "--megablocks-layer", "dmoe"]
        megablocks_layer = "dmoe"
    else:
        raise ValueError(f"Hardware profiling currently supports MegaBlocks backends only, got {backend!r}.")

    if allow_bias_mismatch:
        args.append("--allow-bias-mismatch")
    else:
        args.append("--use-expert-biases")
    return args, backend, megablocks_layer


def gpu_metadata() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"gpu_metadata_status": "torch_unavailable"}
    if not torch.cuda.is_available():
        return {"gpu_metadata_status": "cuda_unavailable"}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_metadata_status": "ok",
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": int(props.multi_processor_count),
        "total_memory_bytes": int(props.total_memory),
    }


def run_json_profile(
    *,
    profile_script: Path,
    raw_path: Path,
    common_args: list[str],
    dry_run: bool,
) -> tuple[dict, str, str, str]:
    if raw_path.exists():
        raw_path.unlink()
    cmd = [sys.executable, str(profile_script), *common_args, "--jsonl-out", str(raw_path)]
    command_text = " ".join(cmd)
    if dry_run:
        return {"dry_run_command": command_text}, "", "", command_text
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Profiler failed for {raw_path.name}: {message}")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Profiler wrote no rows to {raw_path}")
    return rows[-1], result.stdout, result.stderr, command_text


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def save_dashboard(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))
    by_backend: dict[str, list[dict]] = {}
    for row in rows:
        by_backend.setdefault(str(row["backend"]), []).append(row)

    for backend, backend_rows in sorted(by_backend.items()):
        ordered = sorted(backend_rows, key=lambda item: int(item["N"]))
        axes[0].plot(
            [int(row["N"]) for row in ordered],
            [float(row["mean_forward_ms"]) for row in ordered],
            marker="o",
            linewidth=2,
            label=backend,
        )
        util_rows = [row for row in ordered if row.get("clock_compute_util_pct") not in (None, "")]
        if util_rows:
            axes[1].plot(
                [int(row["N"]) for row in util_rows],
                [float(row["clock_compute_util_pct"]) for row in util_rows],
                marker="o",
                linewidth=2,
                label=backend,
            )
        unused_rows = [row for row in ordered if row.get("clock_equivalent_unused_sms") not in (None, "")]
        if unused_rows:
            axes[2].plot(
                [int(row["N"]) for row in unused_rows],
                [float(row["clock_equivalent_unused_sms"]) for row in unused_rows],
                marker="o",
                linewidth=2,
                label=backend,
            )

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("N input-token rows")
    axes[0].set_ylabel("mean forward time (ms)")
    axes[0].grid(True, color="0.9")
    axes[0].legend()

    has_util_rows = any(row.get("clock_compute_util_pct") not in (None, "") for row in rows)
    if has_util_rows:
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("N input-token rows")
        axes[1].set_ylabel("clock-derived compute utilization (%)")
        axes[1].grid(True, color="0.9")
        axes[1].legend()
    else:
        axes[1].set_axis_off()
        axes[1].text(
            0.5,
            0.5,
            "Clock-derived metric unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    has_unused_rows = any(row.get("clock_equivalent_unused_sms") not in (None, "") for row in rows)
    if has_unused_rows:
        axes[2].set_xscale("log", base=2)
        axes[2].set_xlabel("N input-token rows")
        axes[2].set_ylabel("equivalent unused SMs")
        axes[2].grid(True, color="0.9")
        axes[2].legend()
    else:
        axes[2].set_axis_off()
        axes[2].text(
            0.5,
            0.5,
            "Unused-SM estimate unavailable",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    title_scope = rows[0].get("timing_scope", "unknown") if rows else "unknown"
    fig.suptitle(f"Clock-Derived MoE Compute Utilization ({title_scope})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_notes(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict],
    metadata: dict[str, object],
    failures: list[str],
) -> None:
    first = rows[0] if rows else {}
    timing_scope = resolved_timing_scope(args.timing_scope)
    lines = [
        "# Hardware MoE Profile",
        "",
        f"model_shape_name: `{args.model_shape_name}`",
        f"timing_scope: `{timing_scope}`",
        f"weight_source: `{args.weight_source}`",
        f"dtype: `{args.dtype}`",
        "",
        "Metric:",
        "",
        "`clock_compute_utilization = W / (t * f * P * R)`",
        "",
        "- `W`: active useful selected-expert FLOPs.",
        "- `t`: measured MoE runtime from `profile_moe_layer.py`.",
        "- `f`: SM clock in cycles/second.",
        "- `P`: SM count / PE count.",
        "- `R`: assumed peak FLOPs per SM cycle.",
        "",
        "This is clock-derived compute utilization, not observed GPU idle cycles,",
        "SM active cycles, hardware occupancy, or measured memory stalls.",
        "",
        "GPU and denominator:",
        "",
        f"- name: `{metadata.get('gpu_name', 'unknown')}`",
        f"- compute capability: `{metadata.get('compute_capability', 'unknown')}`",
        f"- SM count used: `{first.get('sm_count', 'unknown')}` ({first.get('sm_count_source', 'unknown')})",
        f"- current SM clock MHz reported: `{metadata.get('current_sm_clock_mhz', 'unknown')}`",
        f"- max SM clock MHz reported: `{metadata.get('max_sm_clock_mhz', 'unknown')}`",
        f"- SM clock MHz used: `{first.get('sm_clock_mhz', 'unknown')}` ({first.get('sm_clock_source', 'unknown')})",
        f"- peak FLOPs per SM-cycle: `{args.peak_flops_per_sm_cycle}`",
        f"- peak TFLOP/s: `{first.get('clock_peak_tflops_per_second', 'unknown')}`",
        "",
        "Per-op timing diagnostics:",
        "",
        "The optional `moe_op_*` fields are independent diagram-level replays.",
        "They are separate from the clock-derived metric. The expert block timing",
        "is the whole MegaBlocks expert dispatch/compute/combine call. The",
        "`moe_op_gate_multiply_combine_ms` field times the weighted scatter/combine",
        "subset, so it is not additive with the expert-block timing.",
        "",
        "The full presentation timing boundary is `moe_layer`:",
        "router projection, top-k, selected-logit softmax/gating, expert block,",
        "gate multiply, and combine back to Nano `[N x D]` layout.",
        "",
    ]
    if failures:
        lines.extend([
            "Failures:",
            "",
            *[f"- {failure}" for failure in failures],
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_backend_status(path: Path, rows: list[dict], failures: list[str], requested_backends: list[str]) -> None:
    """Record max successful N by backend plus unsupported/failing rows."""

    max_by_backend: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("backend", "unknown"))
        max_by_backend[backend] = max(max_by_backend.get(backend, 0), int(row["N"]))

    lines = [
        "# Backend Status",
        "",
        "Requested backends:",
        "",
        *[f"- `{backend}`" for backend in requested_backends],
        "",
        "Max successful `N`:",
        "",
    ]
    if max_by_backend:
        lines.extend(f"- `{backend}`: `{tokens}`" for backend, tokens in sorted(max_by_backend.items()))
    else:
        lines.append("- none")
    if failures:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shape = load_shape(args.model_shapes_config, args.model_shape_name)
    timing_scope = resolved_timing_scope(args.timing_scope)
    tokens_grid = parse_csv_ints(args.tokens)
    backends = parse_csv_strings(args.backends)
    result_dir = args.result_root / args.model_shape_name
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata = gpu_metadata()
    clock_metadata = query_nvidia_smi_clocks()
    metadata.update(clock_metadata)
    sm_count, sm_count_source = select_sm_count(
        override_count=args.sm_count,
        metadata=metadata,
        fallback_count=args.fallback_sm_count,
    )
    sm_clock_mhz, sm_clock_source = select_sm_clock_mhz(
        override_mhz=args.sm_clock_mhz,
        clock_source=args.clock_source,
        clock_metadata=clock_metadata,
        fallback_mhz=args.fallback_sm_clock_mhz,
    )
    per_token_flops = active_expert_flops_per_token(shape)
    profile_script = Path(__file__).with_name("profile_moe_layer.py")

    rows: list[dict] = []
    failures: list[str] = []
    for tokens in tokens_grid:
        if tokens % args.seq_len:
            raise SystemExit(f"N={tokens} is not divisible by seq_len={args.seq_len}")
        batch_size = tokens // args.seq_len
        for backend in backends:
            backend_cli, backend_name, megablocks_layer = backend_args(
                backend,
                allow_bias_mismatch=args.allow_bias_mismatch,
            )
            label = (
                f"{args.model_shape_name}_{backend_name}_hardware_"
                f"tok{tokens}_t{args.seq_len}_{args.dtype}_{args.weight_source}_{timing_scope}"
            )
            common = [
                *backend_cli,
                "--model-shape-name",
                args.model_shape_name,
                "--model-shapes-config",
                str(args.model_shapes_config),
                "--batch-size",
                str(batch_size),
                "--seq-len",
                str(args.seq_len),
                "--d-model",
                str(shape["hidden_size"]),
                "--d-ff",
                str(shape["expert_intermediate_size"]),
                "--n-experts",
                str(shape["num_routed_experts"]),
                "--top-k",
                str(shape["num_experts_per_token"]),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--timing-scope",
                timing_scope,
                "--weight-source",
                args.weight_source,
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--checkpoint-block-index",
                str(args.checkpoint_block_index),
                "--nano-jax-dir",
                str(args.nano_jax_dir),
                "--label",
                label,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--trials",
                str(args.trials),
            ]
            if not args.skip_moe_op_profile:
                common.extend([
                    "--moe-op-profile",
                    "--moe-op-warmup",
                    str(args.moe_op_warmup),
                    "--moe-op-iters",
                    str(args.moe_op_iters),
                ])
            if args.skip_memory_preflight:
                common.append("--skip-memory-preflight")
            common.extend([
                "--memory-preflight-fraction",
                str(args.memory_preflight_fraction),
                "--memory-preflight-safety-multiplier",
                str(args.memory_preflight_safety_multiplier),
            ])

            timing_raw = result_dir / f"timing_{backend_name}_N{tokens}.jsonl"
            print(f"N={tokens} backend={backend_name} timing_scope={timing_scope}", flush=True)
            try:
                timing_row, _, _, command_text = run_json_profile(
                    profile_script=profile_script,
                    raw_path=timing_raw,
                    common_args=common,
                    dry_run=args.dry_run,
                )
            except RuntimeError as exc:
                failure = f"N={tokens} backend={backend_name}: {exc}"
                failures.append(failure)
                print(f"  failed {failure}", flush=True)
                continue

            active_flops = int(tokens * per_token_flops)
            clock_metrics = clock_derived_metrics(
                mean_forward_ms=timing_row.get("mean_forward_ms"),
                active_expert_flops=active_flops,
                sm_count=sm_count,
                sm_clock_mhz=sm_clock_mhz,
                sm_clock_source=sm_clock_source,
                peak_flops_per_sm_cycle=args.peak_flops_per_sm_cycle,
            )

            row = {
                "model_shape_name": args.model_shape_name,
                "backend": backend_name,
                "megablocks_layer": megablocks_layer,
                "timing_scope": timing_scope,
                "weight_source": args.weight_source,
                "dtype": args.dtype,
                "N": tokens,
                "batch_size": batch_size,
                "seq_len": args.seq_len,
                "d_model": shape["hidden_size"],
                "d_ff": shape["expert_intermediate_size"],
                "n_experts": shape["num_routed_experts"],
                "top_k": shape["num_experts_per_token"],
                "mean_forward_ms": timing_row.get("mean_forward_ms"),
                "active_expert_flops": active_flops,
                "active_expert_flops_per_token": per_token_flops,
                "sm_count": sm_count,
                "sm_count_source": sm_count_source,
                "timing_json": str(timing_raw),
                "profiler_command": command_text,
                **clock_metrics,
            }
            for key in MOE_OP_FIELDS:
                if key in timing_row:
                    row[key] = timing_row[key]
            rows.append(row)
            util = row.get("clock_compute_util_pct")
            util_text = f"{float(util):.2f}%" if isinstance(util, (float, int)) else "n/a"
            print(f"  done ms={row['mean_forward_ms']} clock_compute_util={util_text}", flush=True)

    raw_path = result_dir / "raw.jsonl"
    summary_path = result_dir / "summary.csv"
    dashboard_path = result_dir / "graphs_clock_compute.png"
    legacy_dashboard_path = result_dir / "dashboard.png"
    moe_op_dashboard_path = result_dir / "graphs_moe_ops.png"
    notes_path = result_dir / "notes.md"
    backend_status_path = result_dir / "backend_status.md"
    config_path = result_dir / "config.json"
    metadata_path = result_dir / "gpu_metadata.json"

    write_jsonl(rows, raw_path)
    write_csv(rows, summary_path)
    save_dashboard(rows, dashboard_path)
    save_dashboard(rows, legacy_dashboard_path)
    wrote_moe_op_dashboard = save_moe_op_dashboard(
        rows,
        moe_op_dashboard_path,
        x_key="N",
        series_key="backend",
        title="MoE Layer Operation Profile",
    )
    write_notes(notes_path, args=args, rows=rows, metadata=metadata, failures=failures)
    write_backend_status(backend_status_path, rows, failures, backends)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "model_shape_name": args.model_shape_name,
                "tokens": tokens_grid,
                "backends": backends,
                "timing_scope": timing_scope,
                "requested_timing_scope": args.timing_scope,
                "weight_source": args.weight_source,
                "dtype": args.dtype,
                "clock_source": args.clock_source,
                "sm_clock_mhz": sm_clock_mhz,
                "sm_clock_source": sm_clock_source,
                "sm_count": sm_count,
                "sm_count_source": sm_count_source,
                "peak_flops_per_sm_cycle": args.peak_flops_per_sm_cycle,
                "active_expert_flops_per_token": per_token_flops,
                "moe_op_profile": not args.skip_moe_op_profile,
                "moe_op_warmup": args.moe_op_warmup,
                "moe_op_iters": args.moe_op_iters,
                "skip_memory_preflight": args.skip_memory_preflight,
                "memory_preflight_fraction": args.memory_preflight_fraction,
                "memory_preflight_safety_multiplier": args.memory_preflight_safety_multiplier,
                "gpu_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"raw_jsonl={raw_path}")
    print(f"summary_csv={summary_path}")
    print(f"dashboard={dashboard_path}")
    if wrote_moe_op_dashboard:
        print(f"moe_op_dashboard={moe_op_dashboard_path}")
    print(f"notes={notes_path}")
    print(f"backend_status={backend_status_path}")
    print(f"gpu_metadata={metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
