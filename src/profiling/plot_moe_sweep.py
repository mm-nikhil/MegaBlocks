"""Plot MoE sweep latency and speedup from profiling JSONL."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_KEYS = ("tokens", "d_model", "d_ff", "n_experts", "top_k", "dtype", "weight_source")
LABEL_KEYS = (*CONFIG_KEYS, "backend_variant", "device")


@dataclass(frozen=True)
class PlotSpec:
    axis_key: str
    filename_key: str
    xlabel: str
    fixed_keys: tuple[str, ...]


PLOTS = (
    PlotSpec(
        axis_key="tokens",
        filename_key="tokens",
        xlabel="tokens = batch_size * seq_len",
        fixed_keys=("d_model", "d_ff", "n_experts", "top_k", "dtype", "weight_source"),
    ),
    PlotSpec(
        axis_key="d_ff",
        filename_key="d_ff",
        xlabel="d_ff",
        fixed_keys=("tokens", "d_model", "n_experts", "top_k", "dtype", "weight_source"),
    ),
    PlotSpec(
        axis_key="n_experts",
        filename_key="n_experts",
        xlabel="n_experts",
        fixed_keys=("tokens", "d_model", "d_ff", "top_k", "dtype", "weight_source"),
    ),
    PlotSpec(
        axis_key="d_model",
        filename_key="d_model",
        xlabel="d_model",
        fixed_keys=("tokens", "d_ff", "n_experts", "top_k", "dtype", "weight_source"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots/moe_sweep"))
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--weight-source", default="nano_jax_init")
    parser.add_argument(
        "--series-label",
        choices=("auto", "backend", "backend_device"),
        default="auto",
        help="Use backend_device for combined CPU/GPU plots.",
    )
    parser.add_argument(
        "--reference-device",
        help=(
            "Device for speedup denominator. Defaults to cuda when both cuda and "
            "cpu reference rows are present."
        ),
    )
    parser.add_argument("--latest-per-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-points", type=int, default=2, help="Minimum distinct x-axis values required.")
    parser.add_argument("--formats", default="png", help="Comma-separated output formats, e.g. png,pdf.")
    return parser.parse_args()


def load_rows(path: Path, *, latest_per_label: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not latest_per_label:
        return rows

    by_label: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        label = row.get("label") or json.dumps(
            {key: row.get(key) for key in LABEL_KEYS},
            sort_keys=True,
        )
        if label in by_label:
            del by_label[label]
        by_label[label] = row
    return list(by_label.values())


def backend_name(row: dict) -> str:
    return str(row.get("backend_variant", row.get("backend", "unknown")))


def device_name(row: dict) -> str:
    return str(row.get("device", "unknown"))


def is_reference(row: dict) -> bool:
    return backend_name(row) == "reference"


def label_mode(rows: Iterable[dict], requested: str) -> str:
    if requested != "auto":
        return requested
    devices = {device_name(row) for row in rows}
    return "backend_device" if len(devices) > 1 else "backend"


def series_name(row: dict, mode: str) -> str:
    name = backend_name(row)
    if mode == "backend_device":
        return f"{name}_{device_name(row)}"
    return name


def config_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in CONFIG_KEYS)


def base_filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tokens": args.tokens,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "dtype": args.dtype,
        "weight_source": args.weight_source,
    }


def matches(row: dict, filters: dict[str, object]) -> bool:
    return all(row.get(key) == value for key, value in filters.items())


def rows_for_plot(rows: Iterable[dict], spec: PlotSpec, filters: dict[str, object]) -> list[dict]:
    fixed = {key: filters[key] for key in spec.fixed_keys}
    return [
        row for row in rows
        if row.get(spec.axis_key) is not None
        and row.get("mean_forward_ms") is not None
        and matches(row, fixed)
    ]


def grouped_points(rows: Iterable[dict], axis_key: str, mode: str) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[series_name(row, mode)].append((float(row[axis_key]), float(row["mean_forward_ms"])))
    return {
        name: sorted(points)
        for name, points in sorted(groups.items())
    }


def resolve_reference_device(rows: Iterable[dict], requested: str | None) -> str | None:
    if requested is not None:
        return requested
    devices = sorted({device_name(row) for row in rows if is_reference(row)})
    if "cuda" in devices:
        return "cuda"
    return devices[0] if devices else None


def reference_times(rows: Iterable[dict], *, reference_device: str | None) -> dict[tuple, float]:
    refs = {}
    for row in rows:
        if is_reference(row) and (reference_device is None or device_name(row) == reference_device):
            refs[config_key(row)] = float(row["mean_forward_ms"])
    return refs


def speedup_points(
    rows: Iterable[dict],
    axis_key: str,
    refs: dict[tuple, float],
    mode: str,
) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if is_reference(row):
            continue
        name = series_name(row, mode)
        ref_ms = refs.get(config_key(row))
        if ref_ms is None:
            continue
        groups[name].append((float(row[axis_key]), ref_ms / float(row["mean_forward_ms"])))
    return {
        name: sorted(points)
        for name, points in sorted(groups.items())
    }


def save_line_plot(
    *,
    groups: dict[str, list[tuple[float, float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: list[str],
    hline: float | None = None,
) -> list[str]:
    if not groups:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, points in groups.items():
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=name)

    if hline is not None:
        ax.axhline(hline, color="0.45", linewidth=1, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.9")
    ax.legend()
    fig.tight_layout()

    written = []
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=160)
        written.append(str(out_path))
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    rows = load_rows(args.jsonl, latest_per_label=args.latest_per_label)
    filters = base_filters(args)
    mode = label_mode(rows, args.series_label)
    reference_device = resolve_reference_device(rows, args.reference_device)
    formats = [item for item in args.formats.split(",") if item]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    refs = reference_times(rows, reference_device=reference_device)
    manifest = {
        "source": str(args.jsonl),
        "filters": filters,
        "series_label": mode,
        "reference_device": reference_device,
        "plots": [],
    }

    for spec in PLOTS:
        plot_rows = rows_for_plot(rows, spec, filters)
        axis_values = {row.get(spec.axis_key) for row in plot_rows}
        if len(axis_values) < args.min_points:
            continue

        latency_paths = save_line_plot(
            groups=grouped_points(plot_rows, spec.axis_key, mode),
            title=f"MoE latency vs {spec.xlabel}",
            xlabel=spec.xlabel,
            ylabel="mean forward time (ms)",
            out_base=args.out_dir / f"latency_vs_{spec.filename_key}",
            formats=formats,
        )
        speedup_paths = save_line_plot(
            groups=speedup_points(plot_rows, spec.axis_key, refs, mode),
            title=f"MegaBlocks speedup vs {reference_device or 'reference'} reference by {spec.xlabel}",
            xlabel=spec.xlabel,
            ylabel="reference ms / backend ms",
            out_base=args.out_dir / f"speedup_vs_{spec.filename_key}",
            formats=formats,
            hline=1.0,
        )
        if latency_paths or speedup_paths:
            manifest["plots"].append({
                "axis": spec.axis_key,
                "points": len(plot_rows),
                "latency": latency_paths,
                "speedup": speedup_paths,
            })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest['plots'])} plot groups to {args.out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
