"""Summarize MoE profiling JSONL as a compact comparison table."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


CONFIG_KEYS = ("tokens", "d_model", "d_ff", "n_experts", "top_k", "dtype", "weight_source")
LABEL_KEYS = (*CONFIG_KEYS, "backend_variant", "device")


def parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item for item in value.split(",") if item}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--backends", help="Comma-separated backend_variant filter.")
    parser.add_argument("--dtypes", help="Comma-separated dtype filter.")
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--all-runs", action="store_true", help="Do not collapse repeated labels to the latest run.")
    parser.add_argument(
        "--reference-device",
        help=(
            "Device for speedup denominator. Defaults to cuda when both cuda and "
            "cpu reference rows are present."
        ),
    )
    parser.add_argument("--show-throughput", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, *, latest_per_label: bool) -> list[dict]:
    if latest_per_label:
        rows_by_label: OrderedDict[str, dict] = OrderedDict()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label") or json.dumps({key: row.get(key) for key in LABEL_KEYS}, sort_keys=True)
            if label in rows_by_label:
                del rows_by_label[label]
            rows_by_label[label] = row
        return list(rows_by_label.values())

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def config_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in CONFIG_KEYS)


def sort_key(row: dict) -> tuple:
    return (
        row.get("tokens", 0),
        row.get("d_model", 0),
        row.get("d_ff", 0),
        row.get("n_experts", 0),
        row.get("top_k", 0),
        row.get("dtype", ""),
        row.get("weight_source", ""),
        row.get("device", ""),
        row.get("backend_variant", row.get("backend", "")),
    )


def is_reference(row: dict) -> bool:
    return row.get("backend_variant") == "reference" or row.get("backend") == "reference"


def device_name(row: dict) -> str:
    return str(row.get("device", "unknown"))


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


def fmt_float(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_int(value) -> str:
    if value is None:
        return "-"
    return str(int(value))


def fmt_max_abs(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def table_row(row: dict, refs: dict[tuple, float], *, show_throughput: bool) -> list[str]:
    ref_ms = refs.get(config_key(row))
    speedup = None
    if ref_ms is not None and row.get("mean_forward_ms"):
        speedup = ref_ms / float(row["mean_forward_ms"])
    mem_delta = row.get("peak_memory_allocated_delta_bytes")
    mem_delta_mb = None if mem_delta is None else float(mem_delta) / (1024 * 1024)
    values = [
        fmt_int(row.get("tokens")),
        fmt_int(row.get("d_model")),
        fmt_int(row.get("d_ff")),
        fmt_int(row.get("n_experts")),
        fmt_int(row.get("top_k")),
        str(row.get("dtype", "-")),
        str(row.get("weight_source", "-")),
        str(row.get("backend_variant", row.get("backend", "-"))),
        str(row.get("device", "-")),
        fmt_float(row.get("mean_forward_ms"), 4),
        fmt_float(row.get("std_forward_ms"), 4),
        fmt_float(speedup, 2),
        fmt_float(mem_delta_mb, 1),
        fmt_max_abs(row.get("max_abs_vs_reference")),
        fmt_int(row.get("router_expert_set_mismatch_count")),
        str(row.get("outlier_diagnosis", "-")),
    ]
    if show_throughput:
        values.insert(11, fmt_int(row.get("tokens_per_second")))
        values.insert(12, fmt_float(row.get("backend_estimated_tflops_per_second"), 3))
    return values


def print_table(rows: list[dict], *, reference_device: str | None, show_throughput: bool) -> None:
    refs = reference_times(rows, reference_device=reference_device)
    headers = [
        "tokens",
        "d",
        "ff",
        "experts",
        "k",
        "dtype",
        "weights",
        "backend",
        "device",
        "ms",
        "std",
        "speedup",
        "mem_delta_MB",
        "max_abs",
        "router_flips",
        "diagnosis",
    ]
    if show_throughput:
        headers.insert(11, "tok/s")
        headers.insert(12, "TF/s")
    table = [headers, ["---"] * len(headers)]
    table.extend(table_row(row, refs, show_throughput=show_throughput) for row in rows)
    for values in table:
        print("| " + " | ".join(values) + " |")


def main() -> None:
    args = parse_args()
    backend_filter = parse_csv_strings(args.backends)
    dtype_filter = parse_csv_strings(args.dtypes)
    rows = load_rows(args.jsonl, latest_per_label=not args.all_runs)
    if backend_filter is not None:
        rows = [
            row for row in rows
            if row.get("backend_variant", row.get("backend")) in backend_filter
        ]
    if dtype_filter is not None:
        rows = [row for row in rows if row.get("dtype") in dtype_filter]
    rows = sorted(rows, key=sort_key)

    if args.max_rows is not None:
        rows = rows[:args.max_rows]

    reference_device = resolve_reference_device(rows, args.reference_device)
    print(f"rows={len(rows)} source={args.jsonl}")
    print(f"speedup_reference_device={reference_device}")
    print_table(rows, reference_device=reference_device, show_throughput=args.show_throughput)


if __name__ == "__main__":
    main()
