"""Shared plotting helpers for profiling run dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


MOE_OP_PLOT_METRICS = (
    ("moe_op_router_projection_matmul_ms", "router matmul"),
    ("moe_op_topk_selection_ms", "top-k"),
    ("moe_op_selected_softmax_gating_ms", "selected softmax"),
    ("moe_op_expert_block_dispatch_compute_combine_ms", "expert path"),
    ("moe_op_gate_multiply_combine_ms", "gate/combine subset"),
    ("moe_op_output_layout_to_nano_ms", "output layout"),
)


def _row_has_value(row: dict, key: str) -> bool:
    return row.get(key) not in (None, "")


def save_moe_op_dashboard(
    rows: Iterable[dict],
    out_path: Path,
    *,
    x_key: str,
    series_key: str,
    title: str,
) -> bool:
    """Plot logical MoE op timings by run axis and series.

    The expert-path line is the whole MegaBlocks dispatch/compute/combine path.
    The gate/combine line is a subset timing, so the figure is diagnostic rather
    than a stacked/additive breakdown.
    """

    op_rows = [row for row in rows if row.get("moe_op_profile")]
    if not op_rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2))
    panels = (
        MOE_OP_PLOT_METRICS[:3],
        MOE_OP_PLOT_METRICS[3:],
    )

    for ax, metrics in zip(axes, panels):
        for metric, label in metrics:
            grouped: dict[str, list[tuple[float, float]]] = {}
            for row in op_rows:
                if not _row_has_value(row, x_key) or not _row_has_value(row, metric):
                    continue
                series = str(row.get(series_key, row.get("backend_variant", row.get("backend", "unknown"))))
                grouped.setdefault(series, []).append((float(row[x_key]), float(row[metric])))
            for series, points in sorted(grouped.items()):
                ordered = sorted(points)
                ax.plot(
                    [point[0] for point in ordered],
                    [point[1] for point in ordered],
                    marker="o",
                    linewidth=2,
                    label=f"{series}: {label}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N input-token rows")
        ax.set_ylabel("time (ms)")
        ax.grid(True, color="0.9")
        ax.legend(fontsize=8)

    axes[0].set_title("Router / Gating")
    axes[1].set_title("Expert Path / Combine")
    fig.suptitle(
        f"{title}\nIndependent diagnostic replays; gate/combine is a subset of expert path.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True

