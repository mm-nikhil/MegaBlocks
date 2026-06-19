"""Shared plotting helpers for profiling run dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from moe_profile.config import DMOE_BF16_ONLY_DTYPE_POLICY


MOE_OP_PLOT_METRICS = (
    ("moe_op_input_layout_to_megablocks_ms", "input layout"),
    ("moe_op_router_projection_matmul_ms", "router projection matmul"),
    ("moe_op_router_full_softmax_ms", "router softmax"),
    ("moe_op_topk_selection_ms", "top-k selection"),
    ("moe_op_selected_softmax_gating_ms", "gate softmax"),
    ("moe_op_router_aux_loss_ms", "aux bookkeeping"),
    ("moe_op_expert_path_dispatch_compute_combine_ms", "expert block"),
    ("moe_op_output_layout_to_nano_ms", "output layout"),
    ("moe_op_disjoint_replay_sum_ms", "component sum"),
    ("moe_op_whole_moe_layer_replay_ms", "whole replay"),
)


def _row_has_value(row: dict, key: str) -> bool:
    return row.get(key) not in (None, "")


def _series_label(row: dict, series_key: str) -> str:
    """Label plotted series with dtype/policy when the row carries them."""

    base = str(row.get(series_key, row.get("backend_variant", row.get("backend", "unknown"))))
    dtype = row.get("dtype")
    policy = row.get("dtype_policy")

    # Keep graph legends compact. The full backend / dtype / policy strings are
    # still stored in CSV/config; the plot needs enough detail to distinguish
    # FP32 MoE from the local BF16-only dMoE path without burying the data.
    if "megablocks_dmoe" in base:
        base = "dMoE"
    elif "megablocks_moe" in base:
        base = "MoE"
    if dtype == "bfloat16":
        dtype_label = "BF16"
    elif dtype == "float32":
        dtype_label = "FP32"
    elif dtype:
        dtype_label = str(dtype)
    else:
        dtype_label = ""
    if dtype_label:
        base = f"{base} {dtype_label}"
    if policy == DMOE_BF16_ONLY_DTYPE_POLICY:
        base = f"{base} only"
    elif policy and policy not in {"", "requested"}:
        base = f"{base} ({policy})"
    return base


def save_moe_op_dashboard(
    rows: Iterable[dict],
    out_path: Path,
    *,
    x_key: str,
    series_key: str,
    title: str,
) -> bool:
    """Plot logical MoE op timings by run axis and series.

    Components are disjoint at the replay level. The whole-replay and component
    sum lines make replay overhead visible, while ``mean_forward_ms`` remains
    the authoritative production timing in the primary latency graph.
    """

    op_rows = [row for row in rows if row.get("moe_op_profile")]
    if not op_rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 7.4))
    panels = (
        MOE_OP_PLOT_METRICS[:6],
        MOE_OP_PLOT_METRICS[6:8],
        MOE_OP_PLOT_METRICS[8:],
    )
    legend_specs = []

    for ax, metrics in zip(axes, panels):
        for metric, label in metrics:
            grouped: dict[str, list[tuple[float, float]]] = {}
            for row in op_rows:
                if not _row_has_value(row, x_key) or not _row_has_value(row, metric):
                    continue
                series = _series_label(row, series_key)
                grouped.setdefault(series, []).append((float(row[x_key]), float(row[metric])))
            for series, points in sorted(grouped.items()):
                ordered = sorted(points)
                ax.plot(
                    [point[0] for point in ordered],
                    [point[1] for point in ordered],
                    marker="o",
                    markersize=4,
                    linewidth=1.7,
                    label=f"{series} - {label}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N input-token rows")
        ax.set_ylabel("time (ms)")
        ax.grid(True, color="0.9")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = 2 if len(labels) > 8 else 1
            legend_specs.append((ax, handles, labels, ncol))

    axes[0].set_title("Routing / Gate Weights")
    axes[1].set_title("Expert Block / Output Layout")
    axes[2].set_title("Replay Check")
    fig.suptitle(
        f"{title}\nDisjoint replay diagnostics; compare component sum with whole replay, not with production latency.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.40, top=0.78, wspace=0.36)
    for ax, handles, labels, ncol in legend_specs:
        position = ax.get_position()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(position.x0, 0.34),
            bbox_transform=fig.transFigure,
            fontsize=7,
            ncol=ncol,
            frameon=False,
            handlelength=1.8,
            columnspacing=0.9,
            labelspacing=0.35,
        )
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True
