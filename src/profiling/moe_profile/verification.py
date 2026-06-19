"""Correctness gates shared by profiling run orchestrators."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class NanoJAXGateCase:
    """One small-N correctness check before a NanoJAX performance sweep."""

    case_id: str
    backend: str
    megablocks_layer: str
    dtype: str
    threshold: float
    note: str


def _verification_cases(
    requested_backends: Iterable[str],
    *,
    fp32_threshold: float,
    bf16_threshold: float,
) -> list[NanoJAXGateCase]:
    """Build the requested NanoJAX correctness checks.

    ``megablocks_moe`` is checked in FP32 because that is the intended NanoJAX
    performance dtype. ``megablocks_dmoe`` is checked separately in BF16 because
    the local grouped GEMM extension used by dMoE is BF16-only.
    """

    backend_set = set(requested_backends)
    cases: list[NanoJAXGateCase] = []
    if "megablocks_moe" in backend_set:
        cases.append(
            NanoJAXGateCase(
                case_id="nanojax_fp32_megablocks_moe",
                backend="megablocks_moe",
                megablocks_layer="moe",
                dtype="float32",
                threshold=fp32_threshold,
                note="FP32 standard MoE adapter vs PyTorch NanoJAX reference.",
            ),
        )
    if "megablocks_dmoe" in backend_set:
        cases.append(
            NanoJAXGateCase(
                case_id="nanojax_bf16_megablocks_dmoe",
                backend="megablocks_dmoe",
                megablocks_layer="dmoe",
                dtype="bfloat16",
                threshold=bf16_threshold,
                note=(
                    "BF16-only dMoE adapter vs PyTorch NanoJAX reference; "
                    "local grouped_gemm does not support FP32."
                ),
            ),
        )
    return cases


def _latest_jsonl_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else None


def _write_summary_files(
    *,
    result_dir: Path,
    summary: dict[str, object],
) -> None:
    json_path = result_dir / "verification_summary.json"
    md_path = result_dir / "verification_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Verification Summary",
        "",
        f"status: `{summary['status']}`",
        f"model_shape_name: `{summary.get('model_shape_name', '')}`",
        f"weight_source: `{summary.get('weight_source', '')}`",
        f"checkpoint_dir: `{summary.get('checkpoint_dir', '')}`",
        f"verification_tokens: `{summary.get('verification_tokens', '')}`",
        "",
        "These checks run at small `N` before the performance sweep. They verify",
        "the MegaBlocks adapter against the PyTorch NanoJAX MoE reference using",
        "the selected NanoJAX weights. Large performance rows are not dense-reference",
        "checked row-by-row.",
        "",
    ]
    reason = summary.get("reason")
    if reason:
        lines.extend(["Reason:", "", str(reason), ""])

    cases = summary.get("cases", [])
    if isinstance(cases, list) and cases:
        lines.extend(["Cases:", ""])
        for case in cases:
            if not isinstance(case, dict):
                continue
            lines.extend([
                f"- `{case.get('case_id')}`: `{case.get('status')}`",
                f"  backend: `{case.get('backend')}`",
                f"  dtype: `{case.get('dtype')}`",
                f"  threshold: `{case.get('outlier_abs_threshold')}`",
                f"  max_abs_vs_reference: `{case.get('max_abs_vs_reference')}`",
                f"  aux_loss_abs_diff: `{case.get('aux_loss_abs_diff')}`",
                f"  router_expert_set_mismatch_count: `{case.get('router_expert_set_mismatch_count')}`",
                f"  note: {case.get('note')}",
            ])
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_nanojax_correctness_gate(
    *,
    result_dir: Path,
    profile_script: Path,
    model_shape_name: str,
    model_shapes_config: Path,
    shape: dict[str, object],
    requested_backends: Iterable[str],
    weight_source: str,
    checkpoint_dir: Path,
    checkpoint_block_index: int,
    nano_jax_dir: Path,
    device: str,
    seed: int,
    verification_batch_size: int,
    verification_seq_len: int,
    fp32_threshold: float,
    bf16_threshold: float,
    dry_run: bool,
    skip: bool,
) -> dict[str, object]:
    """Run and persist the NanoJAX correctness gate.

    The gate only applies to the exact NanoJAX adapter target. Shape-only model
    simulations, such as OLMoE-shaped synthetic runs, get a not-applicable
    summary because they do not have exact checkpoint/reference semantics here.
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    simulation_level = str(shape.get("simulation_level", "custom"))
    requested_backends = list(requested_backends)
    tokens = verification_batch_size * verification_seq_len
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "model_shape_name": model_shape_name,
        "simulation_level": simulation_level,
        "requested_backends": requested_backends,
        "weight_source": weight_source,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_block_index": checkpoint_block_index,
        "verification_batch_size": verification_batch_size,
        "verification_seq_len": verification_seq_len,
        "verification_tokens": tokens,
        "cases": [],
    }

    if skip:
        summary["status"] = "skipped"
        summary["reason"] = "Skipped by --no-verify / --skip-correctness-gate."
        _write_summary_files(result_dir=result_dir, summary=summary)
        return summary

    if model_shape_name != "nano_moe_jax" or simulation_level != "exact_adapter":
        summary["status"] = "not_applicable"
        summary["reason"] = "Correctness gate is only defined for the exact NanoJAX adapter target."
        _write_summary_files(result_dir=result_dir, summary=summary)
        return summary

    if weight_source != "trained_nano_checkpoint":
        summary["status"] = "failed"
        summary["reason"] = (
            "NanoJAX performance runs must use trained_nano_checkpoint for the "
            "mandatory correctness gate."
        )
        _write_summary_files(result_dir=result_dir, summary=summary)
        raise SystemExit(summary["reason"])

    cases = _verification_cases(
        requested_backends,
        fp32_threshold=fp32_threshold,
        bf16_threshold=bf16_threshold,
    )
    if not cases:
        summary["status"] = "not_required"
        summary["reason"] = "No MegaBlocks backend was requested."
        _write_summary_files(result_dir=result_dir, summary=summary)
        return summary

    for case in cases:
        raw_path = result_dir / f"verification_{case.case_id}.jsonl"
        if raw_path.exists() and not dry_run:
            raw_path.unlink()
        cmd = [
            sys.executable,
            str(profile_script),
            "--backend",
            "megablocks",
            "--megablocks-layer",
            case.megablocks_layer,
            "--model-shape-name",
            model_shape_name,
            "--model-shapes-config",
            str(model_shapes_config),
            "--batch-size",
            str(verification_batch_size),
            "--seq-len",
            str(verification_seq_len),
            "--d-model",
            str(shape["hidden_size"]),
            "--d-ff",
            str(shape["expert_intermediate_size"]),
            "--n-experts",
            str(shape["num_routed_experts"]),
            "--top-k",
            str(shape["num_experts_per_token"]),
            "--dtype",
            case.dtype,
            "--device",
            device,
            "--seed",
            str(seed),
            "--timing-scope",
            "moe_layer",
            "--weight-source",
            weight_source,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-block-index",
            str(checkpoint_block_index),
            "--nano-jax-dir",
            str(nano_jax_dir),
            "--use-expert-biases",
            "--check-output",
            "--outlier-abs-threshold",
            str(case.threshold),
            "--warmup",
            "1",
            "--iters",
            "1",
            "--trials",
            "1",
            "--jsonl-out",
            str(raw_path),
            "--label",
            case.case_id,
        ]
        case_summary: dict[str, object] = {
            "case_id": case.case_id,
            "backend": case.backend,
            "megablocks_layer": case.megablocks_layer,
            "dtype": case.dtype,
            "outlier_abs_threshold": case.threshold,
            "note": case.note,
            "command": " ".join(cmd),
            "jsonl": str(raw_path),
        }
        if dry_run:
            case_summary["status"] = "dry_run"
            summary["cases"].append(case_summary)
            continue

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        case_summary["returncode"] = result.returncode
        if result.returncode != 0:
            case_summary["status"] = "failed"
            case_summary["stderr"] = result.stderr.strip()
            case_summary["stdout"] = result.stdout.strip()
            summary["cases"].append(case_summary)
            continue

        record = _latest_jsonl_record(raw_path)
        if record is None:
            case_summary["status"] = "failed"
            case_summary["stderr"] = "Profiler completed but wrote no verification row."
            summary["cases"].append(case_summary)
            continue

        passed = bool(record.get("correctness_passed"))
        case_summary.update({
            "status": "passed" if passed else "failed",
            "correctness_passed": passed,
            "max_abs_vs_reference": record.get("max_abs_vs_reference"),
            "mean_abs_vs_reference": record.get("mean_abs_vs_reference"),
            "max_rel_vs_reference": record.get("max_rel_vs_reference"),
            "aux_loss_abs_diff": record.get("aux_loss_abs_diff"),
            "router_expert_set_mismatch_count": record.get("router_expert_set_mismatch_count"),
            "router_gate_max_abs": record.get("router_gate_max_abs"),
            "outlier_diagnosis": record.get("outlier_diagnosis"),
        })
        summary["cases"].append(case_summary)

    case_statuses = [str(case.get("status")) for case in summary["cases"] if isinstance(case, dict)]
    if dry_run:
        summary["status"] = "dry_run"
    elif all(status == "passed" for status in case_statuses):
        summary["status"] = "passed"
    else:
        summary["status"] = "failed"
        summary["reason"] = "One or more NanoJAX correctness gate cases failed."

    _write_summary_files(result_dir=result_dir, summary=summary)
    if summary["status"] == "failed":
        raise SystemExit(str(summary["reason"]))
    return summary
