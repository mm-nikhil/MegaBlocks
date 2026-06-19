"""Runtime utilities for timing, memory preflight, and GPU metadata."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Optional

import torch

from moe_profile.config import dtype_nbytes


def memory_preflight(
    args: argparse.Namespace,
    *,
    model_shape: dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, int | float | bool]:
    """Conservative memory estimate before allocating weights and activations.

    This is not a CUDA allocator simulator. It is a guardrail for large shape
    sweeps so obviously-too-large runs fail cleanly instead of OOMing halfway
    through a batch of experiments.
    """

    if args.skip_memory_preflight or device.type != "cuda":
        return {"memory_preflight_enabled": False}

    if not 0 < args.memory_preflight_fraction <= 1:
        raise RuntimeError("--memory-preflight-fraction must be in (0, 1].")
    if args.memory_preflight_safety_multiplier < 1:
        raise RuntimeError("--memory-preflight-safety-multiplier must be >= 1.")

    nbytes = dtype_nbytes(dtype)
    tokens = args.batch_size * args.seq_len
    assignments = tokens * args.top_k
    expert_type = str(model_shape.get("expert_type", "ffn") or "ffn")
    weight_mats = 3 if expert_type == "glu" else 2
    shared_experts = int(model_shape.get("num_shared_experts", 0) or 0)
    shared_hidden = int(model_shape.get("shared_expert_intermediate_size", 0) or 0)

    routed_param_elems = weight_mats * args.n_experts * args.d_model * args.d_ff
    shared_param_elems = weight_mats * shared_experts * args.d_model * shared_hidden
    router_param_elems = args.d_model * args.n_experts

    # Synthetic runs allocate Nano-style FFN weights before MegaBlocks layer
    # construction, so include the temporary adapter tensors in the estimate.
    synthetic_adapter_elems = 0
    if args.weight_source == "synthetic":
        synthetic_adapter_elems = (
            2 * args.n_experts * args.d_model * args.d_ff
            + args.n_experts * (args.d_ff + args.d_model)
            + router_param_elems
        )

    input_elems = tokens * args.d_model
    router_elems = tokens * args.n_experts * 2
    if args.backend == "reference":
        dense_rows = tokens * args.n_experts
        assignment_elems = dense_rows * (args.d_model + args.d_ff + args.d_model)
        if expert_type == "glu":
            assignment_elems += dense_rows * args.d_ff
    else:
        assignment_elems = assignments * (args.d_model + args.d_ff + args.d_model)
        if expert_type == "glu":
            assignment_elems += assignments * args.d_ff

    base_estimated_bytes = int(
        nbytes
        * (
            routed_param_elems
            + shared_param_elems
            + router_param_elems
            + synthetic_adapter_elems
            + input_elems
            + router_elems
            + assignment_elems
        ),
    )
    estimated_bytes = int(base_estimated_bytes * args.memory_preflight_safety_multiplier)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allowed_bytes = int(free_bytes * args.memory_preflight_fraction)
    if estimated_bytes > allowed_bytes:
        raise RuntimeError(
            "Memory preflight rejected this run before allocation. "
            f"estimated={estimated_bytes} base_estimated={base_estimated_bytes} allowed={allowed_bytes} "
            f"free={free_bytes} total={total_bytes} fraction={args.memory_preflight_fraction}. "
            "Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.",
        )

    return {
        "memory_preflight_enabled": True,
        "memory_preflight_estimated_bytes": estimated_bytes,
        "memory_preflight_base_estimated_bytes": base_estimated_bytes,
        "memory_preflight_cuda_free_bytes": int(free_bytes),
        "memory_preflight_cuda_total_bytes": int(total_bytes),
        "memory_preflight_allowed_bytes": allowed_bytes,
        "memory_preflight_fraction": float(args.memory_preflight_fraction),
        "memory_preflight_safety_multiplier": float(args.memory_preflight_safety_multiplier),
    }


def cuda_time_ms(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    after_warmup: Optional[Callable[[], None]] = None,
    cuda_profiler_range: bool = False,
) -> float:
    """Average callable runtime in milliseconds.

    CUDA runs use events so kernel launches are measured on the GPU timeline.
    CPU fallback uses wall time for developer smoke checks.
    """

    if device.type != "cuda":
        for _ in range(warmup):
            fn()
        if after_warmup is not None:
            after_warmup()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return 1000.0 * (time.perf_counter() - start) / iters

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    if after_warmup is not None:
        after_warmup()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    profiler_started = False
    if cuda_profiler_range:
        # Nsight Compute can be asked to profile only this measured region.
        status = torch.cuda.cudart().cudaProfilerStart()
        if "success" not in str(status).lower():
            raise RuntimeError(f"cudaProfilerStart failed: {status}")
        profiler_started = True
    try:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
    finally:
        if profiler_started:
            status = torch.cuda.cudart().cudaProfilerStop()
            if "success" not in str(status).lower():
                raise RuntimeError(f"cudaProfilerStop failed: {status}")
    return float(start.elapsed_time(end) / iters)


def wall_time_ms(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """Wall-clock timing for small CPU-side decisions or sync-heavy code."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / iters


def measure_forward(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    trials: int,
    device: torch.device,
    cuda_profiler_range: bool = False,
) -> dict[str, float | int]:
    """Measure a configured forward callable and steady-state memory deltas."""

    trial_ms = []
    peak_allocated = 0
    peak_reserved = 0
    peak_allocated_delta = 0
    baseline_allocated_max = 0

    for _ in range(trials):
        baseline_allocated = 0

        def after_warmup() -> None:
            nonlocal baseline_allocated, baseline_allocated_max
            if device.type != "cuda":
                return
            torch.cuda.synchronize(device)
            # Reset after warmup so lazy kernel setup and allocator priming do not
            # get reported as steady-state forward memory.
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_allocated_max = max(baseline_allocated_max, baseline_allocated)
            torch.cuda.reset_peak_memory_stats(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        trial_ms.append(cuda_time_ms(
            fn,
            warmup=warmup,
            iters=iters,
            device=device,
            after_warmup=after_warmup,
            cuda_profiler_range=cuda_profiler_range,
        ))

        if device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            peak_allocated_delta = max(
                peak_allocated_delta,
                max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated),
            )

    ms_tensor = torch.tensor(trial_ms, dtype=torch.float64)
    std_ms = 0.0 if trials == 1 else float(ms_tensor.std(unbiased=False).item())
    return {
        "mean_forward_ms": float(ms_tensor.mean().item()),
        "std_forward_ms": std_ms,
        "min_forward_ms": float(ms_tensor.min().item()),
        "max_forward_ms": float(ms_tensor.max().item()),
        "trials": trials,
        "baseline_memory_allocated_bytes": int(baseline_allocated_max),
        "peak_memory_allocated_bytes": int(peak_allocated),
        "peak_memory_reserved_bytes": int(peak_reserved),
        "peak_memory_allocated_delta_bytes": int(peak_allocated_delta),
    }


def gpu_metadata(device: torch.device) -> dict[str, object]:
    """Small CUDA metadata block stored in every profiler record."""

    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_capability": list(torch.cuda.get_device_capability(index)),
        "torch_cuda": torch.version.cuda,
    }

