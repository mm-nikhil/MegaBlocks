# RTX 3080 Focused MoE Performance Sweep

Date: 2026-06-15  
Host: `aorus`  
GPU: NVIDIA GeForce RTX 3080, 10 GB  
Driver: 580.126.09  
Torch: 2.7.0+cu126, Torch CUDA 12.6  
Raw JSONL: `results/raw/rtx3080_focused_moe_20260615.jsonl`

Submodules:

```text
Nano-MoE-JAX: a41cc95
MegaBlocks:   952db33
grouped_gemm: f1429a3
```

## What Was Measured

This sweep profiles one NanoMoE-style MoE layer.

The correctness source of truth is Nano-MoE-JAX. We first verify that the PyTorch
NanoMoE reference matches Nano-MoE-JAX, then verify that the MegaBlocks adapter
matches the PyTorch reference.

For MegaBlocks timing, the default timing scope is `megablocks_core`:

```text
prepare Nano-compatible routing outside timing
time MegaBlocks dispatch / expert / combine
```

This intentionally excludes adapter setup: `(B,T,D)->(T,B,D)` layout conversion,
router logits, top-k, gates, and aux-loss bookkeeping. Correctness is still
checked for the full adapter output after timing.

Command:

```bash
.venv/bin/python src/profiling/sweep_moe_layer.py \
  --tokens 512,2048,4096 \
  --seq-len 128 \
  --d-models 128,256 \
  --d-ffs 512,1024 \
  --n-experts 4,8 \
  --top-ks 1,2 \
  --dtypes float32,float16,bfloat16 \
  --backends reference,megablocks_moe \
  --device cuda \
  --warmup 10 \
  --iters 50 \
  --trials 3 \
  --jsonl-out results/raw/rtx3080_focused_moe_20260615.jsonl
```

## Verification

```text
check_nano_moe_port.py: all 4 checks passed
verify_moe_layer.py:    all 4 checks passed
```

The focused sweep itself passed correctness for all FP32/FP16 MegaBlocks MoE
rows. The BF16 standard MoE row had no router flips and zero aux-loss error, but
showed numeric output error:

```text
max_abs_vs_reference = 0.015625
diagnosis = numeric_or_expert_path
```

## Results

`speedup` is reference latency divided by MegaBlocks latency for the same shape.
Values below 1 mean MegaBlocks is slower than the dense PyTorch reference for
that small config. The reference computes all experts densely, so it is a sanity
baseline rather than an optimized MoE implementation.

| tokens | d | ff | experts | k | dtype | backend | ms | std | speedup | mem_delta_MB | max_abs | router_flips | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 512 | 4 | 2 | float32 | reference | 0.3170 | 0.0015 | 1.00 | 8.0 | - | - | - |
| 512 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0494 | 0.0001 | 0.30 | 4.7 | 0 | 0 | within_threshold |
| 2048 | 128 | 512 | 4 | 2 | float32 | reference | 0.3698 | 0.0002 | 1.00 | 32.2 | - | - | - |
| 2048 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0492 | 0.0003 | 0.35 | 18.9 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 1 | float32 | reference | 0.6094 | 0.0009 | 1.00 | 64.3 | - | - | - |
| 4096 | 128 | 512 | 4 | 1 | float32 | megablocks_moe | 1.0494 | 0.0001 | 0.58 | 20.2 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float32 | reference | 0.6426 | 0.0017 | 1.00 | 64.3 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0491 | 0.0002 | 0.61 | 36.7 | 0 | 0 | within_threshold |
| 4096 | 256 | 512 | 4 | 2 | float32 | reference | 0.9546 | 0.0040 | 1.00 | 68.3 | - | - | - |
| 4096 | 256 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0493 | 0.0001 | 0.91 | 40.5 | 0 | 0 | within_threshold |
| 4096 | 128 | 1024 | 4 | 2 | float32 | reference | 1.0808 | 0.0041 | 1.00 | 128.3 | - | - | - |
| 4096 | 128 | 1024 | 4 | 2 | float32 | megablocks_moe | 1.0564 | 0.0100 | 1.02 | 69.3 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 8 | 2 | float32 | reference | 1.1200 | 0.0044 | 1.00 | 128.5 | - | - | - |
| 4096 | 128 | 512 | 8 | 2 | float32 | megablocks_moe | 1.0494 | 0.0001 | 1.07 | 38.6 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float16 | reference | 0.3464 | 0.0001 | 1.00 | 32.2 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float16 | megablocks_moe | 1.0495 | 0.0001 | 0.33 | 18.4 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | bfloat16 | reference | 0.3495 | 0.0000 | 1.00 | 32.2 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | bfloat16 | megablocks_moe | 1.0495 | 0.0001 | 0.33 | 18.4 | 0.016 | 0 | numeric_or_expert_path |

## Initial Read

- For the small NanoMoE-sized configs in this sweep, MegaBlocks standard MoE is
  approximately flat around 1.05 ms. This suggests fixed dispatch/kernel overhead
  dominates at these sizes.
- MegaBlocks begins to match or slightly beat the dense reference only when the
  dense reference does substantially more all-expert work, e.g. `d_ff=1024` or
  `n_experts=8`.
- FP32 and FP16 standard MoE rows match the reference exactly in this sweep.
- BF16 standard MoE needs investigation. Routing and aux loss match, so the error
  appears to be numeric/expert-path related rather than a router-choice issue.
- Memory delta is lower for MegaBlocks than the dense reference in the larger
  FP32 rows because the reference materializes all expert outputs.

## Plots

Generated from `results/raw/rtx3080_focused_moe_20260615.jsonl`:

- `results/plots/rtx3080_focused_moe_20260615/latency_vs_tokens.png`
- `results/plots/rtx3080_focused_moe_20260615/latency_vs_d_ff.png`
- `results/plots/rtx3080_focused_moe_20260615/latency_vs_n_experts.png`
- `results/plots/rtx3080_focused_moe_20260615/latency_vs_d_model.png`
- `results/plots/rtx3080_focused_moe_20260615/speedup_vs_tokens.png`
- `results/plots/rtx3080_focused_moe_20260615/speedup_vs_d_ff.png`
- `results/plots/rtx3080_focused_moe_20260615/speedup_vs_n_experts.png`
- `results/plots/rtx3080_focused_moe_20260615/speedup_vs_d_model.png`

## Follow-Up Items

1. Keep `megablocks_core` as the default MegaBlocks timing scope for
   dispatch/expert/combine performance.
2. Add plotted sweeps over larger MoE shapes where dispatch overhead is less
   dominant.
3. Add `megablocks_dmoe` rows for zero-bias runs, then investigate nonzero
   expert-bias support.
4. Add checkpoint-like router distributions before interpreting
   `tokens_per_expert_max` and padding costs as model-level behavior.
5. Treat BF16 standard MoE as pending until the numeric/expert-path difference is
   explained.
