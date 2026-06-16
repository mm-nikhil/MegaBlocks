# RTX 3080 NanoMoE Token Sweep

Date: 2026-06-15  
Host: `aorus`  
GPU: NVIDIA GeForce RTX 3080, 10 GB  
Driver: 580.126.09  
Torch: 2.7.0+cu126, Torch CUDA 12.6  
Raw JSONL: `results/raw/rtx3080_nano_token_sweep_20260615.jsonl`

Submodules:

```text
Nano-MoE-JAX: a41cc95
MegaBlocks:   952db33
grouped_gemm: f1429a3
```

## Scope

This sweep keeps the Nano-MoE-JAX MoE layer shape fixed:

```text
d_model=128
d_ff=512
n_experts=4
top_k=2
seq_len=128
```

Only `tokens = batch_size * seq_len` changes for the FP32 rows. The FP16 rows are
the baseline `tokens=4096` smoke comparison.

Token count is a workload variable. In deployment it can increase through larger
batches, batched prefill, longer prompts, sequence packing, offline evaluation,
or training batches. During interactive autoregressive decoding, token count
mainly increases through serving batch size because each step usually adds one
new token per active sequence.

MegaBlocks timing uses the default `megablocks_core` scope:

```text
time MegaBlocks dispatch / expert / combine
```

Correctness is checked against the PyTorch NanoMoE reference after timing.

## Command

```bash
.venv/bin/python src/profiling/sweep_moe_layer.py \
  --preset focused \
  --tokens 512,1024,2048,4096,8192,16384 \
  --seq-len 128 \
  --d-models 128 \
  --d-ffs 512 \
  --n-experts 4 \
  --top-ks 2 \
  --dtypes float32,float16 \
  --backends reference,megablocks_moe \
  --device cuda \
  --warmup 10 \
  --iters 50 \
  --trials 3 \
  --jsonl-out results/raw/rtx3080_nano_token_sweep_20260615.jsonl \
  --continue-on-error
```

## Results

`speedup` is reference latency divided by backend latency for the same shape.

| tokens | d | ff | experts | k | dtype | backend | ms | std | speedup | mem_delta_MB | max_abs | router_flips | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 128 | 512 | 4 | 2 | float32 | reference | 0.3225 | 0.0011 | 1.00 | 8.0 | - | - | - |
| 512 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0494 | 0.0002 | 0.31 | 4.7 | 0 | 0 | within_threshold |
| 1024 | 128 | 512 | 4 | 2 | float32 | reference | 0.3217 | 0.0008 | 1.00 | 16.1 | - | - | - |
| 1024 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0491 | 0.0002 | 0.31 | 9.3 | 0 | 0 | within_threshold |
| 2048 | 128 | 512 | 4 | 2 | float32 | reference | 0.3703 | 0.0001 | 1.00 | 32.2 | - | - | - |
| 2048 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0492 | 0.0002 | 0.35 | 18.9 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float32 | reference | 0.6431 | 0.0006 | 1.00 | 64.3 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0493 | 0.0001 | 0.61 | 36.7 | 0 | 0 | within_threshold |
| 8192 | 128 | 512 | 4 | 2 | float32 | reference | 1.1808 | 0.0041 | 1.00 | 128.6 | - | - | - |
| 8192 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.0633 | 0.0198 | 1.11 | 73.3 | 0 | 0 | within_threshold |
| 16384 | 128 | 512 | 4 | 2 | float32 | reference | 2.2399 | 0.0082 | 1.00 | 257.3 | - | - | - |
| 16384 | 128 | 512 | 4 | 2 | float32 | megablocks_moe | 1.9881 | 0.1587 | 1.13 | 144.9 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float16 | reference | 0.3466 | 0.0001 | 1.00 | 32.2 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float16 | megablocks_moe | 1.0494 | 0.0002 | 0.33 | 18.4 | 0 | 0 | within_threshold |

## Read

- MegaBlocks standard MoE is slower for the exact Nano layer at small token
  counts because dispatch overhead dominates.
- FP32 crossover appears at `tokens=8192` in this run.
- At `tokens=16384`, MegaBlocks standard MoE is `1.13x` faster than the dense
  PyTorch reference.
- All MegaBlocks rows in this sweep passed output checks with `max_abs=0` and
  zero router expert-set mismatches.
- MegaBlocks used less peak memory delta than the dense reference for each FP32
  token count.

## Plots

- `results/plots/rtx3080_nano_token_sweep_20260615/latency_vs_tokens.png`
- `results/plots/rtx3080_nano_token_sweep_20260615/speedup_vs_tokens.png`
