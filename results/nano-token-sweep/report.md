# NanoMoE Token Sweep

Date: 2026-06-16  
Host: `aorus`  
GPU: NVIDIA GeForce RTX 3080, 10 GB  
Raw JSONL: `results/raw/rtx3080_nano_token_sweep_with_cpu_20260616.jsonl`

## Scope

Fixed Nano-MoE-JAX MoE layer shape:

```text
d_model=128, d_ff=512, n_experts=4, top_k=2, seq_len=128, dtype=float32
```

Rows:

```text
reference/cpu:        dense PyTorch NanoMoE reference on CPU
reference/cuda:       dense PyTorch NanoMoE reference on GPU
megablocks_moe/cuda:  MegaBlocks standard MoE adapter on GPU
```

JAX is not timed here. JAX remains the semantic source used by
`check_nano_moe_port.py`.

## Command Shape

The CUDA and CPU rows were run as separate sweeps into the same JSONL:

```bash
.venv/bin/python src/profiling/sweep_moe_layer.py \
  --preset grid \
  --tokens 512,1024,2048,4096,8192,16384 \
  --seq-len 128 \
  --d-models 128 \
  --d-ffs 512 \
  --n-experts 4 \
  --top-ks 2 \
  --dtypes float32 \
  --weight-source nano_jax_init \
  --warmup 3 \
  --iters 10 \
  --trials 2
```

CUDA sweep used `--backends reference,megablocks_moe --device cuda`. CPU sweep
used `--backends reference --device cpu`.

## Results

Speedup is `reference/cuda ms / backend ms`.

| tokens | reference/cuda ms | megablocks_moe/cuda ms | reference/cpu ms | MegaBlocks speedup vs CUDA ref |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.3339 | 1.0376 | 1.9342 | 0.32x |
| 1024 | 0.3356 | 1.0392 | 4.9616 | 0.32x |
| 2048 | 0.3790 | 1.0390 | 10.5434 | 0.36x |
| 4096 | 0.6533 | 1.0397 | 26.0400 | 0.63x |
| 8192 | 1.1926 | 1.0382 | 50.6691 | 1.15x |
| 16384 | 2.2562 | 2.0832 | 104.9297 | 1.08x |

All MegaBlocks rows passed output checks with `max_abs=0` and zero router
expert-set mismatches.

## Plots

- `latency_vs_tokens.png`
- `speedup_vs_tokens.png`

## Read

The CPU reference is much slower than the CUDA reference for this exact layer.
The earlier interpretation remains unchanged: the meaningful GPU comparison is
dense PyTorch reference on CUDA vs MegaBlocks on CUDA. CPU reference rows are
useful as a device baseline, not as the main MegaBlocks comparison.
