# RTX 3080 Correctness Smoke Results

Date: 2026-06-15
Host GPU: NVIDIA GeForce RTX 3080, 10 GB
Driver: 580.126.09
CUDA compiler: not on `PATH` for this run
Python: 3.10.12
Torch: 2.7.0+cu126
Torch CUDA runtime: 12.6

Submodules:

```text
a41cc95bf22cd7b6e14ff47517902f8157e5c641 third_party/Nano-MoE-JAX
f1429a3c44c98f7912aa4b00125144cdf4e7fdb2 third_party/grouped_gemm
952db33d6eac334d22c61e47a0d5d41446298784 third_party/megablocks
```

Shape:

```text
batch=32
seq=128
tokens=4096
d_model=128
d_ff=512
n_experts=4
top_k=2
weight_source=nano_jax_init
warmup=10
iters=50
```

## Semantic Check

Command:

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
```

Result:

```text
preset: smoke cases=4
[1/4] batch=2 seq=8 d_model=32 d_ff=64 experts=4 top_k=1 seed=0 max_abs(output)=4.76837e-07 max_abs(gates)=0 abs(aux_loss)=0 indices_equal=True
[2/4] batch=2 seq=8 d_model=32 d_ff=64 experts=4 top_k=2 seed=0 max_abs(output)=7.15256e-07 max_abs(gates)=1.19209e-07 abs(aux_loss)=0 indices_equal=True
[3/4] batch=2 seq=8 d_model=32 d_ff=64 experts=4 top_k=4 seed=0 max_abs(output)=9.53674e-07 max_abs(gates)=1.78814e-07 abs(aux_loss)=0 indices_equal=True
[4/4] batch=3 seq=5 d_model=16 d_ff=32 experts=4 top_k=2 seed=1 max_abs(output)=9.53674e-07 max_abs(gates)=8.9407e-08 abs(aux_loss)=0 indices_equal=True
all 4 checks passed
```

## Timing Smoke

Command:

```bash
scripts/run_smoke_matrix.sh results/raw/review_fixed_smoke_nowarn.jsonl
```

Summary:

| tokens | d | ff | experts | k | dtype | weights | backend | ms | std | tok/s | speedup | TF/s | mem_delta_MB | max_abs | router_flips | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 128 | 512 | 4 | 2 | bfloat16 | nano_jax_init | megablocks_dmoe | 1.0493 | 0.0000 | 3903429 | 0.33 | 2.051 | 19.2 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | bfloat16 | nano_jax_init | reference | 0.3495 | 0.0000 | 11719894 | 1.00 | 12.301 | 32.2 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float16 | nano_jax_init | megablocks_moe | 1.0493 | 0.0000 | 3903581 | 0.33 | 2.086 | 19.5 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float16 | nano_jax_init | reference | 0.3464 | 0.0000 | 11825244 | 1.00 | 12.412 | 32.2 | - | - | - |
| 4096 | 128 | 512 | 4 | 2 | float32 | nano_jax_init | megablocks_moe | 1.0494 | 0.0000 | 3903124 | 0.61 | 2.086 | 38.9 | 0 | 0 | within_threshold |
| 4096 | 128 | 512 | 4 | 2 | float32 | nano_jax_init | reference | 0.6445 | 0.0000 | 6355258 | 1.00 | 6.670 | 64.3 | - | - | - |

Interpretation:

- These timings include the public `(B, T, D)` adapter boundary and Nano-style auxiliary loss.
- MegaBlocks rows use Nano-compatible top-k-on-logits routing before MegaBlocks dispatch/expert/combine.
- Output, router expert sets, router gates, and auxiliary loss match the PyTorch Nano reference for the tested smoke rows.
- This is still one MoE layer with Nano-JAX initialized weights, not full model training or checkpoint performance.
