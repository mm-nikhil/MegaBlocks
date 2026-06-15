# Usage

Run commands from the repo root.

## Verify NanoMoE Semantics

This compares a real Nano-MoE-JAX `MoELayer` against the PyTorch reference port.
It answers: did we port the Nano-MoE-JAX MoE math correctly into PyTorch?

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
```

Expected result:

```text
preset: smoke cases=4
all 4 checks passed
```

Use `--preset single` for one explicit shape. Use `--tie-diagnostic` to print the
known framework-specific ordering for exactly tied router logits.

Then verify the MegaBlocks adapter against the PyTorch reference:

```bash
.venv/bin/python src/profiling/verify_moe_layer.py
```

This answers: given Nano-compatible routing and weights, does the MegaBlocks
dispatch/expert/combine path produce the same MoE output and aux loss as the
PyTorch reference? Use `--preset single` for one explicit shape and `--verbose`
for detailed router/error diagnostics.

## Reference Timing

This runs the exact PyTorch reference semantics. It is useful for checking shapes
and timing harness behavior, but it is not MegaBlocks performance.

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend reference \
  --device cuda \
  --batch-size 32 \
  --seq-len 128 \
  --d-model 128 \
  --d-ff 512 \
  --n-experts 4 \
  --top-k 2 \
  --dtype float16
```

By default, profiling uses `--weight-source nano_jax_init`: a real Nano-MoE-JAX
`MoELayer` is initialized with Nano's Flax initializers, then converted into the
PyTorch/MegaBlocks adapter. Use `--weight-source synthetic` only when you
intentionally want synthetic `N(0, 0.02)` weights.

To run the current smoke matrix and store JSONL:

```bash
scripts/run_smoke_matrix.sh
```

The script writes to `results/raw/...jsonl` by default. Raw files are ignored by
git; promote curated summaries into `results/`.

To run a focused MoE-layer sweep:

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
  --trials 3 \
  --jsonl-out results/raw/sweep.jsonl
```

The default `--preset focused` varies one axis at a time around the baseline
instead of running the full Cartesian product. A count like `18/18` means 18
benchmark configurations for the same one-layer MoE boundary, not 18 model
layers. Start with `--backends reference,megablocks_moe`; add `megablocks_dmoe`
only when explicitly investigating dMoE.

Summarize the JSONL as a compact table:

```bash
.venv/bin/python src/profiling/summarize_moe_sweep.py results/raw/sweep.jsonl
```

Use `--show-throughput` only when you explicitly want tokens/sec and estimated
TFLOP/s columns. The default table focuses on latency and correctness.

Use `--dry-run` to inspect commands and `--limit N` for a small test. Use
`--preset grid` only for a deliberate full Cartesian sweep.

## MegaBlocks Timing

After `megablocks_ops` and `grouped_gemm` are built:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --megablocks-layer moe \
  --dtype float32 \
  --use-expert-biases \
  --timing-scope megablocks_core \
  --check-output
```

`--use-expert-biases` is the exact NanoMoE standard-MoE adapter. It uses
Nano-compatible routing with MegaBlocks dispatch/expert/combine. If expert
biases are nonzero, it swaps the stock bias-free expert MLP for a bias-aware
batched expert MLP. Nano-JAX initialized biases are zero, so the stock expert MLP
is kept for those runs.

Default MegaBlocks timing uses `--timing-scope megablocks_core`: routing and
adapter layout conversion are prepared outside the timed region, and the timed
call is MegaBlocks dispatch/expert/combine. Use `--timing-scope adapter_boundary`
only when measuring the full Nano-compatible adapter boundary.

The default console output reports timing plus actual correctness errors. Values
such as reference output scale and reference aux-loss value are stored in JSONL,
but are printed only with `--verbose-checks`.

The standard `moe` float32 path with `--use-expert-biases` is the current
correctness smoke. Lower precision and `dmoe` rows should still be interpreted
only with their printed output, router, and aux-loss check metrics.

## Useful Variants

Dropless grouped dMoE with Nano-JAX initialized zero expert biases:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --megablocks-layer dmoe \
  --dtype bfloat16 \
  --zero-expert-biases \
  --check-output
```

BF16:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --dtype bfloat16 \
  --zero-expert-biases
```

Synthetic nonzero expert biases for stress-testing the standard MoE bias-aware
adapter:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --megablocks-layer moe \
  --weight-source synthetic \
  --use-expert-biases \
  --check-output
```
