# Usage

Run commands from the repo root.

## Verify NanoMoE Semantics

This compares a real Nano-MoE-JAX `MoELayer` against the PyTorch reference port.

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
```

Expected result:

```text
indices_equal:   True
all checks passed
```

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

To run the current smoke matrix and store JSONL:

```bash
scripts/run_smoke_matrix.sh
```

The script writes to `results/raw/...jsonl` by default. Raw files are ignored by
git; promote curated summaries into `results/`.

## MegaBlocks Timing

After `megablocks_ops` and `grouped_gemm` are built:

```bash
.venv/bin/python src/profiling/profile_moe_layer.py \
  --backend megablocks \
  --megablocks-layer moe \
  --dtype float32 \
  --zero-expert-biases \
  --check-output
```

`--zero-expert-biases` is intentional. Stock MegaBlocks expert MLPs are bias-free,
while Nano-MoE-JAX experts have Dense biases. Do not call this an exact NanoMoE
port unless the source model/checkpoint has zero expert biases or MegaBlocks bias
support has been implemented.

The standard `moe` float32 path is the current correctness smoke. Use lower
precision and `dmoe` runs for timing experiments only after reviewing the printed
output-check metrics.

## Useful Variants

Dropless grouped dMoE, currently not yet semantically validated:

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
