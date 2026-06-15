#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

OUT="${1:-results/raw/smoke_$(date -u +%Y%m%dT%H%M%SZ).jsonl}"
mkdir -p "$(dirname "$OUT")"

run_profile() {
  python src/profiling/profile_moe_layer.py \
    --batch-size 32 \
    --seq-len 128 \
    --d-model 128 \
    --d-ff 512 \
    --n-experts 4 \
    --top-k 2 \
    --warmup 10 \
    --iters 50 \
    --jsonl-out "$OUT" \
    "$@"
}

run_profile --label reference_fp16 --backend reference --device cuda --dtype float16

run_profile \
  --label megablocks_moe_fp32 \
  --backend megablocks \
  --megablocks-layer moe \
  --zero-expert-biases \
  --check-output \
  --dtype float32

run_profile \
  --label megablocks_moe_fp16 \
  --backend megablocks \
  --megablocks-layer moe \
  --zero-expert-biases \
  --check-output \
  --dtype float16

run_profile \
  --label megablocks_dmoe_bf16 \
  --backend megablocks \
  --megablocks-layer dmoe \
  --zero-expert-biases \
  --check-output \
  --dtype bfloat16

echo "matrix_jsonl=$OUT"
