# Verification

This document explains how we verify the Nano-MoE-JAX MoE layer, why the
PyTorch reference exists, and what the trained-weight verification run proves.

## Verification Target

The project verifies one MoE layer boundary:

```text
input:  hidden states, shape (batch, seq_len, d_model)
output: hidden states, shape (batch, seq_len, d_model)
extra:  scalar auxiliary router load-balancing loss
```

The full language model, optimizer, attention layers, embeddings, and text
quality are outside this boundary unless a script explicitly says otherwise.

For Nano-MoE-JAX, the MoE layer does this:

```text
router logits = x @ W_router
top-k expert ids = top_k(router logits)
gates = softmax(selected top-k logits)
expert output = expert_i(x)
MoE output = sum_k gates[k] * expert_output[top_k_id[k]]
aux loss = load-balancing penalty from router probabilities and top-1 choices
```

`aux_loss` is not the language-model cross-entropy loss. It is a scalar penalty
that encourages the router to use experts more evenly.

## Why PyTorch Is In The Middle

MegaBlocks is a PyTorch/CUDA library. Its layers are `torch.nn.Module` objects,
its weights are `torch.nn.Parameter` tensors, its inputs are `torch.Tensor`
objects, and its CUDA/Triton extensions are launched through PyTorch.

Nano-MoE-JAX is a JAX/Flax model. A JAX `MoELayer` cannot be passed directly to
MegaBlocks.

So the verification ladder is:

```text
Nano-MoE-JAX source of truth
  -> PyTorch NanoMoE reference
  -> MegaBlocks adapter
```

The PyTorch reference is not a third model. It is the literal Nano-MoE-JAX MoE
math written in the framework MegaBlocks can interoperate with.

## Step 1: Original NanoJAX vs PyTorch Reference

Command:

```bash
.venv/bin/python src/profiling/check_nano_moe_port.py
```

What it does:

```text
1. Create a real Nano-MoE-JAX MoELayer.
2. Initialize Flax params.
3. Run the JAX layer on a deterministic hidden-state tensor.
4. Convert the same Flax params to PyTorch tensors.
5. Run the PyTorch reference.
6. Compare output, aux loss, router gates, and router expert indices.
```

Current result on this workspace:

```text
all 4 checks passed
max_abs(output): up to 9.53674e-07 in the smoke cases
indices_equal: True
```

One caveat: exactly tied router logits can have framework-specific top-k order.
The verifier has a tie diagnostic for this. Normal non-tied cases are checked
with exact expert-index equality.

## Step 2: PyTorch Reference vs MegaBlocks

Command:

```bash
.venv/bin/python src/profiling/verify_moe_layer.py
```

What it does:

```text
1. Build Nano-compatible weights.
2. Build the MegaBlocks MoE or dMoE layer.
3. Copy router and expert weights into MegaBlocks.
4. Compute Nano-compatible router choices and gates.
5. Run MegaBlocks dispatch/expert/combine.
6. Compare against the PyTorch reference.
```

Current result on this workspace:

```text
all 4 checks passed
moe_fp32_nano_init: output_max_abs_error=0
moe_fp16_nano_init: output_max_abs_error=0
moe_fp32_synthetic_bias: output_max_abs_error=0
dmoe_bf16_zero_bias: output_max_abs_error=0
router_set_mismatches=0 for all cases
```

MegaBlocks is not called as `layer(x)` for Nano verification. The adapter
prepares Nano-compatible routing first, then feeds the expert ids and gates into
MegaBlocks dispatch. This keeps the router semantics aligned with NanoJAX.

## Step 3: Trained NanoJAX Weights

Some audiences expect trained weights even though implementation correctness
does not require them. For that reason, this repo now includes a checkpointed
trained-weight verifier:

```bash
.venv/bin/python src/profiling/verify_trained_nano_moe.py
```

Recommended workflow:

```text
1. Train NanoJAX once and save a reusable checkpoint.
2. Run verification repeatedly from that saved checkpoint.
```

The checkpoint contains:

```text
metadata.json  - model config, data mode, training losses, training settings
params.msgpack - full trained NanoJAX/Flax parameter tree
```

Train once:

```bash
.venv/bin/python src/profiling/verify_trained_nano_moe.py \
  --mode train \
  --train-steps 1000 \
  --batch-size 32 \
  --block-size 128 \
  --n-layers 4 \
  --n-heads 4 \
  --d-model 128 \
  --d-ff 512 \
  --n-experts 4 \
  --top-k 2 \
  --data-mode tiny_shakespeare \
  --checkpoint-dir results/trained_nano_moe_checkpoint
```

Verify later without training:

```bash
.venv/bin/python src/profiling/verify_trained_nano_moe.py \
  --mode verify_saved \
  --checkpoint-dir results/trained_nano_moe_checkpoint \
  --verify-batch-size 2 \
  --verify-seq-len 128 \
  --device cuda \
  --dtype float32 \
  --megablocks-layer moe \
  --artifact-dir results/trained_nano_moe_verification \
  --json-out results/trained_nano_moe_verification/verification.json \
  --save-tensors
```

What `verify_saved` does:

```text
1. Load the saved full NanoJAX params and model config.
2. Extract a trained MoE layer, by default block_0/MoELayer_0.
3. Run the trained model prefix up to that block's MoE input.
4. Run the original JAX MoE layer on that real hidden-state input.
5. Convert the trained MoE weights to PyTorch.
6. Run the PyTorch reference.
7. Copy the trained weights into MegaBlocks.
8. Run MegaBlocks.
9. Compare output tensors, aux loss, gates, and expert choices.
```

For quick smoke checks, `--mode train_and_verify` still trains and verifies in
one command. Use `--mode train` plus `--mode verify_saved` for reusable results.

Current saved-checkpoint verification result:

```text
mode: verify_saved
checkpoint_dir: results/trained_nano_moe_checkpoint
train_steps: 1000
data_mode: tiny_shakespeare
loss: 4.25299 -> 1.99969
trained_expert_bias_max_abs: 0.043312

JAX vs PyTorch output:
  exact_equal: False
  max_abs: 5.72205e-06
  mean_abs: 4.01386e-07

PyTorch vs MegaBlocks output:
  exact_equal: True
  max_abs: 0
  mean_abs: 0

router_expert_set_mismatches: 0
router_gate_max_abs: 0
aux_loss_abs_diff: 0
correctness_passed: True
```

The outputs are not bit-exact. That is expected across JAX CPU, PyTorch CUDA, and
MegaBlocks CUDA because floating-point operation order and kernels differ. The
important result is that differences are around `1e-6`, no output element exceeds
the `1e-3` threshold, router expert sets match exactly, gates match, and aux loss
matches.

The run also saved tensors under:

```text
results/trained_nano_moe_verification/
```

Saved files include the MoE input, JAX output, PyTorch output, MegaBlocks output,
router indices, and router gates.

## Trained Scaling Verification

After saving a trained checkpoint, we can sweep verification batch size without
retraining:

```bash
.venv/bin/python src/profiling/verify_trained_nano_moe.py \
  --mode verify_saved \
  --checkpoint-dir results/trained_nano_moe_checkpoint \
  --verify-seq-len 128 \
  --sweep-batch-sizes 4,8,16,32,64,128 \
  --device cuda \
  --dtype float32 \
  --megablocks-layer moe \
  --timing-warmup 10 \
  --timing-iters 100 \
  --timing-trials 3 \
  --artifact-dir results/trained_nano_moe_verification \
  --jsonl-out results/trained_nano_moe_verification/scaling_verify.jsonl
```

Current saved-checkpoint scaling result:

```text
N      PyTorch ref ms  MegaBlocks ms  speedup  correct
512    0.313583        0.589848       0.53x    yes
1024   0.352433        0.535675       0.66x    yes
2048   0.410519        0.541065       0.76x    yes
4096   0.639340        0.631484       1.01x    yes
8192   1.168391        0.983733       1.19x    yes
16384  2.214143        1.669369       1.33x    yes
```

Artifacts:

```text
results/trained_nano_moe_verification/scaling_verify.jsonl
results/trained_nano_moe_verification/scaling_verify_summary.csv
results/trained_nano_moe_verification/scaling_verify_dashboard.png
```

## Trained Token-Capacity Dashboard

For the promoted NanoJAX token-capacity dashboard, the profiler loads the saved
trained NanoJAX checkpoint, extracts `block_0/MoELayer_0`, and uses scalable
random hidden-state inputs at the MoE layer boundary. This keeps the run
comparable to the original token-capacity dashboard without running the full JAX
transformer prefix at very large batch sizes.

```bash
.venv/bin/python src/profiling/run_model_token_capacity.py \
  --model-shape-name nano_moe_jax \
  --result-root results/current/token_capacity \
  --tokens 512,1024,2048,4096,8192,16384,32768,65536,131072,262144 \
  --seq-len 128 \
  --backends reference,megablocks_moe,megablocks_dmoe \
  --dtype bfloat16 \
  --device cuda \
  --warmup 5 \
  --iters 20 \
  --trials 2 \
  --timing-scope auto \
  --weight-source trained_nano_checkpoint \
  --checkpoint-dir results/trained_nano_moe_checkpoint \
  --checkpoint-block-index 0 \
  --outlier-abs-threshold 0.02
```

The BF16 threshold is explicit because grouped `dmoe` requires BF16 in this
checkout. The promoted run has `check_output=True`; all MegaBlocks rows that ran
passed, with zero router mismatches, zero aux-loss difference, and max output
absolute error `0.015625`.

Artifacts:

```text
results/current/token_capacity/nano_moe_jax/raw.jsonl
results/current/token_capacity/nano_moe_jax/summary.csv
results/current/token_capacity/nano_moe_jax/dashboard.png
results/current/token_capacity/nano_moe_jax/backend_status.md
```

## How To Present This

Use this concise story:

```text
We verify the MoE layer in two stages. First, a literal PyTorch port is checked
against the original NanoJAX layer. Then MegaBlocks is checked against that
PyTorch reference using the same weights, same hidden inputs, same router choices,
and same gates. We also trained NanoJAX once on Tiny Shakespeare, saved the
checkpoint, and extracted trained MoE weights. The trained-weight checks produced
matching router choices; the FP32 single-layer check matched within ~1e-6, and
the BF16 token-capacity dashboard passes with max absolute error 0.015625 under
an explicit 0.02 BF16 tolerance.
```

Training is useful for audience confidence and realistic nonzero expert biases.
It is not required to prove the implementation math: arbitrary initialized
weights are already valid inputs for a deterministic equivalence check.
