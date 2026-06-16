# MoE Performance Mapping Plan

This document defines the profiling target, configuration terms, and next
measurement phases for the Nano-MoE-JAX + MegaBlocks prototype.

## Target

The measured unit is one MoE layer. The full Transformer model, tokenizer,
attention, optimizer, and training loop are outside the current benchmark.

```text
input:    (batch_size, seq_len, d_model)
output:   (batch_size, seq_len, d_model)
aux loss: scalar load-balancing loss
```

Nano-MoE-JAX is the semantic source of truth. The PyTorch reference is a literal
MoE-layer port used for comparison. The MegaBlocks adapter maps Nano-compatible
weights, routing, gates, and layout into MegaBlocks execution.

## Current Execution Paths

```text
reference:
  PyTorch NanoMoE reference, dense all-expert computation

megablocks_moe:
  Nano-compatible routing + MegaBlocks standard MoE dispatch/expert/combine

megablocks_dmoe:
  Nano-compatible routing + MegaBlocks dropless MoE path, currently zero-bias only
```

Default MegaBlocks timing scope:

```text
megablocks_core = MegaBlocks dispatch/expert/combine only
```

Full adapter timing scope:

```text
adapter_boundary = Nano layout conversion + routing + MegaBlocks execution + output layout
```

Correctness checks use the full adapter output regardless of timing scope.

## Configuration Terms

```text
tokens    = batch_size * seq_len
seq_len   = sequence length inside one forward call
d_model   = hidden width entering and leaving the MoE layer
d_ff      = inner width of each expert MLP
n_experts = number of experts in the layer
top_k     = number of selected experts per token
dtype     = activation/weight dtype for the run
backend   = reference, megablocks_moe, or megablocks_dmoe
```

`tokens` is total layer work in one forward call. It can increase through larger
batch size, longer sequence length, or both.

## Deployment Meaning Of Token Scaling

Token count is a workload variable, not a model-quality knob.

```text
tokens = batch_size * seq_len
```

Increasing token count for a benchmark means the MoE layer processes more token
positions in one forward call. This can happen in deployment through larger
batches, batched prefill, longer prompts, sequence packing, offline evaluation,
or training batches.

Interactive autoregressive decoding usually processes one new token per active
sequence per step. In that phase, token count mainly increases through serving
batch size, not through making a single generated token larger.

Token-scaling results answer this deployment question:

```text
Is the expected workload large enough to amortize MegaBlocks routing/dispatch overhead?
```

They do not imply that a model should receive unnecessary extra tokens. Larger
token counts increase total work and latency; the relevant metric is whether
MegaBlocks becomes more efficient than the dense reference for the same workload.

## Nano Baseline

Nano-MoE-JAX default configuration:

```text
n_layers:    4
n_heads:     4
d_model:     128
d_ff:        512
n_experts:   4
top_k:       2
block_size:  128
batch_size:  32
tokens:      4096
```

Source: `third_party/Nano-MoE-JAX/nano_moe/config.py`.

Exact Nano-layer benchmarking should keep `d_model=128`, `d_ff=512`,
`n_experts=4`, and `top_k=2`. Token scaling can vary `batch_size`, `seq_len`, or
both while keeping the same MoE layer shape.

## Scaling Study

Changing `d_model`, `d_ff`, `n_experts`, or `top_k` creates a different MoE layer
shape. This is useful for performance mapping because it shows where routed MoE
execution starts to offset dispatch overhead.

Dense reference expert work estimate:

```text
tokens * n_experts * d_model * d_ff
```

MegaBlocks active expert work estimate:

```text
tokens * top_k * d_model * d_ff + dispatch/combine overhead
```

The dense reference mirrors Nano-MoE-JAX's simple nano-scale implementation: run
all experts, then gather the selected expert outputs. MegaBlocks evaluates the
routed execution path.

## Public MoE Config Anchors

These public model configs show that larger MoE models commonly vary the same
axes used by the sweep.

| model | hidden width | expert width | experts | top-k | source |
| --- | ---: | ---: | ---: | ---: | --- |
| Nano-MoE-JAX | 128 | 512 | 4 | 2 | local config |
| Mixtral-8x7B | 4096 | 14336 | 8 | 2 | [config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json) |
| Qwen1.5-MoE-A2.7B | 2048 | 1408 | 60 | 4 | [config](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blame/e5335224ea59a3d50e17f0113118b305d5eda11b/config.json) |
| DeepSeek-V2-Lite | 2048 | 1408 | 64 routed + 2 shared | 6 | [config](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json) |

RTX 3080 experiments should stay below memory limits. Public model shapes are
anchors for scaling direction, not required sweep targets for this GPU.

## dMoE Status

MegaBlocks describes dMoE as its core dropless MoE path. dMoE uses block-sparse
operations to avoid token dropping and reduce padding waste.

Current adapter status:

```text
standard MoE + nonzero Nano expert biases: supported
dMoE + zero expert biases: supported in smoke tests
dMoE + nonzero Nano expert biases: pending
```

Nano expert MLPs include `b1` before GELU and `b2` after the second projection.
dMoE correctness for trained Nano checkpoints requires either dMoE bias support,
a biasless model variant, or an explicit semantic mismatch record.

## Measurement Phases

Phase 1: Nano exact baseline.

```text
fixed: d_model=128, d_ff=512, n_experts=4, top_k=2
vary:  tokens, dtype, backend
```

Phase 2: standard MoE scaling.

```text
vary one axis at a time:
tokens, d_model, d_ff, n_experts, top_k

compare:
reference vs megablocks_moe
```

Phase 3: dMoE investigation.

```text
start with zero expert biases
compare megablocks_moe vs megablocks_dmoe
then investigate nonzero expert-bias support
```

Phase 4: larger scaling grid.

```text
reference rows: include while memory/runtime are practical
MegaBlocks rows: continue after dense reference becomes impractical
```

## Default Plot Set

Generate one plot group per axis:

```text
latency_vs_tokens
latency_vs_d_ff
latency_vs_n_experts
latency_vs_d_model
speedup_vs_tokens
speedup_vs_d_ff
speedup_vs_n_experts
speedup_vs_d_model
```

Primary y-axis:

```text
mean_forward_ms
```

Secondary comparison:

```text
speedup = reference_mean_forward_ms / backend_mean_forward_ms
```

Values above `1.0` mean the backend is faster than the dense reference for the
same layer shape.

## Report Contents

Each curated report should include:

- GPU, driver, Torch CUDA runtime, and submodule commits.
- Sweep command.
- Timing scope.
- Shape and dtype columns.
- Mean/std forward latency.
- Memory delta after warmup.
- Correctness status and max absolute output error.
- Router expert-set mismatch count.
- BF16 or dMoE caveats when present.
- Plot links for the selected sweep.
