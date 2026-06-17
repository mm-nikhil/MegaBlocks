# Metrics And Results

This document defines the benchmark vocabulary, result metrics, simulation
levels, and current result interpretation. MegaBlocks implementation details live
in `docs/moe_megablocks_deep_dive.md`.

## Token Rows

In this project, `N` means input-token hidden rows at one MoE layer:

```text
N = B * T
B = batch size
T = sequence length
```

These are not generated output text tokens. The MoE layer consumes hidden rows
and returns hidden rows:

```text
input:  (B, T, D)
output: (B, T, D)
```

Routing creates token-expert assignments:

```text
assignments = N * K
K = top-k routed experts per token
```

For example, if `N=4096` and `K=8`, the layer has `32768` routed expert
assignments.

## Shape Fields

Every result should report the shape fields needed to interpret timing:

```text
D = hidden size / d_model
H = expert intermediate size
E = number of routed experts
K = active routed experts per token
S = shared experts
H_shared = shared expert hidden size, if present
expert_type = ffn or glu
activation = gelu_tanh, silu, etc.
dtype
timing_scope
```

Raw latency or tokens/sec is not comparable across models unless these fields
are visible. A tiny model can have high tokens/sec while doing much less useful
math per token than a larger model.

## Required Metrics

`mean_forward_ms`:

```text
Average timed forward call for the selected timing scope.
```

For `reference` backends, this is the reference PyTorch boundary. For
`megablocks_core`, routing is prepared before timing and the timed call is the
MegaBlocks dispatch/expert/scatter path.

`ms_per_input_token`:

```text
mean_forward_ms / N
```

This is latency normalized by input-token hidden row. It is useful within one
shape sweep, but it is not enough for cross-model comparison because larger
models do more compute per row.

`assignments`:

```text
N * K
```

This is the number of routed token-expert pairs.

`active_expert_flops_per_token`:

For FFN experts:

```text
4 * K * D * H
```

For GLU experts:

```text
6 * K * D * H
```

Shared experts add:

```text
4 * S * D * H_shared for FFN
6 * S * D * H_shared for GLU
```

`active_expert_tflops_per_second`:

```text
(N * active_expert_flops_per_token) / seconds / 1e12
```

This is the main compute-normalized throughput metric for comparing shape
scaling. It answers: given the useful active expert math implied by this MoE
configuration, how much of that work per second did the backend deliver?

`padding_factor`:

```text
backend_expert_rows / assignments
```

Interpretation:

```text
grouped dMoE: usually 1.0 in our current grouped path
standard moe: E * max(tokens_per_expert) / (N*K)
dense reference: (N*E) / (N*K) = E/K
```

For dense references, this is not MegaBlocks padding. It means the reference
computed all experts, not just top-k experts.

## Phase Metrics

Phase metrics are diagnostic only. They help explain where time goes inside
MegaBlocks but should not be treated as an exact additive decomposition of
`mean_forward_ms`.

Current phase fields:

```text
phase_sort_ms
phase_histogram_ms
phase_cumsum_ms
phase_capacity_decision_wall_ms
phase_gather_ms
phase_expert_mlp_ms
phase_scatter_ms
phase_gpu_sum_ms
phase_expert_capacity
```

Use phase metrics to answer questions such as:

```text
Is the flat small-N region dominated by capacity sync or dispatch kernels?
Does expert MLP time dominate once D/H/N grow?
Does standard moe spend extra time in padded gather/scatter?
```

## Correctness Metrics

Correctness metrics are for verification, not headline plots:

```text
max_abs_vs_reference
mean_abs_vs_reference
router_expert_set_mismatch_count
router_gate_max_abs
correctness_passed
```

For exact-adapter runs, correctness must pass before timing is interpreted.
For Level 1 synthetic model-shape runs, correctness means internal consistency
against the intended synthetic reference, not exact model checkpoint execution.

## Timing Scopes

`adapter_boundary` includes adapter layout conversion and routing work.

`megablocks_core` prepares routing before timing, then times the MegaBlocks
dispatch/expert/combine path:

```text
sort/histogram/cumsum
gather
expert MLP
scatter/combine
shared expert, if enabled
```

Do not compare scopes as if they are the same workload.

## Simulation Levels

`exact_adapter`:

```text
A local PyTorch reference and MegaBlocks adapter are intended to match the model
semantics at the MoE layer. Nano is the current exact-adapter target.
```

`level0_shape`:

```text
Shape-only simulation. Useful for memory and rough dispatch geometry, but not
for expert-type or router claims.
```

`level1_shape_expert`:

```text
Uses catalog shape fields plus expert type and activation with synthetic
weights. OLMoE-shaped runs are currently Level 1. These are not checkpoint or
exact-router runs.
```

`level2_router_semantic`:

```text
Adds model-specific router behavior. This is needed before claiming exact
OLMoE or DeepSeek routing behavior.
```

`level3_weight_faithful`:

```text
Loads real checkpoint weights and model-specific semantics. This is out of
scope for the current first benchmark pass.
```

## Plot Contract

`token_capacity` is the main comparison dashboard. It uses four panels:

```text
mean_forward_ms vs N
ms_per_input_token vs N
active_expert_tflops_per_second vs N
padding_factor vs N
```

It should request the relevant backend families:

```text
reference
megablocks_moe
megablocks_dmoe
```

If a backend fails or is unsupported, record it in `backend_status.md`; do not
silently omit it.

`phase_profile` is diagnostic only. It should show phase timings, not duplicate
the token-capacity dashboard.

## Current Results

Result files live under:

```text
results/current/
```

Per-run context is in:

```text
notes.md
backend_status.md
summary.csv
raw.jsonl
```

### Nano-MoE-JAX

Current backend variants:

```text
reference_dense_ffn
megablocks_moe
megablocks_dmoe
```

Nano is the exact-adapter target. It is small:

```text
D=128 H=512 E=4 K=2 expert_type=ffn
```

Because the useful expert math is small, fixed sparse-dispatch overhead is large
relative to compute at small `N`. This is why small Nano timing should not be
summarized as "MegaBlocks is bad for small models." The more precise claim is:
for this tiny Nano shape, fixed dispatch and capacity costs dominate until there
are enough rows to amortize them.

Standard `megablocks_moe` fails at high Nano `N` when its binned-gather
`expert_capacity` exceeds the observed Triton grid-y limit. This is a
MegaBlocks implementation finding and is explained in
`docs/moe_megablocks_deep_dive.md`.

Grouped `megablocks_dmoe` avoids that standard-moe binned grid limit in the
current runs.

### OLMoE-1B-7B-0924 Shape

Current backend variants:

```text
reference_dense_glu
megablocks_moe
megablocks_dmoe
```

This result is:

```text
simulation_level = level1_shape_expert
weights = synthetic
D=2048 H=1024 E=64 K=8 expert_type=glu activation=silu dtype=bf16
```

Important caveat:

```text
This is OLMoE-shaped synthetic GLU simulation, not exact OLMoE checkpoint or
router execution.
```

The green `reference_dense_glu` point is a PyTorch CUDA dense all-expert GLU
baseline. It uses OLMoE-shaped dimensions and synthetic weights. It computes all
`E=64` GLU experts for every input-token row, then selects the routed outputs.

Observed reference behavior:

```text
N=4096: succeeds, about 62.59 ms on CUDA
N=8192: memory preflight rejects before allocation
```

At `N=8192`, preflight estimated about `10.56 GB` against about `8.86 GB`
allowed on this GPU. Larger `N` values are rejected more strongly. On a larger
GPU, the dense reference could produce more green points, but it would take more
time and memory because dense all-expert GLU scales with `N * E`.

MegaBlocks OLMoE-shaped behavior:

```text
megablocks_moe max successful N = 49152
megablocks_dmoe max successful N = 49152
N=65536 rejected by memory preflight on this GPU
```

Standard `megablocks_moe` for this OLMoE-shaped GLU run uses a local adapter GLU
wrapper on the standard padded `moe` dispatch path. This is not a claim that
stock standard `moe` directly implements OLMoE.

## Model Catalog

The current shape catalog is:

```text
configs/moe_model_shapes.json
```

Current entries:

```text
nano_moe_jax
olmoe_1b_7b_0924
deepseek_v3_moe_layer
```

Use catalog values directly when possible. If a value is not in the catalog,
either add a sourced field or compute it from existing fields with a documented
formula.
