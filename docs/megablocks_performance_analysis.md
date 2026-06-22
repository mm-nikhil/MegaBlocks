# MegaBlocks Performance Analysis

This document consolidates the current MegaBlocks/Nano-MoE-JAX performance
analysis, verification method, timing boundaries, formulas, hardware assumptions,
and promoted RTX 3080 results.

## Executive Summary

This project measures one Nano-MoE-JAX MoE layer on an NVIDIA GeForce RTX 3080,
using MegaBlocks to execute the selected expert path where NanoJAX semantics can
be matched.

The benchmark is intentionally not an end-to-end language-model benchmark. It
does not include token embedding, attention, layer norm outside the MoE boundary,
logits, sampling, optimizer work, or text generation. In every result here,
`N` means MoE-layer input-token hidden rows:

```text
N = batch_size * seq_len
```

Nano-MoE-JAX is the semantic source of truth. A PyTorch reference mirrors the
NanoJAX MoE math so MegaBlocks, which is a PyTorch/CUDA library, can be checked
against it. The current promoted runs use trained NanoJAX checkpoint weights,
including the trained expert biases, and time the full Nano-compatible MoE-layer
boundary with CUDA events.

Current RTX 3080 findings:

- The exact Nano adapter target is small: `D=128`, `H=512`, `E=4`, `K=2`,
  FFN experts, `gelu_tanh`.
- Useful selected-expert work is `524,288` FLOPs per input token.
- Standard MegaBlocks `moe` runs in FP32 through `N=65,536`; larger rows hit the
  observed standard-MoE binned-copy grid limit or memory preflight.
- Grouped MegaBlocks `dmoe` runs through `N=1,048,576`, but in this checkout the
  grouped GEMM extension is BF16-only.
- Hardware-utilization plots use a configured RTX 3080 CUDA-core FP32 roof of
  `36.82 TFLOP/s`. BF16 dMoE rows are still normalized to that denominator unless
  `peak_flops_per_sm_cycle` is changed.

## What MegaBlocks Is

MegaBlocks is a PyTorch/CUDA library for efficient mixture-of-experts execution.
Its local checkout provides standard `MoE` and dropless `dMoE` layers. Upstream
MegaBlocks emphasizes dropless MoE execution: routing tokens to selected experts
without dropping overflow tokens and using sparse/block-sparse or grouped expert
computation to keep hardware efficiency high.

For this project, MegaBlocks is not given the JAX model directly. The adapter:

1. Loads or constructs Nano-compatible MoE weights.
2. Computes Nano-compatible routing, expert indices, and gates.
3. Feeds those selected routes into MegaBlocks dispatch, expert compute, and
   weighted combine.

So the result is a measured implementation mapping for one MoE layer, not an
automatic JAX-to-MegaBlocks compiler.

## Benchmark Scope

The measured semantic boundary is:

```text
input:  hidden states, shape (B, T, D)
output: hidden states, shape (B, T, D)
extra:  scalar auxiliary router load-balancing loss
```

Inside the current `moe_layer` timing scope:

```text
Nano layout -> router projection -> full router softmax for aux probabilities
            -> top-k over raw logits -> selected-logit softmax gates
            -> MegaBlocks expert dispatch/compute/combine
            -> Nano layout
```

The narrower `expert_path` scope is available for diagnostics. It starts after
routing has been prepared and times only MegaBlocks dispatch/sort/binning,
gather, expert MLP, weighted scatter/combine, and shared expert combine if
configured.

## Symbols And Shape

```text
B = batch size
T = sequence length
N = B * T input-token hidden rows at one MoE layer
D = hidden size / d_model
H = expert intermediate size / d_ff
E = number of routed experts
K = top-k routed experts per token
S = number of shared experts
H_shared = shared expert intermediate size
```

Current exact Nano shape:

```text
D = 128
H = 512
E = 4
K = 2
S = 0
expert_type = ffn
activation = gelu_tanh
max_position_embeddings = 128
```

Routing creates token-expert assignments:

```text
assignments = N * K
```

For example, `N=65,536` and `K=2` creates `131,072` routed token-expert
assignments.

## Model Size At One MoE Layer

At one isolated MoE layer, a larger benchmark can mean two different things.
Increasing `N` gives the same layer more input rows to process. It increases the
number of routed assignments, but it does not make one input row more expensive.
Increasing `D`, `H`, `E`, `K`, or changing the expert type changes the amount of
work represented by each row.

That distinction is necessary for interpreting MegaBlocks. A small Nano row and
an OLMoE-shaped row are not comparable just because they have the same `N`.
MegaBlocks overhead may dominate when each row has little expert math, while the
same overhead can be amortized when `D`, `H`, and `K` make the expert work large.

Useful shape-derived quantities:

```text
assignments = N * K

router_params = D * E
router_flops_per_input_row = 2 * D * E

FFN_routed_expert_params ~= 2 * E * D * H
FFN_active_expert_flops_per_input_row ~= K * 4 * D * H

GLU_routed_expert_params ~= 3 * E * D * H
GLU_active_expert_flops_per_input_row ~= K * 6 * D * H

FFN_shared_expert_flops_per_input_row ~= S * 4 * D * H_shared
GLU_shared_expert_flops_per_input_row ~= S * 6 * D * H_shared
```

The constants are not arbitrary. They come from the number of learned matrices
and the FLOP convention used for matrix multiplication:

```text
2 in router_flops:
  each router output logit is a length-D dot product
  multiply-add accounting counts one multiply and one add as 2 FLOPs
  E logits gives 2 * D * E FLOPs per input row

2 in FFN params:
  an FFN expert has W1: D x H and W2: H x D
  ignoring small bias vectors, that is D*H + H*D = 2*D*H weights per expert

3 in GLU params:
  a GLU expert has gate: D x H, up: D x H, and down: H x D
  ignoring bias vectors, that is 3*D*H weights per expert

4 in FFN FLOPs:
  an active FFN expert performs two matmuls
  x @ W1 costs 2*D*H FLOPs and hidden @ W2 costs 2*H*D FLOPs
  total = 4*D*H FLOPs per selected expert row

6 in GLU FLOPs:
  an active GLU expert performs three matmuls
  gate, up, and down each cost about 2*D*H FLOPs
  total = 6*D*H FLOPs per selected expert row
```

For the current Nano shape:

```text
router_flops_per_input_row = 2 * 128 * 4 = 1,024
FFN_active_expert_flops_per_input_row = 2 * 4 * 128 * 512
                                      = 524,288
dense_all_expert_FFN_flops_per_input_row = 4 * 4 * 128 * 512
                                          = 1,048,576
```

The dense reference computes all `E=4` experts and then gathers the selected
outputs. MegaBlocks computes selected experts, so useful sparse expert work is
based on `K=2`, not `E=4`.

## Input Contract

MegaBlocks consumes hidden activations and routing metadata. It does not consume
token ids, and it is not handed the JAX model object. The adapter receives Nano
hidden states in `(B, T, D)` layout, prepares the Nano-compatible router outputs,
uses MegaBlocks' expected `(T, B, D)` layout internally, and returns `(B, T, D)`.

```text
external input:        X, shape (B, T, D)
MegaBlocks layer view: x, shape (T, B, D)
flattened routing:    x_flat, shape (N, D)
output:               shape (B, T, D)
```

The layer configuration must provide the MoE geometry:

```text
hidden_size = D
ffn_hidden_size = H
moe_num_experts = E
moe_top_k = K
mlp_type = "mlp" for FFN experts, "glu" for GLU/SwiGLU experts
activation_fn = gelu_tanh, silu, or the catalog activation
shared_expert settings when S > 0
dtype and device
```

The learned tensors are the router projection and expert weights. For Nano FFN
experts, the mathematically relevant tensors are:

```text
router weight: W_router, shape (E, D)
expert W1:     E matrices of shape (D, H)
expert b1:     E vectors of shape (H)
expert W2:     E matrices of shape (H, D)
expert b2:     E vectors of shape (D)
```

Stock MegaBlocks experts are bias-free. Trained NanoJAX experts are not. Exact
trained-checkpoint comparison therefore requires bias-aware handling around the
MegaBlocks expert path.

## NanoJAX Semantics And Execution Boundary

For hidden input `x`, NanoJAX routing is:

```text
router_logits = x @ W_router
router_probs  = softmax(router_logits)
top_values, top_indices = top_k(router_logits, K)
gates = softmax(top_values)
```

`router_probs` is used for the auxiliary load-balancing loss. `top_indices` and
`gates` define the selected expert computation. The auxiliary loss is:

```text
aux_loss = E * sum_i(top1_token_fraction_i * mean_router_probability_i)
```

Nano FFN expert math is:

```text
hidden = gelu_tanh(x @ W1 + b1)
out    = hidden @ W2 + b2
```

The MoE output is:

```text
output[token] = sum_k gates[token, k] * expert_output[top_indices[token, k]]
```

The PyTorch reference mirrors this NanoJAX math. MegaBlocks is then used for the
selected expert path: dispatch routed rows to experts, run expert computation,
and gate-combine the selected outputs back to one row per input token. In
standard `moe`, this path pads each expert to the busiest expert's routed row
count before batched expert compute. In grouped `dmoe`, routed rows stay compact
and `tokens_per_expert` becomes the grouped GEMM batch-size list. In this
checkout, the grouped GEMM extension used by `dmoe` accepts BF16 rows only.

## Trained-Checkpoint Verification

Trained checkpoint weights are used to verify the adapter against realistic
router and expert parameters, including nonzero expert biases. The verification
ladder is:

```text
NanoJAX MoE layer
  -> PyTorch reference with the same MoE math
  -> MegaBlocks adapter with the same routing choices, gates, and expert weights
```

The comparison covers output tensors, auxiliary loss, selected expert sets,
router gates, and expert-path numeric error. This verifies the MoE layer
boundary only; it does not claim end-to-end language-model throughput.

Current promoted correctness gate:

| Case | Backend | Dtype | Threshold | Max abs vs reference | Aux-loss diff | Expert-set mismatches |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `nanojax_fp32_megablocks_moe` | `megablocks_moe` | FP32 | `0.001` | `0.0` | `0.0` | `0` |
| `nanojax_bf16_megablocks_dmoe` | `megablocks_dmoe` | BF16 | `0.02` | `0.0` | `0.0` | `0` |

For large token-capacity and hardware sweeps, correctness is checked once at
small `N` before timing. Large performance rows are not dense-reference checked
row by row unless row-level output checking is explicitly enabled.

## Timing Method

CUDA timing uses warmup iterations, CUDA events, and synchronization:

```text
warm up fn()
synchronize
record start event
run timed iterations
record end event
synchronize
mean_forward_ms = elapsed_event_time / iterations
```

The current promoted token-capacity run uses:

```text
device = cuda
seq_len = 128
timing_scope = moe_layer
warmup = 10
iters = 50
trials = 3
weight_source = trained_nano_checkpoint
requested_dtype = float32
```

`megablocks_dmoe` rows may use effective `dtype=bfloat16` with
`dtype_policy=dmoe_bf16_only_local_grouped_gemm` even when the requested dtype is
FP32.

## Hardware FLOP And Throughput Formulas

The hardware-facing FLOP metrics use selected expert GEMM work as the numerator.
This keeps the comparison focused on the useful MoE expert math and avoids
pretending that routing, top-k, sorting, and scatter/gather are cleanly
expressible as equivalent expert matmul FLOPs.

Define:

```text
m = expert FLOP multiplier per selected row
  = 4 for FFN experts
  = 6 for GLU/SwiGLU experts

A = N * K routed token-expert assignments
W_active = active useful selected-expert FLOPs
s = measured runtime in seconds
```

The multiplier `m` includes the multiply-add convention. FFN has two matmuls,
so `m = 2 matmuls * 2 FLOPs per multiply-add = 4`. GLU has three matmuls, so
`m = 3 * 2 = 6`.

For routed experts:

```text
W_routed = N * K * m * D * H
```

For shared experts:

```text
W_shared = N * S * m * D * H_shared
```

Total useful expert work:

```text
W_active = W_routed + W_shared
```

For the current Nano FFN shape, `m=4` and `S=0`:

```text
W_active_per_input_row = K * m * D * H
                       = 2 * 4 * 128 * 512
                       = 524,288 FLOPs

W_active = N * 524,288
```

Throughput metrics:

```text
s = mean_forward_ms / 1000
ms_per_input_token = mean_forward_ms / N
tokens_per_second = N / s
active_expert_tflops_per_second = W_active / s / 1e12
```

`mean_forward_ms` answers how long the measured boundary took. `ms_per_input_token`
normalizes by row count. `active_expert_tflops_per_second` normalizes by useful
expert math, which is the safer metric when comparing shapes with different
`D`, `H`, `K`, or expert type.

Backend-row accounting is separate from useful work:

```text
reference backend expert rows = N * E
standard moe expert rows = E * max(tokens_per_expert)
grouped dmoe expert rows = N * K
padding_factor = backend_expert_rows / (N * K)
```

For the dense Nano reference, `padding_factor = E/K = 4/2 = 2`. That is not
MegaBlocks padding. It records that the reference computed all experts while the
sparse MoE execution computes selected experts.

## Hardware Compute-Utilization Formula

The clock-derived hardware metric compares `W_active` against a theoretical GPU
compute roof over the measured runtime. Define:

```text
W = W_active, useful selected-expert FLOPs
t = measured runtime in seconds
f = SM clock in cycles/second
P = number of SMs
R = peak FLOPs per SM-cycle
```

The denominator is the number of FLOPs the GPU could theoretically issue during
the measured interval under the configured compute roof:

```text
theoretical_compute_capacity = t * f * P * R
clock_compute_utilization = W / theoretical_compute_capacity
```

The same quantity can be read as a cycle ratio:

```text
ideal_cycles_per_SM = W / (P * R)
elapsed_cycles_per_SM = t * f
clock_compute_utilization = ideal_cycles_per_SM / elapsed_cycles_per_SM
```

The current RTX 3080 denominator uses `R=256` FLOPs per SM-cycle. That value
comes from `128` FP32 CUDA lanes per SM and `2` FLOPs per fused multiply-add.

Derived presentation fields:

```text
clock_estimated_unused_compute = 1 - clock_compute_utilization
clock_equivalent_unused_SMs = P * clock_estimated_unused_compute
```

These are algebraic roofline-style estimates. They are not measured SM active
cycles, hardware occupancy, physical idle-SM counts, memory stalls, or Nsight
counter results.

## Hardware Configuration

Local hardware and software observed for this analysis:

```text
GPU: NVIDIA GeForce RTX 3080
Driver: 580.126.09
Compute capability: 8.6
SM count: 68
Memory: 10,351,935,488 bytes reported by Torch, 10,240 MiB by nvidia-smi
Current SM clock sampled before/after run: 225 MHz
Max SM clock used for denominator: 2115 MHz
Python: 3.10.12
Torch: 2.7.0+cu126
Torch CUDA runtime: 12.6
Repo commit: 839daa0
MegaBlocks submodule: 952db33
grouped_gemm submodule: f1429a3
Nano-MoE-JAX submodule: a41cc95
```

Configured RTX 3080 FP32 CUDA-core roof:

```text
P = 68 SMs
R = 256 FLOPs / SM-cycle
f = 2.115e9 cycles / second

peak_compute_capacity = f * P * R
                      = 2.115e9 * 68 * 256
                      = 36.81792e12 FLOPs / second
                      = 36.82 TFLOP/s
```

If we want a BF16/Tensor Core roof, `R` must be changed to the appropriate
Tensor Core FLOPs per SM-cycle. The current mixed FP32/BF16 hardware table does
not do that.

## Current Token-Capacity Results

| N | Reference dense FFN FP32 ms | MegaBlocks MoE FP32 ms | MegaBlocks dMoE BF16 ms | Notes |
| ---: | ---: | ---: | ---: | --- |
| 512 | 0.2974 | 0.5348 | 0.6984 | Sparse overhead dominates this tiny Nano shape. |
| 8,192 | 1.1829 | 0.8512 | 0.7421 | Both MegaBlocks paths are faster than the dense reference here. |
| 65,536 | 8.5228 | 5.2770 | 3.8051 | Largest successful standard-MoE row in the promoted sweep. |
| 262,144 | 33.6239 | failed | 14.4782 | Standard MoE failed earlier; dMoE continues. |
| 1,048,576 | not run successfully | failed | 56.5065 | Largest successful dMoE row in the promoted sweep. |

Max successful `N` by backend:

| Backend variant | Effective dtype | Max successful N | Main limitation after that |
| --- | --- | ---: | --- |
| `reference_dense_ffn` | FP32 | 262,144 | Memory preflight rejects larger dense all-expert rows. |
| `megablocks_moe` | FP32 | 65,536 | Standard-MoE binned-copy grid limit at larger rows, then memory preflight. |
| `megablocks_dmoe` | BF16 | 1,048,576 | Largest requested row succeeded. |

Interpretation:

- MegaBlocks is not expected to look best at very small Nano sizes because the
  useful expert GEMMs are small and dispatch/routing overhead is comparatively
  large.
- Standard MoE has a padding factor near `1.0` at large balanced rows, but it is
  still limited by the binned gather/scatter launch shape observed in this
  checkout.
- Grouped dMoE avoids the standard-MoE padded expert-capacity tensor and scales
  further on this 10 GiB RTX 3080, with the BF16-only local grouped GEMM caveat.

## Current Hardware-Clock Results

The token-capacity and hardware-clock sweeps are separate runs, so their
per-row times are close but not bit-identical.

| N | Backend | Dtype | Mean ms | Active expert TFLOP/s | Clock compute util | Equivalent unused SMs |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 512 | `megablocks_moe` | FP32 | 0.5458 | 0.4918 | 1.34% | 67.09 / 68 |
| 8,192 | `megablocks_moe` | FP32 | 0.8525 | 5.0381 | 13.68% | 58.70 / 68 |
| 65,536 | `megablocks_moe` | FP32 | 5.2691 | 6.5210 | 17.71% | 55.96 / 68 |
| 65,536 | `megablocks_dmoe` | BF16 | 3.8818 | 8.8514 | 24.04% | 51.65 / 68 |
| 262,144 | `megablocks_dmoe` | BF16 | 14.4958 | 9.4813 | 25.75% | 50.49 / 68 |
| 1,048,576 | `megablocks_dmoe` | BF16 | 56.4866 | 9.7325 | 26.43% | 50.02 / 68 |

Interpretation:

```text
Based on measured MoE-layer runtime and the configured RTX 3080 FP32 compute
roof, useful selected-expert FLOPs occupy X% of the available compute slots.
```

This metric does not measure:

```text
GPU idle cycles
SM occupancy
physical idle-SM count
memory stalls
kernel launch overhead
```

## Results/Plots

- [Token-capacity plot](/home/nikhil/workspace/MegaBlocks/results/current/token_capacity/nano_moe_jax/graphs_token_capacity.png)
- [Hardware-clock plot](/home/nikhil/workspace/MegaBlocks/results/current/hardware_clock/nano_moe_jax/graphs_clock_compute.png)
