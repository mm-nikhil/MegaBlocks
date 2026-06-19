# Hardware Details

## Goal

We want one simple presentation metric for the NanoJAX MoE layer:

```text
How much of the RTX 3080's theoretical compute capacity is occupied by the
useful MoE expert computation during the measured MoE runtime?
```

This does not require external GPU performance counters. It uses only:

```text
measured runtime
GPU SM clock
number of SMs
assumed peak FLOPs per SM per cycle
estimated useful MoE expert FLOPs
```

Call the metric:

```text
clock-derived compute utilization
```

Do not call it observed GPU idle cycles. It is a roofline-style estimate from
time, clock, GPU size, and useful work.

## Local GPU

Current local device:

```text
GPU: NVIDIA GeForce RTX 3080
Architecture: Ampere
Compute capability: 8.6
SM count: 68
Memory: about 10 GiB
```

For this project:

```text
PE = processing element = SM = Streaming Multiprocessor
```

We use SMs as the PE unit because the GPU's execution resources and clock-domain
capacity are cleanly expressed at the SM level for this presentation.

The server currently reports:

```text
current SM clock: 225 MHz
max SM clock: 2115 MHz
```

Use the max SM clock as the default denominator:

```text
f = 2115 MHz = 2.115e9 cycles / second
```

The current clock can be an idle-state value when sampled before or after a run.
If clocks are locked or measured separately during the run, use that measured
clock instead.

## MoE Timing Boundary

The measured time `t` should correspond to the MoE layer boundary we want to
present.

For the full NanoJAX MoE layer, the boundary is:

```text
hidden input [N x D]
router projection [D x E]
top-k selection
softmax over selected router logits
expert block for selected experts
gate multiply
combine / reduce back to [N x D]
```

This is the `moe_layer` timing scope in our profiling code. It is the right
default for presentation because it matches the whole MoE layer diagram.

The timed MegaBlocks path keeps NanoJAX MoE semantics fixed: router choices and
gates are computed with NanoJAX-compatible math, then the selected experts are
executed by MegaBlocks through sparse dispatch, expert MLP compute, and weighted
scatter/combine.

## Per-Op Timing Diagnostics

The hardware sweep can also record separate diagnostic timings with
`--moe-op-profile`. These fields are separate from the clock-derived hardware
metric and should be used only to explain where MoE-layer time is going.

The diagram-level timing labels are:

```text
moe_op_input_layout_to_megablocks_ms
moe_op_router_projection_matmul_ms
moe_op_router_full_softmax_ms
moe_op_topk_selection_ms
moe_op_selected_softmax_gating_ms
moe_op_router_aux_loss_ms
moe_op_expert_path_dispatch_compute_combine_ms
moe_op_output_layout_to_nano_ms
moe_op_disjoint_replay_sum_ms
moe_op_whole_moe_layer_replay_ms
moe_op_replay_sum_minus_whole_ms
```

These fields are a disjoint diagnostic replay of logical MoE blocks. The
component sum is reported, and a whole replay is also reported, but neither
replaces the authoritative production timing in `mean_forward_ms`. The
MegaBlocks expert path is kept as one implementation unit:
dispatch/sort/binning/gather, expert MLP, weighted scatter/combine, and shared
expert combine if configured. Lower-level gather/MLP/scatter timings belong to
the separate implementation phase profile.

## Useful Work

For this metric, useful work means selected expert MLP math:

```text
W = active_expert_flops
```

For NanoJAX FFN experts:

```text
D = hidden size / d_model
H = expert intermediate size / d_ff
K = top-k experts per token
N = batch_size * seq_len
```

Each selected expert row performs two matrix multiplications:

```text
x @ W1: [1 x D] * [D x H]
h @ W2: [1 x H] * [H x D]
```

Counting multiply-add as two FLOPs:

```text
FLOPs per selected expert row = 4 * D * H
active_expert_flops_per_token = K * 4 * D * H
active_expert_flops = N * K * 4 * D * H
```

For NanoJAX:

```text
D = 128
H = 512
K = 2

active_expert_flops_per_token = 2 * 4 * 128 * 512
                              = 524,288 FLOPs
```

So:

```text
active_expert_flops = N * 524,288
```

This intentionally does not add router projection, top-k, softmax, dispatch,
sort, gather, scatter, or combine into the FLOP count. Those operations are real
runtime costs, but they are not cleanly represented as expert matmul FLOPs. If
the measured time includes them, the utilization number correctly goes down:
the useful expert math occupied a smaller fraction of the total MoE-layer time.

## Capacity Model

Use:

```text
W = useful work, in FLOPs
t = measured runtime, in seconds
f = SM clock, in cycles per second
P = number of SMs / PEs
R = peak FLOPs per SM per cycle
```

Then:

```text
elapsed_cycles_per_SM = t * f
total_SM_cycle_slots = t * f * P
theoretical_compute_capacity = t * f * P * R
```

The utilization metric is:

```text
clock_compute_utilization = W / (t * f * P * R)
```

Equivalent forms:

```text
ideal_cycles_per_SM = W / (P * R)
clock_compute_utilization = ideal_cycles_per_SM / elapsed_cycles_per_SM
```

Estimated unused compute capacity:

```text
clock_estimated_unused_compute = 1 - clock_compute_utilization
clock_equivalent_unused_SMs = P * clock_estimated_unused_compute
```

This matches the whiteboard idea:

```text
cycles = work / (PEs * ops_per_PE_per_cycle)
```

For a dot product with `n` multiplies and about `n - 1` additions:

```text
W ~= 2n FLOPs
cycles ~= 2n / (P * R)
```

## RTX 3080 Assumption

For a simple RTX 3080 CUDA-core FP32/FMA roof:

```text
P = 68 SMs
R = 256 FLOPs per SM-cycle
  = 128 FP32 lanes per SM * 2 FLOPs per FMA
```

So for the local RTX 3080:

```text
peak_compute_capacity_per_second = f * P * R
                                 = 2.115e9 * 68 * 256
                                 = 36.82e12 FLOPs / second
                                 = 36.82 TFLOP/s
```

This is a theoretical compute roof. It is useful as a denominator for a clean
presentation metric.

For a BF16/Tensor Core roof, `R` should be changed to the appropriate Tensor
Core FLOPs per SM-cycle. That will produce a lower utilization percentage
because the peak denominator is larger. For now, use the CUDA-core FP32/FMA roof
unless we explicitly decide to present a Tensor Core roof.

## Example

For `N = 16384`:

```text
W = 16384 * 524,288
  = 8,589,934,592 FLOPs
```

Using:

```text
P = 68
R = 256 FLOPs / SM-cycle
f = 2.115e9 cycles / second
```

For `megablocks_moe`, measured time was about:

```text
t = 0.875 ms = 0.000875 seconds
```

Then:

```text
theoretical_compute_capacity = t * f * P * R
                             ~= 0.000875 * 2.115e9 * 68 * 256
                             ~= 32.2e9 FLOPs

clock_compute_utilization = W / theoretical_compute_capacity
                           ~= 8.59e9 / 32.2e9
                           ~= 26.7%
```

Presentation row:

```text
N=16384, backend=megablocks_moe
clock-derived compute utilization: 26.7%
estimated unused compute capacity: 73.3%
equivalent unused SMs: 49.9 / 68
```

For `megablocks_dmoe`, measured time was about:

```text
t = 1.186 ms
clock-derived compute utilization ~= 19.7%
estimated unused compute capacity ~= 80.3%
equivalent unused SMs ~= 54.6 / 68
```

## Interpretation

Correct interpretation:

```text
Based on measured runtime and the RTX 3080 theoretical compute roof, the useful
NanoJAX MoE expert FLOPs occupy about X% of available compute slots.
```

This is simple, reproducible, and explainable from the quantities we already
have.

Do not interpret it as:

```text
the GPU was physically idle X% of cycles
all SMs were actually unused for X% of time
measured warp occupancy
measured memory stalls
measured dispatch/sort overhead
```

Those are different hardware-counter questions. Our chosen metric is a
theoretical compute-slot utilization estimate.

## What To Plot

For hardware-style presentation plots, use:

```text
x-axis: N = batch_size * seq_len
y-axis: clock_compute_utilization %
series: backend
```

Useful table columns:

```text
N
backend
mean_forward_ms
active_expert_flops
sm_clock_mhz
sm_count
peak_flops_per_sm_cycle
clock_elapsed_cycles_per_sm
clock_ideal_expert_cycles_per_sm
clock_compute_util_pct
clock_estimated_unused_compute_pct
clock_equivalent_unused_sms
```

This keeps the result focused on the one metric we agreed to present.
