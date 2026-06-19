# MegaBlocks MoE/dMoE Deep Dive

This document is only about how MegaBlocks implements MoE execution in this
checkout: what inputs it needs, how routing is represented, what standard `moe`
and `dmoe` do, which kernels are used, and what limitations we have observed.

Benchmark metrics, plots, result interpretation, and benchmark scope labels live in
`docs/metrics_and_results.md`. Work planning lives in `docs/next_steps.md`.

## Symbols

```text
B = batch size
T = sequence length
N = input-token hidden rows at one MoE layer = B * T
D = hidden size / d_model
H = expert intermediate size / ffn_hidden_size
E = number of routed experts
K = top-k routed experts per token
S = number of shared experts
```

MegaBlocks consumes hidden activations, not tokenizer ids:

```text
external adapter input: X, shape (B, T, D)
MegaBlocks layer input: x, shape (T, B, D)
flattened routing view: x_flat, shape (N, D)
```

The MoE layer returns one hidden row per input hidden row:

```text
output shape = (T, B, D) inside MegaBlocks
adapter returns (B, T, D)
```

There are no generated output text tokens inside this isolated MoE-layer
benchmark.

## Model Size At One MoE Layer

For one isolated MoE layer, a bigger benchmark can mean two different things.

More batch or sequence length gives more input-token rows:

```text
N = B * T
```

This makes the layer process more rows, but it does not by itself make each row
more expensive.

A bigger MoE layer means each input row carries more routing, expert, and
dispatch work. The main model-shape axes are:

```text
D = hidden size
H = expert intermediate size
E = number of routed experts
K = active routed experts per input row
expert_type = FFN/MLP or GLU/SwiGLU
H_shared = shared expert intermediate size, if a shared expert is present
```

So `128k` Nano rows and `128k` OLMoE-shaped rows are not equal work. They have
the same `N`, but not the same compute per row. Nano has small `D`, `H`, `E`,
and `K`; OLMoE-shaped GLU uses much larger hidden dimensions, many more experts,
and larger `K`.

Useful shape-derived quantities are:

```text
router_params = D * E

FFN routed_expert_params ~= 2 * E * D * H
GLU routed_expert_params ~= 3 * E * D * H

router_flops_per_input_row ~= 2 * D * E

FFN active_expert_flops_per_input_row ~= K * 4 * D * H
GLU active_expert_flops_per_input_row ~= K * 6 * D * H

shared_expert_flops_per_input_row ~= 4 * D * H_shared for FFN shared expert
shared_expert_flops_per_input_row ~= 6 * D * H_shared for GLU shared expert
```

The constants in these formulas come from weight count and multiply-add count:

```text
2 in FFN params:
  an FFN expert has two learned matrices:
  W1: D x H
  W2: H x D
  so params per expert ~= 2 * D * H

3 in GLU params:
  a GLU/SwiGLU expert has three learned matrices:
  W_gate: D x H
  W_up:   D x H
  W_down: H x D
  so params per expert ~= 3 * D * H

2 in router flops:
  one matrix multiply row x router_weight uses multiply-add accounting:
  D multiplies + D adds per expert ~= 2 * D * E

4 in FFN active expert flops:
  two FFN matmuls, each ~= 2 * D * H
  total per active expert ~= 4 * D * H

6 in GLU active expert flops:
  three GLU matmuls, each ~= 2 * D * H
  total per active expert ~= 6 * D * H
```

These are approximate GEMM-centered FLOP counts. They intentionally ignore
smaller elementwise work such as activation, GLU multiply, bias add, top-k, and
scatter/gather indexing. They are useful because they separate "more rows" from
"more work per row." Cross-shape comparisons should therefore look at both
latency per input row and compute-normalized throughput.

MegaBlocks can be fed these shape fields through `Arguments`. For Level 1 shape
simulation, we use synthetic weights with the model's MoE shape:

```text
hidden_size = D
ffn_hidden_size = H
moe_num_experts = E
moe_top_k = K
mlp_type = "mlp" for FFN, "glu" for GLU
activation_fn from the catalog
shared_expert/shared_expert_hidden_size from the catalog
```

That tells us how MegaBlocks behaves for the MoE-layer geometry. It does not
claim exact model semantics unless the model-specific router, weight layout, and
checkpoint weights are also matched.

## Input Contract

MegaBlocks is not given a Hugging Face or JAX model object directly. We construct
a MegaBlocks layer from `megablocks.layers.arguments.Arguments`, then copy or
initialize the needed weights.

The main fields we provide are:

```text
hidden_size = D
ffn_hidden_size = H
moe_num_experts = E
moe_top_k = K
activation_fn = GELU, SiLU, or another callable
mlp_type = "mlp" or "glu"
mlp_impl = "grouped" or "sparse"
shared_expert = true or false
shared_expert_hidden_size = H_shared when shared_expert is true
fp16 / bf16 dtype flags
device
```

`mlp_type` is the expert formula type, not a separate routing type, `mlp_type="mlp"` means
a two-matmul FFN expert, while `mlp_type="glu"` means a three-projection
GLU/SwiGLU-style expert.

The required learned tensors are:

```text
router weight: W_router, shape (E, D) in MegaBlocks Linear layout

two-matmul MLP expert:
  w1, shape depends on path, represents E matrices of (D, H)
  w2, shape depends on path, represents E matrices of (H, D)

GLU expert:
  w1 / gate, represents E matrices of (D, H)
  v1 / up,   represents E matrices of (D, H)
  w2 / down, represents E matrices of (H, D)
```

For our profiler, model configs are mapped as:

```text
hidden_size -> D
expert_intermediate_size -> H
num_routed_experts -> E
num_experts_per_token -> K
expert_type = "ffn" -> mlp_type="mlp"
expert_type = "glu" -> mlp_type="glu"
activation = "gelu_tanh" or "silu" -> activation_fn
num_shared_experts/shared_expert_intermediate_size -> shared expert settings
```

Nano exactness additionally needs bias-aware wrappers because Nano experts use
Dense biases and MegaBlocks stock experts are bias-free. The current
OLMoE-shaped benchmark uses synthetic GLU weights and no exact
checkpoint/router semantics.

For Nano FFN experts, the trained NanoJAX expert math is:

```text
hidden = gelu(x @ W1 + b1)
out = hidden @ W2 + b2
```

Stock MegaBlocks FFN expert math is:

```text
hidden = gelu(x @ W1)
out = hidden @ W2
```

So trained Nano weights with nonzero `b1`/`b2` cannot be copied into stock
MegaBlocks FFN experts and still be mathematically equivalent. The router can
select the same experts and the main matmuls can use the same `W1`/`W2`, but the
expert function is missing terms unless the biases are modeled.

Our trained Nano bias adapter preserves correctness by adding those per-expert
biases around the MegaBlocks dispatch/expert/combine path. For standard `moe`,
this means adding `b1` and `b2` in the padded expert-major layout. For grouped
`dmoe`, this means recovering the expert id for each routed row and adding
`b1[expert_id]` and `b2[expert_id]` after the grouped GEMMs.

This makes the trained-with-bias run correctness-equivalent to NanoJAX, but it
is not pure stock MegaBlocks expert compute. The no-bias trained run is only a
performance control: it measures the stock bias-free path with trained `W1`/`W2`
and routing, but it is not equivalent to trained NanoJAX when expert biases are
nonzero.

## Source Findings

This section is the source-backed answer to "what does MegaBlocks do,
exactly?" for this checkout.

### Layer Construction Contract

MegaBlocks layers are configured through `Arguments`, not through a complete
model object:

```text
third_party/megablocks/megablocks/layers/arguments.py:22
```

The fields that determine the MoE execution shape are `hidden_size`,
`ffn_hidden_size`, `moe_num_experts`, `moe_top_k`, `moe_capacity_factor`,
`activation_fn`, `mlp_type`, `mlp_impl`, dtype flags, and shared-expert fields.
`Arguments.__post_init__` also enforces two important runtime facts:

```text
mlp_impl="sparse" is rejected with Triton >= 3.2.
mlp_impl="grouped" requires grouped_gemm to be available.
shared_expert_hidden_size defaults to ffn_hidden_size.
```

Our profiler builds this object in:

```text
src/profiling/profile_moe_layer.py:590
```

The important local choices are:

```text
moe_capacity_factor = 0
moe_normalize_expert_weights = 1
mlp_type = "mlp" for FFN, "glu" for GLU
mlp_impl = "grouped"
shared_expert = true when the catalog shape has shared experts
```

`moe_capacity_factor=0` means "do not use a configured fixed capacity." In
standard `moe`, MegaBlocks then uses the busiest expert's routed count as
capacity, which avoids token dropping but introduces padding.

### Stock Router Semantics

The stock router is `LearnedRouter`:

```text
third_party/megablocks/megablocks/layers/router.py:61
```

It owns a bias-free linear projection:

```text
W_router: (E, D)
logits = x_flat @ W_router.T
scores = softmax(logits)
expert_weights, expert_indices = topk(scores, K)
```

The source lines are:

```text
Linear(D, E): router.py:72
logits:       router.py:96
softmax:      router.py:98
top-k:        router.py:87 and router.py:99
```

So stock MegaBlocks routing uses PyTorch `softmax` and PyTorch `topk`. There is
no custom MegaBlocks exponential/softmax kernel in this path. Optional jitter,
expert-weight normalization, and uniform benchmark assignment are also handled
inside `LearnedRouter`.

### Softmax Implementation

In stock MegaBlocks routing, softmax is implemented by PyTorch, not by a
MegaBlocks custom kernel:

```text
scores = logits.softmax(dim=-1)
```

Source:

```text
third_party/megablocks/megablocks/layers/router.py:96
third_party/megablocks/megablocks/layers/router.py:98
```

Mathematically, this means the router converts each input row's expert logits
into probabilities across the `E` routed experts:

```text
scores[n, e] = exp(logits[n, e]) / sum_j exp(logits[n, j])
```

Then MegaBlocks selects top-k experts from those probabilities:

```text
expert_weights, expert_indices = topk(scores, K)
```

Source:

```text
third_party/megablocks/megablocks/layers/router.py:87
third_party/megablocks/megablocks/layers/router.py:90
third_party/megablocks/megablocks/layers/router.py:99
```

If `moe_normalize_expert_weights` is set, MegaBlocks normalizes the selected
expert weights after top-k:

```text
expert_weights = expert_weights / norm(expert_weights, p=moe_normalize_expert_weights)
```

Source:

```text
third_party/megablocks/megablocks/layers/router.py:100
third_party/megablocks/megablocks/layers/router.py:101
third_party/megablocks/megablocks/layers/router.py:102
third_party/megablocks/megablocks/layers/router.py:103
third_party/megablocks/megablocks/layers/router.py:104
third_party/megablocks/megablocks/layers/router.py:105
```

For our current profiler's `megablocks_core` timing scope, routing is prepared
outside the timed loop. That means the measured MegaBlocks core latency excludes
router linear, softmax, top-k, and adapter aux-loss setup. The timed core path
starts from already-computed `router_probs`, `gates`, and `indices`, then
measures expert dispatch, gather, expert compute, scatter, and shared expert
work if configured.

### Our Adapter Router Semantics

For Nano-compatible and Level 1 synthetic profiling, our adapter does not call
`layer(x)` directly. It prepares routing, then calls `layer.experts(...)`:

```text
src/profiling/profile_moe_layer.py:738
src/profiling/profile_moe_layer.py:761
src/profiling/profile_moe_layer.py:887
```

The adapter routing convention is:

```text
router_probs = softmax(logits)
top_indices = topk(logits, K)
gates = softmax(top-k logits)
```

Source:

```text
src/profiling/profile_moe_layer.py:826
src/profiling/profile_moe_layer.py:827
src/profiling/profile_moe_layer.py:828
src/profiling/profile_moe_layer.py:829
```

This differs from stock MegaBlocks, which selects top-k over `softmax(logits)`.
For exact Nano comparisons this matters because Nano-style routing chooses
experts from logits and then normalizes only the selected logits. In eval mode,
the `scores` tensor passed into `layer.experts` is only used for load-balancing
loss bookkeeping if training/loss is enabled; the timed dispatch uses
`expert_weights` and `top_experts`.

### Standard `moe` Is Padded Sparse Dispatch Plus Dense Batched Expert MLP

Standard `moe` is built by `MoE` and `ParallelMLP`:

```text
third_party/megablocks/megablocks/layers/moe.py:96
third_party/megablocks/megablocks/layers/moe.py:440
```

The stock high-level flow is:

```text
MoE.forward:
  scores, expert_weights, top_experts = router(x)
  out = experts(x, scores, expert_weights, top_experts)
```

Source:

```text
MoE.forward:          moe.py:459
ParallelMLP.forward:  moe.py:425
```

Inside `ParallelMLP.forward_once`, MegaBlocks flattens the `N*K` routed
assignments and builds expert bins:

```text
top_experts_flat -> sort -> histogram -> inclusive_cumsum
```

Source:

```text
indices_and_bins: moe.py:152
forward_once:     moe.py:209
```

The standard path then computes an expert capacity:

```text
expert_capacity = moe_capacity_factor * (K * N * world_size / E)
if expert_capacity == 0:
    expert_capacity = max(tokens_per_expert)
```

Source:

```text
expert_capacity: moe.py:133
max capacity:    moe.py:218
```

With our current profiler settings, that becomes:

```text
expert_capacity = max(tokens_per_expert)
```

The gather step creates a rectangular expert-major tensor:

```text
x_binned: (E, expert_capacity, D)
```

Source:

```text
permute_and_compute: moe.py:185
ops.binned_gather:   moe.py:198
```

The stock standard expert MLP is not a sparse matmul. It is dense `torch.bmm`
over that padded expert-major tensor:

```text
torch.bmm(x, w1)
activation(x)
torch.bmm(x, w2)
```

Source:

```text
third_party/megablocks/megablocks/layers/mlp.py:91
third_party/megablocks/megablocks/layers/mlp.py:162
```

Then `binned_scatter` writes expert outputs back to token order and applies
router weights:

```text
out[token] = sum_k gate[token,k] * expert_output[token,k]
```

Source:

```text
ops.binned_scatter: moe.py:207
backend scatter sum: backend/kernels.py:421
```

Interpretation: standard `moe` is sparse because only `K` experts per input row
are activated. But its expert compute is dense batched GEMM over padded bins, not
a general sparse matrix multiply.

### Standard `moe` Padding And The 65535 Finding

The standard binned gather/scatter kernel launches one Triton program grid over:

```text
(num_experts, expert_capacity)
```

Source:

```text
backend/kernels.py:392
backend/kernels.py:405
backend/kernels.py:421
backend/kernels.py:434
```

At Nano `N=131072`, we observed:

```text
tokens_per_expert = [65345, 65518, 65669, 65612]
expert_capacity = 65669
grid = (4, 65669)
```

That exceeds the observed CUDA/Triton grid-y limit of `65535` for this launch
shape and fails with:

```text
Triton Error [CUDA]: invalid argument
```

This is not a global MegaBlocks token limit. It is a limit of this standard
`moe` binned copy launch shape when `expert_capacity > 65535`. Fixing it would
mean changing the kernel strategy, for example by splitting the expert-capacity
dimension across launches or using a different dispatch path. Grouped dMoE
avoids this exact binned-grid shape.

### Grouped `dmoe` Is Dropless Grouped Dispatch Plus Grouped GEMM

`dMoE` inherits `MoE` but swaps the expert module:

```text
third_party/megablocks/megablocks/layers/dmoe.py:18
third_party/megablocks/megablocks/layers/dmoe.py:323
```

The replacement is `ParallelDroplessMLP`. Its constructor chooses the expert MLP
from the dMoE registry:

```text
self.mlp = dmlp_registry.get(args)
```

Source:

```text
dmoe.py:20
dmlp_registry.py:11
```

For our runs, `mlp_impl="grouped"`, so `forward_once` chooses the grouped path:

```text
ParallelDroplessMLP.forward_once: dmoe.py:282
grouped_forward_once:             dmoe.py:239
grouped_permute_and_compute:      dmoe.py:260
```

The grouped path still sorts and histograms the `N*K` assignments, but it does
not create `(E, max(tokens_per_expert), D)`. Instead:

```text
x_grouped = gather(x_flat, indices, bin_ids, bins, K)
x_grouped shape = (N*K, D)
expert_out = grouped MLP/GLU(x_grouped, tokens_per_expert)
out = scatter(expert_out, indices, bin_ids, gates, bins, K)
```

Source:

```text
ops.gather:  dmoe.py:274
self.mlp:    dmoe.py:277
ops.scatter: dmoe.py:280
```

`tokens_per_expert` becomes the grouped GEMM `batch_sizes`, so each expert GEMM
uses its actual routed row count rather than the busiest expert's padded count:

```text
GroupedMLP.forward: mlp.py:499
GroupedGLU.forward: glu.py:173
```

This is why grouped dMoE is the better main path for high-token-count dropless
MoE experiments in this checkout.

### MLP, GLU, And Shared Expert Support

The dMoE registry supports both FFN-style MLP and GLU experts:

```text
mlp/grouped -> GroupedMLP
mlp/sparse  -> SparseMLP
glu/grouped -> GroupedGLU
glu/sparse  -> SparseGLU
```

Source:

```text
third_party/megablocks/megablocks/layers/dmlp_registry.py:11
```

Grouped MLP math:

```text
y = activation(grouped_gemm(x, w1)) 
out = grouped_gemm(y, w2)
```

Source:

```text
third_party/megablocks/megablocks/layers/mlp.py:499
```

"Grouped MLP" means the expert MLP is executed with grouped GEMM. After routing,
MegaBlocks has one concatenated tensor of routed rows:

```text
x_grouped: (N*K, D)
tokens_per_expert: length E
```

`tokens_per_expert` tells grouped GEMM how many rows belong to each expert. The
kernel then runs the expert-specific GEMMs as a group of independent matrix
multiplies with different row counts. This is different from standard `moe`,
which pads every expert to the busiest expert's capacity and runs dense
`torch.bmm` over a rectangular `(E, expert_capacity, D)` tensor.

Grouped GLU math:

```text
gate = grouped_gemm(x, w1)
up = grouped_gemm(x, v1)
out = grouped_gemm(activation(gate) * up, w2)
```

Source:

```text
third_party/megablocks/megablocks/layers/glu.py:173
```

Standard `moe` does not choose `GroupedGLU` from this registry. In stock source,
`ParallelMLP.__init__` always sets:

```text
self.mlp = mlp.MLP(args)
```

Source:

```text
third_party/megablocks/megablocks/layers/moe.py:112
```

That is why our OLMoE-shaped standard-`moe` benchmark uses a local
`SyntheticGLUBatchedMLP` wrapper:

```text
src/profiling/profile_moe_layer.py:550
src/profiling/profile_moe_layer.py:669
```

This wrapper is not claiming stock standard `moe` has native GLU selection. It
lets us compare the same synthetic GLU expert math on the standard padded path
versus the grouped dMoE path.

Shared experts are selected through a separate registry:

```text
third_party/megablocks/megablocks/layers/sharedexpert_registry.py:9
```

`MoE.forward` computes the routed expert output, computes the shared expert
output, then combines them:

```text
out = routed_out + shared_expert_out
```

Source:

```text
third_party/megablocks/megablocks/layers/moe.py:469
third_party/megablocks/megablocks/layers/mlp.py:554
third_party/megablocks/megablocks/layers/glu.py:208
```

This is relevant for DeepSeek-shaped simulation because the catalog shape has a
shared expert component in addition to routed experts.

### Activation And GELU Implementation

MegaBlocks does not hard-code GELU inside the router or dispatch kernels. The
expert module receives an `activation_fn` callable through `Arguments`:

```text
third_party/megablocks/megablocks/layers/arguments.py:19
third_party/megablocks/megablocks/layers/arguments.py:30
```

The default is:

```text
activation_fn = partial(torch.nn.functional.gelu, approximate="tanh")
```

That is the tanh-approximate GELU:

```text
gelu_tanh(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

For the dense standard `moe` expert MLP, MegaBlocks applies this callable after
the first `torch.bmm`:

```text
third_party/megablocks/megablocks/layers/mlp.py:162
third_party/megablocks/megablocks/layers/mlp.py:165
third_party/megablocks/megablocks/layers/mlp.py:166
```

For grouped dMoE MLP, it applies the same callable after the first grouped GEMM:

```text
third_party/megablocks/megablocks/layers/mlp.py:519
third_party/megablocks/megablocks/layers/mlp.py:521
third_party/megablocks/megablocks/layers/mlp.py:522
```

For grouped dMoE GLU, it applies the callable to the gate projection, multiplies
by the up projection, then runs the down projection:

```text
third_party/megablocks/megablocks/layers/glu.py:200
third_party/megablocks/megablocks/layers/glu.py:202
third_party/megablocks/megablocks/layers/glu.py:203
third_party/megablocks/megablocks/layers/glu.py:204
third_party/megablocks/megablocks/layers/glu.py:205
```

The sparse STK path has a wrapper named `megablocks.layers.gelu.gelu`, but that
wrapper still calls PyTorch GELU on the sparse matrix data:

```text
third_party/megablocks/megablocks/layers/gelu.py:32
third_party/megablocks/megablocks/layers/gelu.py:36
```

Its custom backward uses the derivative of the same tanh approximation:

```text
third_party/megablocks/megablocks/layers/gelu.py:10
third_party/megablocks/megablocks/layers/gelu.py:11
third_party/megablocks/megablocks/layers/gelu.py:12
```

Our profiler maps catalog activations in the same way:

```text
src/profiling/profile_moe_layer.py:109
src/profiling/profile_moe_layer.py:110
src/profiling/profile_moe_layer.py:111
src/profiling/profile_moe_layer.py:112
src/profiling/profile_moe_layer.py:113
```

So for Nano-shaped runs with `activation="gelu_tanh"`, the activation is
PyTorch `F.gelu(..., approximate="tanh")`. For OLMoE-shaped GLU runs with
`activation="silu"`, it is PyTorch `F.silu`.

### Kernel And Library Responsibilities

Routing math:

```text
PyTorch Linear
PyTorch softmax
PyTorch topk
```

Metadata:

```text
ops.sort -> megablocks_ops.sort
ops.histogram -> megablocks_ops.histogram
ops.inclusive_cumsum -> megablocks_ops.inclusive_cumsum
```

Source:

```text
third_party/megablocks/megablocks/ops/sort.py:24
third_party/megablocks/megablocks/ops/histogram.py:56
third_party/megablocks/megablocks/ops/cumsum.py:83
```

Token movement:

```text
standard moe binned_gather/binned_scatter -> Triton _binned_copy
dMoE gather/scatter -> Triton _padded_copy with no padding
```

Source:

```text
third_party/megablocks/megablocks/backend/kernels.py:45
third_party/megablocks/megablocks/backend/kernels.py:141
third_party/megablocks/megablocks/backend/kernels.py:206
third_party/megablocks/megablocks/backend/kernels.py:392
third_party/megablocks/megablocks/backend/kernels.py:421
```

Grouped expert compute:

```text
GroupedMLP/GroupedGLU -> grouped_gemm.ops.gmm -> grouped_gemm.backend.gmm
```

Source:

```text
third_party/grouped_gemm/grouped_gemm/ops.py:34
third_party/grouped_gemm/grouped_gemm/backend.py:24
third_party/grouped_gemm/csrc/grouped_gemm.cu:470
```

The grouped GEMM extension is BF16-only in the local C++ validation:

```text
TORCH_CHECK(a.scalar_type() == torch::kBFloat16)
TORCH_CHECK(b.scalar_type() == torch::kBFloat16)
TORCH_CHECK(c.scalar_type() == torch::kBFloat16)
```

Source:

```text
third_party/grouped_gemm/csrc/grouped_gemm.cu:492
third_party/grouped_gemm/csrc/grouped_gemm.cu:505
```

This is why our dMoE profiler path requires `dtype=bfloat16`.

### Timing Scope In Our Current Profiler

The measurement function uses CUDA events after warmup:

```text
src/profiling/profile_moe_layer.py:338
```

When `timing_scope=megablocks_core`, routing is prepared once outside the timed
loop:

```text
src/profiling/profile_moe_layer.py:1252
```

The timed function then runs:

```text
layer.experts(routing.x_mb, routing.router_probs, routing.gates, routing.indices)
shared_expert if configured
```

Source:

```text
src/profiling/profile_moe_layer.py:761
```

So `megablocks_core` measures dispatch metadata, gather, expert compute, scatter,
and shared expert if present. It excludes adapter layout conversion, router
linear, softmax, top-k, and aux-loss setup. `adapter_boundary` includes the full
adapter forward from `(B,T,D)` input to `(B,T,D)` output.

## Expert Math

Two-matmul MLP experts compute:

```text
expert_i(x) = activation(x @ W1_i) @ W2_i
W1_i: (D, H)
W2_i: (H, D)
```

GLU/SwiGLU experts compute:

```text
expert_i(x) = (activation(x @ W_gate_i) * (x @ W_up_i)) @ W_down_i
W_gate_i: (D, H)
W_up_i:   (D, H)
W_down_i: (H, D)
```

MegaBlocks has stock GLU implementations for dMoE/grouped execution. In this
checkout, standard `moe` constructs `ParallelMLP`, whose stock expert module is
the two-matmul `mlp.MLP`. For the current OLMoE-shaped benchmark, our adapter
replaces that expert module with a small batched GLU wrapper after constructing
standard `moe`; this keeps the standard `moe` line semantically aligned with
the synthetic GLU reference and grouped dMoE GLU path.

Sources:

```text
third_party/megablocks/megablocks/layers/mlp.py
third_party/megablocks/megablocks/layers/glu.py
src/profiling/profile_moe_layer.py
```

## Fixed Costs And Scaling Costs

Fixed or weakly scaling costs:

```text
router/top-k setup when timing adapter_boundary
sort kernel launch/setup
histogram kernel launch/setup
cumsum kernel launch/setup
gather/scatter kernel launches
buffer allocation
standard moe capacity decision: max(tokens_per_expert).item()
```

Scaling costs:

```text
router matmul scales with N * D * E
sort/histogram/gather/scatter scale mostly with N * K and D
expert MLP/GLU scales with active routed rows, D, H, and expert type
standard moe padded expert work scales with E * max(tokens_per_expert)
grouped dMoE expert work scales closer to N * K
shared expert work scales with N * D * H_shared
```

This explains the Nano result shape: Nano has small `D`, small `H`, small `E`,
and only `E/K=2`, so there is limited useful expert math to amortize
MegaBlocks' routing and dispatch machinery. The same fixed costs matter less for
OLMoE-shaped runs because GLU expert compute with `D=2048`, `H=1024`, `E=64`,
and `K=8` is much larger.

## Current MegaBlocks Findings

Standard `moe`:

```text
Sparse activation, padded expert batches.
Useful for comparing padded sparse dispatch against dense references.
In this checkout, stock standard expert MLP is two-matmul MLP.
Stock expert MLP is bias-free; trained Nano exactness needs a local bias adapter.
Our OLMoE-shaped standard-moe line uses a local GLU wrapper for GLU semantics.
Observed high-N failure when expert_capacity exceeded 65535 on binned_gather.
```

Grouped `dmoe`:

```text
Sparse activation, no padding to busiest expert in the grouped path.
Uses grouped_gemm/CUTLASS with tokens_per_expert as batch sizes.
Avoids the standard-moe binned grid limit observed at Nano N=131072.
Requires BF16 in this local grouped_gemm extension.
Trained Nano exactness with nonzero expert biases needs per-routed-row bias adds,
which are extra adapter work outside the stock grouped GEMM path.
```

What not to claim:

```text
Do not claim exact OLMoE or DeepSeek execution from synthetic shape benchmarks.
Exact model claims require model-specific router semantics and checkpoint weights.
Do not claim MegaBlocks is bad for small models based only on Nano small-N timing.
The fair claim is that fixed sparse-dispatch cost dominates tiny Nano shapes.
```

## Sources

Local MegaBlocks source:

```text
third_party/megablocks/megablocks/layers/arguments.py
third_party/megablocks/megablocks/layers/router.py
third_party/megablocks/megablocks/layers/moe.py
third_party/megablocks/megablocks/layers/dmoe.py
third_party/megablocks/megablocks/layers/mlp.py
third_party/megablocks/megablocks/layers/glu.py
third_party/megablocks/megablocks/layers/activation_fn.py
third_party/megablocks/megablocks/layers/gelu.py
third_party/megablocks/megablocks/layers/sharedexpert_registry.py
third_party/megablocks/megablocks/backend/kernels.py
third_party/megablocks/megablocks/ops/
```

Local grouped GEMM source:

```text
third_party/grouped_gemm/grouped_gemm/backend.py
third_party/grouped_gemm/grouped_gemm/ops.py
third_party/grouped_gemm/csrc/grouped_gemm.cu
```

Local adapter/profiler source:

```text
src/profiling/profile_moe_layer.py
src/profiling/run_model_token_capacity.py
configs/moe_model_shapes.json
```
