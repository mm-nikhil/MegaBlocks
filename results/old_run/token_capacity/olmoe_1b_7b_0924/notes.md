# Model Token-Capacity Sweep

model_shape_name: `olmoe_1b_7b_0924`
simulation_level: `level1_shape_expert`

## Simulation Caveat

Caveat: synthetic shape/expert simulation; not exact checkpoint/router semantics.

This means the run uses the catalog's MoE layer geometry and expert type
such as `D`, `H`, `E`, `K`, GLU/SwiGLU, activation, and dtype, with
synthetic weights. It does not load the real model checkpoint and does
not implement the model-specific router exactly.

For OLMoE-shaped runs, `reference_dense_glu` is a PyTorch CUDA dense
all-expert GLU baseline for this synthetic shape. It is useful as a
dense-vs-sparse comparison point, but it is not exact OLMoE execution.

Max successful `N` by backend:

- `megablocks_dmoe`: `49152`
- `megablocks_moe`: `49152`
- `reference_dense_glu`: `4096`

This sweep varies `N = B*T`, the number of input-token hidden rows at one MoE layer.
It is not generated output tokens per second.

The first-cut dashboard shows:

- `mean_forward_ms`: average timed forward call for the selected timing scope.
- `ms_per_input_token`: `mean_forward_ms / N`.
- `active_expert_tflops_per_second`: useful active expert math normalized by runtime.
- `padding_factor`: backend expert rows divided by routed token-expert pairs.

Backend success, failure, and unsupported status is recorded in `backend_status.md`.

Shape:

```json
{
  "activation": "silu",
  "dtype": "bfloat16",
  "expert_intermediate_size": 1024,
  "expert_type": "glu",
  "hidden_size": 2048,
  "max_position_embeddings": 4096,
  "norm_topk_prob": false,
  "notes": "MegaBlocks can approximate this MoE-layer geometry with mlp_type=glu, E=64, K=8, D=2048, H=1024. Exact semantic matching needs an OLMoE-compatible PyTorch reference and router behavior.",
  "num_experts_per_token": 8,
  "num_hidden_layers": 16,
  "num_routed_experts": 64,
  "num_shared_experts": 0,
  "router_aux_loss_coef": 0.01,
  "router_score_function": "softmax_topk",
  "shared_expert_intermediate_size": 0,
  "simulation_level": "level1_shape_expert",
  "source": "https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json",
  "source_verified_on": "2026-06-17"
}
```

Failures:

- N=8192 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10558616371 base_estimated=7821197312 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=16384 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=19304054784 base_estimated=14299299840 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=32768 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=36794931609 base_estimated=27255504896 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=49152 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=54285808435 base_estimated=40211709952 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=reference: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=71776685260 base_estimated=53167915008 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=megablocks_moe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10895523840 base_estimated=8070758400 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=65536 backend=megablocks_dmoe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=10895523840 base_estimated=8070758400 allowed=8858507673 free=9842786304 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
