# Verification Summary

status: `passed`
model_shape_name: `nano_moe_jax`
weight_source: `trained_nano_checkpoint`
checkpoint_dir: `results/trained_nano_moe_checkpoint`
verification_tokens: `512`

These checks run at small `N` before the performance sweep. They verify
the MegaBlocks adapter against the PyTorch NanoJAX MoE reference using
the selected NanoJAX weights. Large performance rows are not dense-reference
checked row-by-row.

Cases:

- `nanojax_fp32_megablocks_moe`: `passed`
  backend: `megablocks_moe`
  dtype: `float32`
  threshold: `0.001`
  max_abs_vs_reference: `0.0`
  aux_loss_abs_diff: `0.0`
  router_expert_set_mismatch_count: `0`
  note: FP32 standard MoE adapter vs PyTorch NanoJAX reference.
- `nanojax_bf16_megablocks_dmoe`: `passed`
  backend: `megablocks_dmoe`
  dtype: `bfloat16`
  threshold: `0.02`
  max_abs_vs_reference: `0.0`
  aux_loss_abs_diff: `0.0`
  router_expert_set_mismatch_count: `0`
  note: BF16-only dMoE adapter vs PyTorch NanoJAX reference; local grouped_gemm does not support FP32.
