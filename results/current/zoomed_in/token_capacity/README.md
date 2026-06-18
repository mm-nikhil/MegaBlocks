# Zoomed-In Token Capacity

This folder contains small-`N` token-capacity sweeps.

## Runs

- `nano_moe_jax/`: no-bias initialized Nano baseline. The sweep uses `N=128..16384`, `T=128`, BF16, `weight_source=nano_jax_init`, and skips output correctness checks.

The dashboard x-axis is intentionally capped at `2^14`.
