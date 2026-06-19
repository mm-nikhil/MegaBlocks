"""Weight and input construction for Nano-compatible MoE profiling."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from nano_moe_torch import NanoMoEWeights, from_flax_moe_params


def make_synthetic_weights(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> NanoMoEWeights:
    """Create deterministic synthetic FFN MoE weights for shape/performance runs."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(*shape: int) -> torch.Tensor:
        return (0.02 * torch.randn(*shape, generator=generator)).to(device=device, dtype=dtype)

    b1 = torch.zeros(args.n_experts, args.d_ff, device=device, dtype=dtype)
    b2 = torch.zeros(args.n_experts, args.d_model, device=device, dtype=dtype)
    if not args.zero_expert_biases:
        b1 = randn(args.n_experts, args.d_ff)
        b2 = randn(args.n_experts, args.d_model)

    return NanoMoEWeights(
        router_kernel=randn(args.d_model, args.n_experts),
        w1=randn(args.n_experts, args.d_model, args.d_ff),
        b1=b1,
        w2=randn(args.n_experts, args.d_ff, args.d_model),
        b2=b2,
    )


def make_synthetic_glu_up_weight(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create the second GLU/SwiGLU up-projection used by synthetic GLU shapes."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 2)
    v1 = 0.02 * torch.randn(args.n_experts, args.d_model, args.d_ff, generator=generator)
    return v1.to(device=device, dtype=dtype).contiguous()


def zero_expert_biases(weights: NanoMoEWeights) -> NanoMoEWeights:
    """Return a copy-like weight bundle with expert Dense biases zeroed."""

    return NanoMoEWeights(
        router_kernel=weights.router_kernel,
        w1=weights.w1,
        b1=torch.zeros_like(weights.b1),
        w2=weights.w2,
        b2=torch.zeros_like(weights.b2),
    )


def make_nano_jax_initialized_weights(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> NanoMoEWeights:
    """Initialize a real Nano-MoE-JAX MoELayer and convert its params to torch."""

    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise RuntimeError(f"Nano-MoE-JAX checkout not found: {nano_dir}")

    # JAX is used only to reproduce Nano-MoE-JAX initializers. Keep that tiny
    # setup on CPU so profiling runs do not allocate JAX GPU memory.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(nano_dir))
    try:
        import jax
        import jax.numpy as jnp
        from nano_moe.config import NanoMoEConfig
        from nano_moe.layers import MoELayer
    except ImportError as exc:
        raise RuntimeError(
            "weight_source=nano_jax_init requires JAX, Flax, and the Nano-MoE-JAX checkout. "
            "Use --weight-source synthetic for a dependency-light synthetic benchmark."
        ) from exc

    config = NanoMoEConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        top_k=args.top_k,
        dropout_rate=0.0,
    )
    rng = jax.random.PRNGKey(args.seed)
    init_x = jnp.zeros((args.batch_size, args.seq_len, args.d_model), dtype=jnp.float32)
    params = MoELayer(config=config).init(rng, init_x, deterministic=True)["params"]
    return from_flax_moe_params(params, device=device, dtype=dtype)


def make_trained_nano_checkpoint_weights(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> NanoMoEWeights:
    """Load one trained Nano checkpoint MoE layer and convert it to torch."""

    checkpoint_dir = args.checkpoint_dir.resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    params_path = checkpoint_dir / "params.msgpack"
    if not metadata_path.exists() or not params_path.exists():
        raise RuntimeError(
            f"Trained Nano checkpoint not found in {checkpoint_dir}. "
            "Expected metadata.json and params.msgpack.",
        )

    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise RuntimeError(f"Nano-MoE-JAX checkout not found: {nano_dir}")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(nano_dir))
    try:
        import jax
        import jax.numpy as jnp
        from flax import serialization
        from nano_moe.config import NanoMoEConfig
        from nano_moe.model import NanoMoE
    except ImportError as exc:
        raise RuntimeError(
            "weight_source=trained_nano_checkpoint requires JAX, Flax, and the "
            "Nano-MoE-JAX checkout."
        ) from exc

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_data = metadata["config"]
    fields = NanoMoEConfig.__dataclass_fields__
    config = NanoMoEConfig(**{key: config_data[key] for key in fields if key in config_data})

    if args.checkpoint_block_index < 0 or args.checkpoint_block_index >= config.n_layers:
        raise RuntimeError(
            f"--checkpoint-block-index must be in [0, {config.n_layers}) for this checkpoint.",
        )

    mismatches = []
    expected = {
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "n_experts": config.n_experts,
        "top_k": config.top_k,
    }
    for key, expected_value in expected.items():
        actual_value = getattr(args, key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: arg={actual_value} checkpoint={expected_value}")
    if mismatches:
        raise RuntimeError(
            "Profiler arguments do not match the trained Nano checkpoint MoE shape: "
            + "; ".join(mismatches),
        )

    dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)
    template = NanoMoE(config=config).init(
        {"params": rng, "dropout": rng},
        dummy,
        deterministic=True,
    )["params"]
    params = serialization.from_bytes(template, params_path.read_bytes())
    moe_params = params[f"block_{args.checkpoint_block_index}"]["MoELayer_0"]
    return from_flax_moe_params(moe_params, device=device, dtype=dtype)


def make_weights(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> NanoMoEWeights:
    """Select the requested weight source and apply bias policy."""

    if args.weight_source == "nano_jax_init":
        weights = make_nano_jax_initialized_weights(args, dtype, device)
    elif args.weight_source == "trained_nano_checkpoint":
        weights = make_trained_nano_checkpoint_weights(args, dtype, device)
    else:
        weights = make_synthetic_weights(args, dtype, device)

    if args.zero_expert_biases:
        weights = zero_expert_biases(weights)
    return weights


def make_input(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create deterministic Nano-layout input with shape ``(B, T, D)``."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, generator=generator)
    return x.to(device=device, dtype=dtype)

