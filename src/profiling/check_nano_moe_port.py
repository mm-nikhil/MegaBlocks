"""Check that the PyTorch NanoMoE MoE port matches Nano-MoE-JAX.

Example:
    python src/profiling/check_nano_moe_port.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from nano_moe_torch import from_flax_moe_params, nano_moe_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano-jax-dir", type=Path, default=Path("third_party/Nano-MoE-JAX"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    return parser.parse_args()


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> None:
    args = parse_args()
    nano_dir = args.nano_jax_dir.resolve()
    if not nano_dir.exists():
        raise SystemExit(f"Nano-MoE-JAX checkout not found: {nano_dir}")
    sys.path.insert(0, str(nano_dir))

    from nano_moe.config import NanoMoEConfig
    from nano_moe.layers import MoELayer, Router

    config = NanoMoEConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        top_k=args.top_k,
        dropout_rate=0.0,
    )
    rng = jax.random.PRNGKey(args.seed)
    x = jax.random.normal(rng, (args.batch_size, args.seq_len, args.d_model))

    layer = MoELayer(config=config)
    params = layer.init(rng, x, deterministic=True)["params"]
    jax_out, jax_aux = layer.apply({"params": params}, x, deterministic=True)

    router = Router(n_experts=args.n_experts, top_k=args.top_k)
    router_params = {"Router_0": params["Router_0"]}
    jax_gates, jax_indices, _ = router.apply(
        {"params": router_params["Router_0"]},
        x,
    )

    torch_weights = from_flax_moe_params(params)
    torch_x = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
    torch_out = nano_moe_forward(
        torch_x,
        torch_weights,
        top_k=args.top_k,
        deterministic=True,
        dropout_p=0.0,
    )

    out_np = torch_out.output.detach().cpu().numpy()
    aux_np = np.asarray(torch_out.aux_loss.detach().cpu())
    gates_np = torch_out.gates.detach().cpu().numpy()
    indices_np = torch_out.indices.detach().cpu().numpy()

    checks = {
        "output": np.allclose(out_np, np.asarray(jax_out), atol=args.atol, rtol=args.rtol),
        "aux_loss": np.allclose(aux_np, np.asarray(jax_aux), atol=args.atol, rtol=args.rtol),
        "gates": np.allclose(gates_np, np.asarray(jax_gates), atol=args.atol, rtol=args.rtol),
        "indices": np.array_equal(indices_np, np.asarray(jax_indices)),
    }

    print("NanoMoE MoE JAX vs PyTorch reference")
    print(f"shape: batch={args.batch_size} seq={args.seq_len} d_model={args.d_model}")
    print(f"experts={args.n_experts} top_k={args.top_k} d_ff={args.d_ff}")
    print(f"max_abs(output): {max_abs(out_np, np.asarray(jax_out)):.6g}")
    print(f"max_abs(gates):  {max_abs(gates_np, np.asarray(jax_gates)):.6g}")
    print(f"abs(aux_loss):   {abs(float(aux_np) - float(jax_aux)):.6g}")
    print(f"indices_equal:   {checks['indices']}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"FAILED checks: {', '.join(failed)}")

    print("all checks passed")


if __name__ == "__main__":
    main()
