# MegaBlocks NanoMoE Profiling

This repo profiles MegaBlocks MoE execution against the MoE layer semantics used by
Nano-MoE-JAX.

Semantic checks come before timing interpretation. The PyTorch reference in
`src/profiling` matches Nano-MoE-JAX at the MoE boundary, then the MegaBlocks
adapter is checked against that reference.

## Layout

- `src/profiling/`: correctness and profiling scripts.
- `docs/`: setup, usage, terminology, and result reporting.
- `third_party/megablocks/`: MegaBlocks submodule.
- `third_party/Nano-MoE-JAX/`: Nano-MoE-JAX submodule.
- `third_party/grouped_gemm/`: grouped GEMM submodule for build/debug context.

## Quick Start

```bash
git submodule update --init --recursive
scripts/bootstrap_python_env.sh
```

Then follow `docs/setup.md`. On a GPU server, the non-negotiable dependency is
`nvcc`. It can come from a system CUDA toolkit or the workspace-local CUDA overlay
described in the setup docs.

## Current Status

The current docs are split by purpose:

- `docs/moe_megablocks_deep_dive.md`: MegaBlocks `moe`/`dmoe` implementation.
- `docs/verification.md`: NanoJAX, PyTorch reference, MegaBlocks, and trained-weight verification.
- `docs/metrics_and_results.md`: metrics, plots, result interpretation.
- `docs/next_steps.md`: completed work and next actions.

Start with:

```bash
source .venv/bin/activate
python src/profiling/check_nano_moe_port.py
python src/profiling/verify_moe_layer.py
```

See:

- `docs/moe_megablocks_deep_dive.md`
- `docs/metrics_and_results.md`
- `docs/next_steps.md`
- `configs/moe_model_shapes.json`
- `docs/setup.md`
- `docs/usage.md`
- `docs/architecture.md`
- `docs/terminology.md`
- `docs/reporting.md`
