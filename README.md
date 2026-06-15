# MegaBlocks NanoMoE Profiling

This repo profiles MegaBlocks MoE execution against the MoE layer semantics used by
Nano-MoE-JAX.

The goal is not to rewrite Nano-MoE-JAX casually. The first-class check is semantic:
the PyTorch reference in `src/profiling` must match Nano-MoE-JAX at the MoE boundary
before MegaBlocks timings are interpreted.

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

The PyTorch NanoMoE reference matches Nano-MoE-JAX for output, router gates, expert
indices, and auxiliary loss. `grouped_gemm` and `megablocks_ops` build successfully
using the local CUDA 12.6 overlay on the RTX 3080 host.

Current validated MegaBlocks correctness smoke:

- `megablocks/moe`, float32, zero expert biases: tight agreement with reference.

Still under investigation:

- `megablocks/dmoe` grouped BF16 mapping.
- FP16 max-error outliers for larger standard MoE runs.

Start with:

```bash
source .venv/bin/activate
python src/profiling/check_nano_moe_port.py
```

See:

- `docs/setup.md`
- `docs/usage.md`
- `docs/architecture.md`
- `docs/terminology.md`
- `docs/reporting.md`
