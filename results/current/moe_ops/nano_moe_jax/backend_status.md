# Backend Status

Requested backend families:

- `megablocks_moe`
- `megablocks_dmoe`

Successful plotted backend variants:

- `megablocks_dmoe [bfloat16] (BF16-only dMoE)`: max successful `N=524288`
- `megablocks_moe [float32]`: max successful `N=65536`

Failures / unsupported rows:

- N=131072 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: returncode=1 reason=error: Memory preflight rejected this run before allocation. estimated=9470216908 base_estimated=7014975488 allowed=8903983104 free=9893314560 total=10351935488 fraction=0.9. Use a smaller N or --skip-memory-preflight if you intentionally want to test the limit.
- N=1048576 backend=megablocks_dmoe: returncode=1 reason=error: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 9.64 GiB of which 1.80 GiB is free. Process 1855005 has 10.85 MiB memory in use. Including non-PyTorch memory, this process has 7.62 GiB memory in use. Of the allocated memory 5.62 GiB is allocated by PyTorch, and 1.74 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
