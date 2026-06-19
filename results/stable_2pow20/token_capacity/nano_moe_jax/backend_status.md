# Backend Status

Requested backend families:

- `reference`
- `megablocks_moe`
- `megablocks_dmoe`

Successful plotted backend variants:

- `megablocks_dmoe`: max successful `N=524288`
- `megablocks_moe`: max successful `N=65536`
- `reference_dense_ffn`: max successful `N=1048576`

Failures / unsupported rows:

- N=131072 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=524288 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=1048576 backend=megablocks_dmoe: returncode=1 reason=error: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 9.64 GiB of which 1.80 GiB is free. Process 1855005 has 10.85 MiB memory in use. Including non-PyTorch memory, this process has 7.62 GiB memory in use. Of the allocated memory 5.34 GiB is allocated by PyTorch, and 2.02 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
