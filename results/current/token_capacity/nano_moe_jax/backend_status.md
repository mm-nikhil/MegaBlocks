# Backend Status

Requested backend families:

- `reference`
- `megablocks_moe`
- `megablocks_dmoe`

Successful plotted backend variants:

- `megablocks_dmoe`: max successful `N=262144`
- `megablocks_moe`: max successful `N=65536`
- `reference_dense_ffn`: max successful `N=262144`

Failures / unsupported rows:

- N=131072 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
- N=262144 backend=megablocks_moe: returncode=1 reason=error: Triton Error [CUDA]: invalid argument
