#!/usr/bin/env bash
set -euo pipefail

CUDA_VERSION_DIR="cuda-12.6"
CUDA_ROOT=".cuda/extract/usr/local/${CUDA_VERSION_DIR}"

mkdir -p .cuda/debs .cuda/extract

(
  cd .cuda/debs
  apt download \
    cuda-nvcc-12-6 \
    cuda-cudart-dev-12-6 \
    cuda-cudart-12-6 \
    cuda-driver-dev-12-6 \
    cuda-cccl-12-6 \
    cuda-nvvm-12-6 \
    cuda-crt-12-6
)

for deb in .cuda/debs/*.deb; do
  dpkg-deb -x "$deb" .cuda/extract
done

if [[ ! -e "${CUDA_ROOT}/include" ]]; then
  ln -s targets/x86_64-linux/include "${CUDA_ROOT}/include"
fi

if [[ ! -e "${CUDA_ROOT}/lib64" ]]; then
  ln -s targets/x86_64-linux/lib "${CUDA_ROOT}/lib64"
fi

"${CUDA_ROOT}/bin/nvcc" --version
echo "CUDA overlay ready at ${PWD}/${CUDA_ROOT}"
