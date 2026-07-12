#!/bin/bash
set -e

# Script to export the main wheel.
#
# Usage:
#   ./docker/export_wheels.sh                   # Export the main wheel
#   CUDA_VERSION=12.8.1 ./docker/export_wheels.sh
#
# Environment variables:
#   CUDA_VERSION      - CUDA version (default: 13.0.2)
#   TARGETPLATFORM    - Target platform (default: linux/amd64)
#   TORCH_CUDA_ARCH_LIST - CUDA arch list to compile into the wheel
#   MAX_JOBS           - Number of parallel jobs for Ninja (default: 2)
#   NVCC_THREADS       - Number of threads for nvcc (default: 8)
#   PYTHON_VERSION     - Python version in the build image (default: 3.12)
#   APHRODITE_VERSION_OVERRIDE - Wheel version override. If unset and the
#                                checkout is exactly on a tag, inferred from it.

CUDA_VERSION="${CUDA_VERSION:-13.0.2}"
TARGETPLATFORM="${TARGETPLATFORM:-linux/amd64}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0 7.5 8.0 8.9 9.0 10.0 11.0 12.0}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-8}"
APHRODITE_VERSION_OVERRIDE="${APHRODITE_VERSION_OVERRIDE:-}"

if [ -z "${APHRODITE_VERSION_OVERRIDE}" ]; then
    if tag="$(git describe --tags --exact-match 2>/dev/null)"; then
        APHRODITE_VERSION_OVERRIDE="${tag#v}"
    fi
fi

echo "Exporting main wheel..."
if [ -n "${APHRODITE_VERSION_OVERRIDE}" ]; then
    echo "Using Aphrodite version override: ${APHRODITE_VERSION_OVERRIDE}"
fi
mkdir -p ./wheels/main
rm -f ./wheels/main/*.whl
DOCKER_BUILDKIT=1 docker build \
    --target main-wheel-export \
    --output ./wheels/main \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg TARGETPLATFORM="${TARGETPLATFORM}" \
    --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
    --build-arg max_jobs="${MAX_JOBS}" \
    --build-arg nvcc_threads="${NVCC_THREADS}" \
    --build-arg APHRODITE_VERSION_OVERRIDE="${APHRODITE_VERSION_OVERRIDE}" \
    -f docker/Dockerfile .
echo "✓ Main wheel exported to ./wheels/main"

echo "Done!"
