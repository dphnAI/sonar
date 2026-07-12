#!/bin/bash

CUDA_VERSION=${CUDA_VERSION:-13.0.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
MAX_JOBS=${MAX_JOBS:-}
NVCC_THREADS=${NVCC_THREADS:-}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0 7.5 8.0 8.9 9.0 10.0 11.0 12.0"}

DOCKER_BUILDKIT=1 docker build . -f docker/Dockerfile --target build --tag alpindale/aphrodite-build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
    ${MAX_JOBS:+--build-arg max_jobs=$MAX_JOBS} \
    ${NVCC_THREADS:+--build-arg nvcc_threads=$NVCC_THREADS}

docker run -d --name aphrodite-build-container alpindale/aphrodite-build tail -f /dev/null
# copies to dist/ within working directory
docker cp aphrodite-build-container:/workspace/dist .
docker stop aphrodite-build-container && docker rm aphrodite-build-container
