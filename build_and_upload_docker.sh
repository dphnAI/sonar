#!/bin/bash

set -e

# NOTE(alpin): These are the default values for my own machine.
MAX_JOBS=64
NVCC_THREADS=64
CUDA_VERSION=13.0.0
PYTHON_VERSION=3.12
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.9 9.0 10.0 11.0 12.0"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_jobs) MAX_JOBS="$2"; shift ;;
        --nvcc_threads) NVCC_THREADS="$2"; shift ;;
        --cuda_version) CUDA_VERSION="$2"; shift ;;
        --python_version) PYTHON_VERSION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Step 1: Build the wheel
DOCKER_BUILDKIT=1 docker build . -f docker/Dockerfile --target build --tag alpindale/aphrodite-build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
    --build-arg max_jobs=$MAX_JOBS \
    --build-arg nvcc_threads=$NVCC_THREADS

# Create a temporary container to extract the wheel
docker run -d --name aphrodite-build-container alpindale/aphrodite-build tail -f /dev/null
# Create dist directory if it doesn't exist
mkdir -p dist
# Copy the wheel to the dist directory
docker cp aphrodite-build-container:/workspace/dist .
docker stop aphrodite-build-container && docker rm aphrodite-build-container

# Step 2: Build the final Docker image using the wheel
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile . --target aphrodite-openai --tag alpindale/aphrodite-openai \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
    --build-arg max_jobs=$MAX_JOBS \
    --build-arg nvcc_threads=$NVCC_THREADS

# Step 3: Tag and push the Docker image
commit=$(git rev-parse --short HEAD)
docker tag alpindale/aphrodite-openai alpindale/aphrodite-openai:${commit}
docker push alpindale/aphrodite-openai:${commit}
docker tag alpindale/aphrodite-openai alpindale/aphrodite-openai:latest
docker push alpindale/aphrodite-openai:latest

echo "Build and upload completed successfully!"
