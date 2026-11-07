#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:?usage: build_cuda_arm64_wheel.sh OUTPUT_DIR}"
builder_name="${CUDA_ARM64_BUILDER_NAME:-sonar-cuda-arm64}"
dockerfile="${CUDA_ARM64_BUILD_DOCKERFILE:-docker/Dockerfile}"
version="${SONAR_VERSION_OVERRIDE:-}"
if [[ -z "$version" ]]; then
  version="$(git describe --tags --exact-match)"
  version="${version#v}"
fi

mkdir -p "$output_dir"
rm -f "$output_dir"/sonar-*.whl

docker buildx build \
  --builder "$builder_name" \
  --platform linux/arm64 \
  --file "$dockerfile" \
  --target main-wheel-export \
  --output "type=local,dest=${output_dir}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION:-13.0.3}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION:-3.12}" \
  --build-arg "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX}" \
  --build-arg "max_jobs=${MAX_JOBS:-64}" \
  --build-arg "nvcc_threads=${NVCC_THREADS:-4}" \
  --build-arg python_abi_tag=cp311 \
  --build-arg "SONAR_VERSION_OVERRIDE=${version}" \
  .

shopt -s nullglob
wheels=("$output_dir"/sonar-*-cp311-abi3-*aarch64.whl)
if (( ${#wheels[@]} != 1 )); then
  echo "Expected exactly one CPython 3.11+ ARM64 CUDA wheel in ${output_dir}." >&2
  exit 1
fi
TMPDIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}" \
  uv run --isolated --no-project --with auditwheel==6.6.0 \
  python .github/scripts/detect_manylinux_tag.py "${wheels[0]}"
