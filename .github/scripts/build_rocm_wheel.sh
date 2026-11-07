#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:?usage: build_rocm_wheel.sh OUTPUT_DIR}"
architectures="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
version="${SONAR_VERSION_OVERRIDE:-}"
max_jobs="${MAX_JOBS:-12}"
dockerfile="${ROCM_BUILD_DOCKERFILE:-docker/Dockerfile.rocm}"

mkdir -p "$output_dir"
rm -f "$output_dir"/*.whl

docker buildx build \
  --file "$dockerfile" \
  --target export_sonar \
  --output "type=local,dest=${output_dir}" \
  --build-arg "ARG_PYTORCH_ROCM_ARCH=${architectures}" \
  --build-arg "SONAR_VERSION_OVERRIDE=${version}" \
  --build-arg "MAX_JOBS=${max_jobs}" \
  .

test -n "$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
