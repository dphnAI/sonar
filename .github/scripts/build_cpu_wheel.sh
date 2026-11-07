#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:?usage: build_cpu_wheel.sh OUTPUT_DIR}"
python_version="${PYTHON_VERSION:-3.13}"
target_platform="${TARGET_PLATFORM:-linux/$(uname -m)}"
version="${SONAR_VERSION_OVERRIDE:-}"
cache_scope="${CACHE_SCOPE:-}"
max_jobs="${MAX_JOBS:-4}"
dockerfile="${CPU_BUILD_DOCKERFILE:-docker/Dockerfile.cpu}"

mkdir -p "$output_dir"
rm -f "$output_dir"/*.whl

args=(
  --file "$dockerfile"
  --target sonar-wheel-export
  --output "type=local,dest=${output_dir}"
  --build-arg "PYTHON_VERSION=${python_version}"
  --build-arg "SONAR_VERSION_OVERRIDE=${version}"
  --build-arg "MAX_JOBS=${max_jobs}"
)

if [[ -n "$target_platform" ]]; then
  args+=(--platform "$target_platform")
fi
if [[ -n "$cache_scope" ]]; then
  args+=(
    --cache-from "type=gha,scope=${cache_scope}"
    --cache-to "type=gha,mode=max,scope=${cache_scope}"
  )
fi

docker buildx build "${args[@]}" .
test -n "$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
