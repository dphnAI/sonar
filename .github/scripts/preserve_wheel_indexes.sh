#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

public_dir="${1:?usage: preserve_wheel_indexes.sh DOCS_PUBLIC_DIR}"
site="${SONAR_SITE_URL:-https://sonar.dphn.ai}"
wheel_storage="${SONAR_WHEEL_STORAGE_URL:-https://sonar-nightly.dphn.ai/platform-wheels}"

preserve() {
  local relative_path=$1
  local fallback_url=${2:-}
  local output="${public_dir}/${relative_path}/index.html"
  mkdir -p "$(dirname "$output")"
  if curl --fail --location --silent --show-error \
    "${site%/}/${relative_path}/" \
    --output "$output"; then
    return
  fi
  if [[ -n "$fallback_url" ]] && curl \
    --fail --location --silent --show-error \
    "$fallback_url" \
    --output "$output"; then
    return
  fi
  rm -f "$output"
}

preserve "nightly/sonar"
preserve "whl"
preserve "simple/sonar"

for channel in release nightly; do
  for platform in \
    cuda/x86_64 \
    cuda/aarch64 \
    cpu/x86_64 \
    cpu/aarch64 \
    rocm/x86_64 \
    xpu/x86_64 \
    metal/aarch64; do
    fallback="${wheel_storage%/}/${channel}/${platform}/index.html"
    preserve "whl/${channel}/${platform}" "$fallback"
    preserve \
      "whl/${channel}/${platform}/simple/sonar" \
      "$fallback"
    if [[ "$channel" == "nightly" ]]; then
      preserve "nightly/${platform}"
    fi
  done
done
