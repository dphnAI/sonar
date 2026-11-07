#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:?usage: build_metal_wheel.sh OUTPUT_DIR}"
: "${METAL_BUILD_PYTHON:?METAL_BUILD_PYTHON is required}"

mkdir -p "$output_dir"
rm -f "$output_dir"/*.whl

if [[ ! -x "$METAL_BUILD_PYTHON" ]]; then
  echo "::error::Metal build Python is not executable: ${METAL_BUILD_PYTHON}"
  exit 1
fi

SONAR_TARGET_DEVICE=metal \
SONAR_REQUIRE_RUST_FRONTEND=1 \
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}" \
  "$METAL_BUILD_PYTHON" setup.py bdist_wheel \
    --dist-dir "$output_dir" \
    --py-limited-api=cp312

wheel="$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
test -n "$wheel"

rm -rf .wheel-test
uv venv --python 3.13 .wheel-test
uv pip install --python .wheel-test/bin/python "$wheel"
.wheel-test/bin/python -c \
  "import sonar; from sonar.metal.metal import get_ops; get_ops()"
test -x .wheel-test/lib/python3.13/site-packages/sonar/sonar-rs
