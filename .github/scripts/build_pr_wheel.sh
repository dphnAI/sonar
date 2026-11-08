#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${PR_WHEEL_NUMBER:?PR_WHEEL_NUMBER is required}"
: "${PLATFORM_BACKEND:?PLATFORM_BACKEND is required}"
: "${PLATFORM_ARCHITECTURE:?PLATFORM_ARCHITECTURE is required}"
: "${PLATFORM_VERSION_LOCAL:?PLATFORM_VERSION_LOCAL is required}"
: "${B2_S3_BUCKET:?B2_S3_BUCKET is required}"
: "${B2_PUBLIC_BASE_URL:?B2_PUBLIC_BASE_URL is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"

output_dir="${1:?usage: build_pr_wheel.sh OUTPUT_DIR INDEX_DIR}"
index_root="${2:?missing index output directory}"
rclone="${RUNNER_TEMP}/rclone-bin/rclone"
remote_root="b2:${B2_S3_BUCKET}/pull-requests/${PR_WHEEL_NUMBER}/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
public_root="${B2_PUBLIC_BASE_URL%/}/pull-requests/${PR_WHEEL_NUMBER}/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
target_commit="${TARGET_COMMIT:-${GITHUB_SHA:?GITHUB_SHA is required}}"

if [[ ! -x "$rclone" ]]; then
  echo "::error::rclone is not installed at ${rclone}"
  exit 1
fi

tag="$(git describe --tags --abbrev=0 --match 'v[0-9]*')"
release="${tag#v}"
IFS=. read -r major minor patch <<<"$release"
timestamp="$(date -u +%Y%m%d%H%M%S)"
short_commit="$(git rev-parse --short=9 "$target_commit")"
version="${major}.${minor}.$((patch + 1)).dev${timestamp}+${PLATFORM_VERSION_LOCAL}.pr${PR_WHEEL_NUMBER}.g${short_commit}"

mkdir -p "$output_dir"
find "$output_dir" -maxdepth 1 -type f -name '*.whl' -delete
if [[ -n "${PLATFORM_BUILD_SCRIPT:-}" ]]; then
  SONAR_VERSION_OVERRIDE="$version" "$PLATFORM_BUILD_SCRIPT" "$output_dir"
  wheel="$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
elif [[ "$PLATFORM_BACKEND" == cuda && "$PLATFORM_ARCHITECTURE" == x86_64 ]]; then
  SONAR_VERSION_OVERRIDE="$version" ./docker/export_wheels.sh
  wheel="$(find wheels/main -maxdepth 1 -type f -name '*.whl' -print -quit)"
else
  echo "::error::PLATFORM_BUILD_SCRIPT is required for ${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
  exit 1
fi
if [[ -z "$wheel" ]]; then
  echo "::error::The PR build did not export a ${PLATFORM_BACKEND} wheel"
  exit 1
fi
python3 .github/scripts/verify_wheel_version.py "$wheel" "$version"

wheel_name="$(basename "$wheel")"
if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$wheel" | cut -d ' ' -f 1)"
else
  digest="$(shasum -a 256 "$wheel" | cut -d ' ' -f 1)"
fi

"$rclone" purge "$remote_root" 2>/dev/null || true
"$rclone" copyto "$wheel" "${remote_root}/wheels/${wheel_name}" \
  --s3-upload-cutoff 16M \
  --s3-chunk-size 16M \
  --s3-upload-concurrency 8 \
  --retries 5 \
  --low-level-retries 10 \
  --retries-sleep 5s

encoded_name="${wheel_name//+/%2B}"
entries="${RUNNER_TEMP}/sonar-pr-wheel.tsv"
printf '%s\t%s\t%s\n' \
  "$wheel_name" "${public_root}/wheels/${encoded_name}" "$digest" >"$entries"

index_dir="${index_root}/whl/pr/${PR_WHEEL_NUMBER}/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
package_dir="${index_dir}/simple/sonar"
mkdir -p "$package_dir"
install_command="uv pip install sonar --extra-index-url ${public_root}/simple --index-strategy first-index"
if [[ "$PLATFORM_BACKEND" == cuda ]]; then
  install_command+=" --torch-backend cu130"
fi
python3 .github/scripts/generate_nightly_index.py \
  --entry-file "$entries" \
  --commit "$target_commit" \
  --title "Sonar PR #${PR_WHEEL_NUMBER} ${PLATFORM_BACKEND} wheel" \
  --description "Temporary ${PLATFORM_ARCHITECTURE} test wheel for PR #${PR_WHEEL_NUMBER}." \
  --install-command "$install_command" \
  --output "${index_dir}/index.html"
cp "${index_dir}/index.html" "${package_dir}/index.html"
"$rclone" copyto "${index_dir}/index.html" "${remote_root}/index.html"
"$rclone" copyto "${package_dir}/index.html" "${remote_root}/simple/sonar/index.html"

echo "PR wheel index: ${public_root}/simple/sonar/"
