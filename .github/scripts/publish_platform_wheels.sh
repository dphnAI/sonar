#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

wheel_dir="${1:?usage: publish_platform_wheels.sh WHEEL_DIR CHANNEL BACKEND ARCH OUTPUT_DIR}"
channel="${2:?missing channel}"
backend="${3:?missing backend}"
architecture="${4:?missing architecture}"
output_dir="${5:?missing output directory}"

: "${B2_S3_BUCKET:?B2_S3_BUCKET is not configured}"
: "${B2_PUBLIC_BASE_URL:?B2_PUBLIC_BASE_URL is not configured}"

rclone="${RCLONE_BIN:-rclone}"
remote_root="b2:${B2_S3_BUCKET}/platform-wheels/${channel}/${backend}/${architecture}"
public_root="${B2_PUBLIC_BASE_URL%/}/platform-wheels/${channel}/${backend}/${architecture}"

shopt -s nullglob
wheels=("$wheel_dir"/sonar-*.whl)
((${#wheels[@]} > 0)) || {
  echo "No Sonar wheels found in ${wheel_dir}" >&2
  exit 1
}

for wheel in "${wheels[@]}"; do
  "$rclone" copyto "$wheel" "${remote_root}/wheels/$(basename "$wheel")" \
    --s3-upload-concurrency 8 \
    --s3-chunk-size 64M
done

if [[ -n "${PRUNE_RELEASE_VERSION:-}" ]]; then
  declare -A current_wheels=()
  for wheel in "${wheels[@]}"; do
    current_wheels["$(basename "$wheel")"]=1
  done

  if [[ "$backend" == cuda ]]; then
    wheel_prefix="sonar-${PRUNE_RELEASE_VERSION}-"
  else
    wheel_prefix="sonar-${PRUNE_RELEASE_VERSION}+${backend}-"
  fi
  while IFS= read -r wheel_name; do
    [[ "$wheel_name" == "${wheel_prefix}"*.whl ]] || continue
    [[ -n "${current_wheels[$wheel_name]:-}" ]] && continue
    "$rclone" deletefile "${remote_root}/wheels/${wheel_name}"
  done < <("$rclone" lsf "${remote_root}/wheels" --files-only --include '*.whl')
fi

entries="${RUNNER_TEMP:-/tmp}/sonar-${channel}-${backend}-${architecture}.tsv"
: >"$entries"
while IFS= read -r wheel_name; do
  [[ -n "$wheel_name" ]] || continue
  local_wheel="${wheel_dir}/${wheel_name}"
  if [[ -f "$local_wheel" ]]; then
    digest="$(sha256sum "$local_wheel" | cut -d ' ' -f 1)"
    printf '%s\t%s\t%s\n' \
      "$wheel_name" \
      "${public_root}/wheels/${wheel_name}" \
      "$digest" >>"$entries"
  else
    printf '%s\t%s\n' \
      "$wheel_name" \
      "${public_root}/wheels/${wheel_name}" >>"$entries"
  fi
done < <(
  "$rclone" lsf "${remote_root}/wheels" --files-only --include 'sonar-*.whl' |
    sort -r
)

index_dir="${output_dir}/whl/${channel}/${backend}/${architecture}"
package_dir="${index_dir}/simple/sonar"
mkdir -p "$package_dir"
install_command="uv pip install sonar --extra-index-url https://sonar.dphn.ai/whl/${channel}/${backend}/${architecture}/simple --index-strategy first-index"
case "$backend" in
  cuda) install_command+=" --torch-backend cu130" ;;
  cpu) install_command="curl -LsSf https://sonar.dphn.ai/install.sh | sh -s -- --backend cpu --channel ${channel}" ;;
esac

python3 .github/scripts/generate_nightly_index.py \
  --entry-file "$entries" \
  --commit "${GITHUB_SHA:-release}" \
  --title "Sonar ${channel} ${backend} wheels" \
  --description "Precompiled ${backend} wheels for ${architecture}." \
  --install-command "$install_command" \
  --output "${index_dir}/index.html"

cp "${index_dir}/index.html" "${package_dir}/index.html"
"$rclone" copyto "${index_dir}/index.html" "${remote_root}/index.html"
