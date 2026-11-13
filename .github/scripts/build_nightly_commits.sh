#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

: "${B2_S3_BUCKET:?B2_S3_BUCKET is required}"
: "${B2_PUBLIC_BASE_URL:?B2_PUBLIC_BASE_URL is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"

rclone="${RUNNER_TEMP}/rclone-bin/rclone"
bucket="b2:${B2_S3_BUCKET}"
public_base="${B2_PUBLIC_BASE_URL%/}"
state_remote="${bucket}/_ci/last-built-main.txt"
target_commit="${GITHUB_SHA}"
index_root="${1:-manual-wheel-index}"

if [[ ! -x "$rclone" ]]; then
  echo "::error::rclone is not installed at ${rclone}"
  exit 1
fi

encode_path() {
  printf '%s' "${1//+/%2B}"
}

build_with_current_rust_helper() (
  local commit="$1"
  local version="$2"
  # Packaging fixes must also apply to commits awaiting their first wheel.
  trap 'git restore --source="$commit" -- tools/build_rust.py' EXIT
  git show "${target_commit}:tools/build_rust.py" > tools/build_rust.py
  SONAR_VERSION_OVERRIDE="$version" ./docker/export_wheels.sh
)

upload_wheel() {
  local wheel="$1"
  local remote="$2"
  local wheel_name
  wheel_name="$(basename "$wheel")"

  for attempt in {1..5}; do
    echo "Uploading ${wheel_name} (attempt ${attempt}/5)"
    if timeout --signal=TERM --kill-after=30s 20m \
      "$rclone" copyto "$wheel" "$remote" \
        --s3-upload-cutoff 16M \
        --s3-chunk-size 16M \
        --s3-upload-concurrency 8 \
        --retries 5 \
        --low-level-retries 10 \
        --retries-sleep 5s \
        --progress \
        --stats 30s \
        --stats-one-line; then
      return 0
    fi
    echo "Upload attempt ${attempt}/5 failed; retrying in 15 seconds"
    sleep 15
  done

  echo "::error::Wheel upload failed after 5 attempts"
  return 1
}

write_commit_index() {
  local commit="$1"
  local wheel_name="$2"
  local digest="${3:-}"
  local encoded_name
  local entries
  local output

  encoded_name="$(encode_path "$wheel_name")"
  entries="$(mktemp "${RUNNER_TEMP}/sonar-commit-entries.XXXXXX")"
  output="$(mktemp "${RUNNER_TEMP}/sonar-commit-index.XXXXXX")"
  if [[ -n "$digest" ]]; then
    printf '%s\t%s\t%s\n' \
      "$wheel_name" \
      "${public_base}/commits/${commit}/${encoded_name}" \
      "$digest" >"$entries"
  else
    printf '%s\t%s\n' \
      "$wheel_name" \
      "${public_base}/commits/${commit}/${encoded_name}" >"$entries"
  fi

  python3 .github/scripts/generate_nightly_index.py \
    --entry-file "$entries" \
    --commit "$commit" \
    --title "Sonar CUDA wheel for ${commit:0:12}" \
    --install-command \
      "uv pip install sonar --extra-index-url ${public_base}/commits/${commit} --index-strategy first-index --torch-backend cu132" \
    --output "$output"
  "$rclone" copyto \
    "$output" \
    "${bucket}/commits/${commit}/sonar/index.html"
  rm -f "$entries" "$output"
}

find_commit_wheel() {
  local commit="$1"
  "$rclone" lsf \
    "${bucket}/commits/${commit}" \
    --files-only \
    --include 'sonar-*.whl' 2>/dev/null |
    head -n 1 || true
}

last_built=""
if last_built="$("$rclone" cat "$state_remote" 2>/dev/null)"; then
  last_built="${last_built//$'\r'/}"
  last_built="${last_built//$'\n'/}"
fi

base_commit=""
if [[ "$last_built" =~ ^[0-9a-f]{40}$ ]] &&
  git cat-file -e "${last_built}^{commit}" 2>/dev/null; then
  if git merge-base --is-ancestor "$last_built" "$target_commit"; then
    base_commit="$last_built"
  else
    # A force-push can replace already-built commits with equivalent commits
    # that have new IDs. Resume from the histories' common ancestor so every
    # replacement commit receives a wheel.
    base_commit="$(git merge-base "$last_built" "$target_commit" || true)"
  fi
fi

if [[ -z "$base_commit" ]] &&
  [[ "${PUSH_BEFORE:-}" =~ ^[0-9a-f]{40}$ ]] &&
  [[ ! "${PUSH_BEFORE}" =~ ^0+$ ]] &&
  git cat-file -e "${PUSH_BEFORE}^{commit}" 2>/dev/null &&
  git merge-base --is-ancestor "$PUSH_BEFORE" "$target_commit"; then
  base_commit="$PUSH_BEFORE"
fi

if [[ -z "$base_commit" ]] &&
  git rev-parse "${target_commit}^" >/dev/null 2>&1; then
  base_commit="$(git rev-parse "${target_commit}^")"
fi

if [[ -n "$base_commit" ]]; then
  mapfile -t commits < <(git rev-list --reverse "${base_commit}..${target_commit}")
else
  commits=("$target_commit")
fi

echo "Target main commit: ${target_commit}"
echo "Last contiguous build: ${last_built:-none}"
echo "Commits requiring reconciliation: ${#commits[@]}"

for commit in "${commits[@]}"; do
  # Historical commits before the cutover cannot produce the Sonar distribution.
  if ! git show "${commit}:pyproject.toml" | grep -E '^name *= *"sonar"$' >/dev/null; then
    echo "Skipping pre-Sonar commit ${commit}"
    printf '%s\n' "$commit" | "$rclone" rcat "$state_remote"
    continue
  fi
  git checkout --detach "$commit"
  wheel_name="$(find_commit_wheel "$commit")"

  if [[ -n "$wheel_name" ]]; then
    echo "Commit ${commit} already has ${wheel_name}; skipping build"
    write_commit_index "$commit" "$wheel_name"
  else
    tag="$(git describe --tags --abbrev=0 --match 'v[0-9]*')"
    release="${tag#v}"
    IFS=. read -r major minor patch <<<"$release"
    timestamp="$(date -u +%Y%m%d%H%M%S)"
    short_commit="$(git rev-parse --short=9 "$commit")"
    version="${major}.${minor}.$((patch + 1)).dev${timestamp}+cu132.g${short_commit}"
    echo "Building ${version} from ${commit}"

    build_with_current_rust_helper "$commit" "$version"
    wheel="$(find wheels/main -maxdepth 1 -type f -name '*.whl' -print -quit)"
    if [[ -z "$wheel" ]]; then
      echo "::error::The build did not export a wheel for ${commit}"
      exit 1
    fi

    wheel_name="$(basename "$wheel")"
    wheel_remote="${bucket}/commits/${commit}/${wheel_name}"
    digest="$(sha256sum "$wheel" | cut -d ' ' -f 1)"
    upload_wheel "$wheel" "$wheel_remote"

    encoded_name="$(encode_path "$wheel_name")"
    curl --fail --silent --show-error --location \
      --retry 10 \
      --retry-all-errors \
      --retry-delay 3 \
      --range 0-0 \
      --output /dev/null \
      "${public_base}/commits/${commit}/${encoded_name}"

    write_commit_index "$commit" "$wheel_name" "$digest"
    rm -f "$wheel"
    docker buildx prune --builder default --force \
      --filter type=regular \
      --max-used-space 25gb
  fi

  printf '%s\n' "$commit" | "$rclone" rcat "$state_remote"
done

git checkout --detach "$target_commit"

entries="${RUNNER_TEMP}/sonar-wheel-index.tsv"
: >"$entries"
while IFS= read -r stored_path; do
  [[ -n "$stored_path" ]] || continue
  wheel_name="$(basename "$stored_path")"
  encoded_path="$(encode_path "$stored_path")"
  printf '%s\t%s\n' \
    "$wheel_name" \
    "${public_base}/${encoded_path}" >>"$entries"
done < <(
  {
    "$rclone" lsf \
      "${bucket}/wheels" \
      --files-only \
      --include 'sonar-*.whl' 2>/dev/null |
      sed 's#^#wheels/#'
    "$rclone" lsf \
      "${bucket}/commits" \
      --recursive \
      --files-only \
      --include 'sonar-*.whl' 2>/dev/null |
      sed 's#^#commits/#'
  } | sort -r
)

index_dir="${index_root}/whl/nightly/cuda/x86_64"
package_dir="${index_dir}/simple/sonar"
legacy_nightly_dir="${index_root}/nightly/sonar"
legacy_simple_dir="${index_root}/simple/sonar"
nightly_cuda_dir="${index_root}/nightly/cuda/x86_64"
mkdir -p \
  "$package_dir" \
  "$legacy_nightly_dir" \
  "$legacy_simple_dir" \
  "$nightly_cuda_dir"

python3 .github/scripts/generate_nightly_index.py \
  --entry-file "$entries" \
  --commit "$target_commit" \
  --title "Sonar nightly CUDA wheels" \
  --description "Precompiled CUDA 13.2 wheels for x86_64 from every commit on main." \
  --install-command \
  "uv pip install sonar --extra-index-url https://sonar.dphn.ai/whl/nightly/cuda/x86_64/simple --index-strategy first-index --torch-backend cu132" \
  --output "${index_dir}/index.html"
cp "${index_dir}/index.html" "${package_dir}/index.html"
cp "${index_dir}/index.html" "${legacy_nightly_dir}/index.html"
cp "${index_dir}/index.html" "${legacy_simple_dir}/index.html"
cp "${index_dir}/index.html" "${nightly_cuda_dir}/index.html"
mkdir -p "${index_root}/whl"
cp "${index_dir}/index.html" "${index_root}/whl/index.html"
"$rclone" copyto \
  "${index_dir}/index.html" \
  "${bucket}/platform-wheels/nightly/cuda/x86_64/index.html"

tip_wheel="$(find_commit_wheel "$target_commit")"
if [[ -z "$tip_wheel" ]]; then
  echo "::error::No wheel is available for target commit ${target_commit}"
  exit 1
fi
"$rclone" copyto \
  "${bucket}/commits/${target_commit}/sonar/index.html" \
  "${bucket}/nightly/sonar/index.html"
