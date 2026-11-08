#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

control_host="${DOCKER_HOST:-tcp://127.0.0.1:23750}"
host_control="${CUDA_ARM64_HOST_DOCKER_HOST:-unix:///var/run/docker.sock}"
dind_container="${CUDA_ARM64_DIND_CONTAINER:-aphrodite-nightly-docker}"
builder_name="${CUDA_ARM64_BUILDER_NAME:-sonar-cuda-arm64}"
container_name="${CUDA_ARM64_BUILDKIT_CONTAINER:-sonar-cuda-arm64-buildkit}"
builder_port="${CUDA_ARM64_BUILDKIT_PORT:-12352}"
state_dir=/home/user/.local/share/buildkit
# Includes BuildKit's userspace ARM64 emulator; no host binfmt registration.
image=moby/buildkit@sha256:504731e577c20559c00f968f33219f30115e70be29ab96728d1d06e963fc494b

dind_address="$(docker --host "$host_control" inspect --format \
  '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$dind_container")"
test -n "$dind_address"
endpoint="tcp://${dind_address}:${builder_port}"

if ! docker --host "$control_host" inspect "$container_name" >/dev/null 2>&1; then
  # Match native CUDA: 300 GiB of tmpfs, without CPU or memory cgroup caps.
  docker --host "$control_host" run --detach \
    --name "$container_name" \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --publish "${builder_port}:1234" \
    --tmpfs "${state_dir}:rw,exec,size=300g,mode=1777" \
    "$image" \
    --oci-worker-no-process-sandbox \
    --oci-worker-platform linux/amd64 \
    --oci-worker-platform linux/arm64 \
    --oci-worker-gc=false \
    --addr tcp://0.0.0.0:1234
elif [[ "$(docker --host "$control_host" inspect --format '{{.State.Running}}' "$container_name")" != true ]]; then
  docker --host "$control_host" start "$container_name" >/dev/null
fi

configured_endpoint="$(docker buildx inspect "$builder_name" 2>/dev/null |
  sed -n 's/^Endpoint:[[:space:]]*//p' || true)"
if [[ -n "$configured_endpoint" && "$configured_endpoint" != "$endpoint" ]]; then
  echo "Builder ${builder_name} already uses ${configured_endpoint}, expected ${endpoint}." >&2
  exit 1
fi
if [[ -z "$configured_endpoint" ]]; then
  docker buildx create --name "$builder_name" --driver remote "$endpoint"
fi
# Do not change the user's default builder: x86 and ROCm jobs share this host.
for _ in $(seq 1 60); do
  if docker buildx inspect "$builder_name" --bootstrap >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done

docker --host "$control_host" logs --tail 100 "$container_name" >&2 || true
echo "ARM64 CUDA BuildKit did not become ready at ${endpoint}." >&2
exit 1
