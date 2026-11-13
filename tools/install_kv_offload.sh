#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Install the optional KVCR offload tier against the active CUDA runtime.
# Usage: tools/install_kv_offload.sh
set -euo pipefail

CUDA_MAJOR=$(python3 -c 'import torch; print((torch.version.cuda or "").split(".", 1)[0])')
if [[ -z "${CUDA_MAJOR}" ]]; then
    exit 0
fi

uv_target=()
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    uv_target=(--system)
fi

uv pip install "${uv_target[@]}" \
    "kvcr @ git+https://github.com/ai-dynamo/kvcr.git@5ce23b1527725cf623a3fc7feb94dc6d3d5fafaa"

# KVCR depends on NIXL; keep its version but select the matching CUDA wheel.
NIXL_VERSION=$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("nixl"))')
uv pip uninstall "${uv_target[@]}" nixl-cu12 nixl-cu13
uv pip install "${uv_target[@]}" --no-deps "nixl-cu${CUDA_MAJOR}==${NIXL_VERSION}"
