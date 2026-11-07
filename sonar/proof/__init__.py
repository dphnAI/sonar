# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in sampled runtime-weight receipts, not proof of inference."""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from aphrodite.config.dphn import DphnNetworkConfig

PROOF_VERSION = 2
NONCE_HEADER = "X-DPHN-Proof-Nonce"


@lru_cache(maxsize=1)
def ensure_native_available() -> None:
    import torch  # noqa: F401 -- load ATen shared libraries before the extension

    try:
        importlib.import_module("aphrodite._proof_C")
    except ImportError as exc:
        raise RuntimeError(
            "--dphn-network requires aphrodite._proof_C. Install a wheel containing "
            "the extension or build/install the _proof_C CMake target. Refusing "
            "to serve without weight receipts."
        ) from exc


def rank_worker_id(base: str, dp_rank: int, rank: int) -> str:
    return f"{base}/dp/{dp_rank}/rank/{rank}"


def _runtime_tensors(model: nn.Module):
    entries = list(model.named_parameters()) + list(model.named_buffers())
    return [tensor for _, tensor in entries], [name for name, _ in entries]


def generate_weight_receipt(
    model: nn.Module,
    config: DphnNetworkConfig,
    *,
    worker_id: str,
    request_id: str,
    nonce: str,
    output_token_ids: list[int],
) -> str:
    import torch

    ensure_native_available()
    tensors, names = _runtime_tensors(model)
    return torch.ops._proof_C.prove(
        tensors,
        names,
        worker_id,
        request_id,
        nonce,
        config.model_manifest_id,
        output_token_ids,
        config.public_key,
        config.key_id,
        config.num_raw_terms,
        config.raw_term_bytes,
    )


def dump_runtime_snapshot(
    model: nn.Module,
    config: DphnNetworkConfig,
    *,
    rank: int,
    dp_rank: int,
    output_dir: str,
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    """Export an approved runtime layout; invoke only on trusted model workers."""
    import torch
    from safetensors.torch import save_file

    from aphrodite.proof.verifier import compute_registry_hash

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, names = _runtime_tensors(model)
    state = {}
    seen = set()
    for name, tensor in zip(names, tensors):
        tensor = tensor.detach()
        if tensor.device.type not in ("cpu", "cuda", "mps"):
            continue
        if tensor.layout != torch.strided or tensor.numel() == 0 or not tensor.is_contiguous():
            continue
        if tensor.is_quantized:
            raise ValueError("Quantized tensor objects require explicit qparameter handling")
        identity = (tensor.device.type, tensor.data_ptr())
        if identity in seen:
            continue
        seen.add(identity)
        state[name] = tensor.cpu()
    if not state:
        raise ValueError("The model has no eligible runtime tensors")
    filename = f"dp-{dp_rank}-rank-{rank}.safetensors"
    save_file(state, str(directory / filename))
    record = {
        "version": PROOF_VERSION,
        "worker_id": rank_worker_id(config.worker_id, dp_rank, rank),
        "rank": rank,
        "dp_rank": dp_rank,
        "model_manifest_id": config.model_manifest_id,
        "num_raw_terms": config.num_raw_terms,
        "raw_term_bytes": config.raw_term_bytes,
        "snapshot": filename,
        "registry_hash": compute_registry_hash(state),
        "runtime": runtime_info,
    }
    (directory / f"{filename}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
