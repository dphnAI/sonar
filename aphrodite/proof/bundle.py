# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify rank coverage against a locally approved runtime-weight manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag

from aphrodite.proof import PROOF_VERSION, rank_worker_id
from aphrodite.proof.verifier import (
    _load_x25519_private_key,
    compute_registry_hash,
    decrypt_payload,
    verify_proof,
)

# Closed schemas of the per-choice bundle and its per-rank receipts, as emitted
# by both the Python (aphrodite/proof/http.py) and Rust frontends.
BUNDLE_FIELDS = frozenset({"version", "request_id", "nonce", "output_token_ids", "proofs"})
RECEIPT_FIELDS = frozenset({"rank", "dp_rank", "worker_id", "proof"})


class RuntimeWeightVerifier:
    """Load trusted CPU snapshots once, then check receipts without GPU work.

    Manifests and snapshots must come from a trusted exporter, never the worker
    being checked. Acceptance does not establish per-request inference provenance.
    """

    def __init__(self, manifest_path: str | Path, private_key: str):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from safetensors.torch import load_file

        if not isinstance(_load_x25519_private_key(private_key), X25519PrivateKey):
            raise ValueError("Verification requires an X25519 private key")
        self.private_key = private_key
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict) or manifest.get("version") != PROOF_VERSION:
            raise ValueError("Unsupported runtime-weight manifest")
        workers = manifest.get("workers")
        if not isinstance(workers, list) or not workers:
            raise ValueError("Manifest must approve at least one worker rank")
        self.worker_id = manifest["worker_id"]
        self.model_manifest_id = manifest["model_manifest_id"]
        if any(
            not isinstance(value, str) or not value or "\0" in value
            for value in (self.worker_id, self.model_manifest_id)
        ):
            raise ValueError("Manifest identifiers must be nonempty strings without NUL")
        self.workers: dict[tuple[int, int], tuple[dict, dict]] = {}
        for record in workers:
            if not isinstance(record, dict):
                raise ValueError("Invalid manifest worker")
            rank, dp_rank = record.get("rank"), record.get("dp_rank")
            if type(rank) is not int or type(dp_rank) is not int or min(rank, dp_rank) < 0:
                raise ValueError("Manifest rank identifiers must be nonnegative integers")
            identity = (dp_rank, rank)
            if identity in self.workers:
                raise ValueError("Manifest contains a duplicate worker rank")
            if record.get("version") != PROOF_VERSION:
                raise ValueError("Unsupported worker manifest version")
            if record.get("worker_id") != rank_worker_id(self.worker_id, dp_rank, rank):
                raise ValueError("Manifest worker identity mismatch")
            if record.get("model_manifest_id") != self.model_manifest_id:
                raise ValueError("Manifest model identity mismatch")
            terms, window = record.get("num_raw_terms"), record.get("raw_term_bytes")
            if type(terms) is not int or not 1 <= terms <= 64:
                raise ValueError("Invalid manifest term count")
            if type(window) is not int or not 16 <= window <= 4096:
                raise ValueError("Invalid manifest byte-window size")
            snapshot = record.get("snapshot")
            if not isinstance(snapshot, str) or Path(snapshot).name != snapshot:
                raise ValueError("Snapshots must be filenames in the manifest directory")
            state = load_file(str(path.parent / snapshot), device="cpu")
            if not state or compute_registry_hash(state) != record.get("registry_hash"):
                raise ValueError(f"Runtime snapshot registry mismatch for rank {identity}")
            self.workers[identity] = (record, state)

    def verify(
        self,
        bundle: dict[str, Any],
        *,
        nonce: str,
        request_id: str,
        output_token_ids: list[int],
    ) -> bool:
        """Expected request context must come from the gateway, not the receipt."""
        if not isinstance(bundle, dict) or set(bundle) != BUNDLE_FIELDS or bundle["version"] != PROOF_VERSION:
            return False
        if bundle.get("request_id") != request_id or bundle.get("nonce") != nonce:
            return False
        tokens = bundle.get("output_token_ids")
        if (
            not isinstance(tokens, list)
            or any(type(token) is not int for token in tokens)
            or tokens != output_token_ids
        ):
            return False
        proofs = bundle.get("proofs")
        if not isinstance(proofs, list) or len(proofs) != len(self.workers):
            return False
        seen = set()
        try:
            for receipt in proofs:
                if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
                    return False
                rank, dp_rank = receipt.get("rank"), receipt.get("dp_rank")
                if type(rank) is not int or type(dp_rank) is not int:
                    return False
                identity = (dp_rank, rank)
                if identity in seen or identity not in self.workers:
                    return False
                seen.add(identity)
                record, state = self.workers[identity]
                if receipt.get("worker_id") != record["worker_id"]:
                    return False
                envelope = receipt.get("proof")
                if not isinstance(envelope, str) or len(envelope) > 1_048_576:
                    return False
                payload = decrypt_payload(
                    envelope,
                    self.private_key,
                    worker_id=record["worker_id"],
                    gateway_nonce=nonce,
                    model_manifest_id=self.model_manifest_id,
                )
                if not verify_proof(
                    payload,
                    state,
                    worker_id=record["worker_id"],
                    nonce=nonce,
                    request_id=request_id,
                    model_manifest_id=self.model_manifest_id,
                    output_token_ids=output_token_ids,
                    num_raw_terms=record["num_raw_terms"],
                    raw_term_bytes=record["raw_term_bytes"],
                ):
                    return False
        except (InvalidTag, ValueError, TypeError, KeyError, OverflowError):
            return False
        return seen == set(self.workers)
