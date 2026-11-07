# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from sonar.proof import dump_runtime_snapshot, generate_weight_receipt, rank_worker_id
from sonar.proof.bundle import RuntimeWeightVerifier
from sonar.proof.verifier import decrypt_payload, generate_x25519_keypair, verify_proof


@pytest.fixture
def receipt_set(tmp_path):
    pytest.importorskip("sonar._proof_C")
    private, public = generate_x25519_keypair()
    config = SimpleNamespace(
        public_key=public,
        worker_id="test-worker",
        model_manifest_id="test-model",
        key_id="test-key",
        num_raw_terms=8,
        raw_term_bytes=256,
    )
    models, records = [], []
    for rank in range(2):
        model = torch.nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5 + rank)
        model.register_buffer("scale", torch.tensor(0.5))
        models.append(model)
        records.append(
            dump_runtime_snapshot(
                model,
                config,
                rank=rank,
                dp_rank=0,
                output_dir=str(tmp_path),
                runtime_info={"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "worker_id": config.worker_id,
                "model_manifest_id": config.model_manifest_id,
                "workers": records,
            }
        )
    )
    verifier = RuntimeWeightVerifier(manifest, private)
    context = {"nonce": "gateway-challenge", "request_id": "request/choice/0", "output_token_ids": [17, 23]}

    def make_bundle():
        proofs = []
        for rank, model in enumerate(models):
            worker_id = rank_worker_id(config.worker_id, 0, rank)
            proof = generate_weight_receipt(model, config, worker_id=worker_id, **context)
            proofs.append({"rank": rank, "dp_rank": 0, "worker_id": worker_id, "proof": proof})
        return {"version": 2, **context, "proofs": proofs}

    return verifier, context, models, make_bundle


def test_weight_receipts_verify_in_any_rank_order(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    bundle = make_bundle()
    assert verifier.verify(bundle, **context)
    bundle["proofs"].reverse()
    assert verifier.verify(bundle, **context)


def test_rank_coverage_and_rank_binding(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    valid = make_bundle()
    missing = copy.deepcopy(valid)
    missing["proofs"].pop()
    assert not verifier.verify(missing, **context)
    duplicate = copy.deepcopy(valid)
    duplicate["proofs"][1] = duplicate["proofs"][0]
    assert not verifier.verify(duplicate, **context)
    substituted = copy.deepcopy(valid)
    substituted["proofs"][0]["proof"], substituted["proofs"][1]["proof"] = (
        substituted["proofs"][1]["proof"],
        substituted["proofs"][0]["proof"],
    )
    assert not verifier.verify(substituted, **context)


def test_changed_live_weights_fail_against_approved_snapshot(receipt_set):
    verifier, context, models, make_bundle = receipt_set
    with torch.no_grad():
        for model in models:
            model.weight.add_(1)
            model.scale.add_(1)
    assert not verifier.verify(make_bundle(), **context)


def test_expected_context_cannot_be_replaced_by_receipt_metadata(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    bundle = make_bundle()
    changed_context = {**context, "nonce": "different-request"}
    bundle["nonce"] = changed_context["nonce"]
    assert not verifier.verify(bundle, **changed_context)
    bundle = make_bundle()
    changed_context = {**context, "output_token_ids": [999]}
    bundle["output_token_ids"] = changed_context["output_token_ids"]
    assert not verifier.verify(bundle, **changed_context)


def test_malformed_ciphertext_rejected_without_exception(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    bundle = make_bundle()
    bundle["proofs"][0]["proof"] = "not-an-envelope"
    assert not verifier.verify(bundle, **context)


def test_bundle_and_receipt_maps_are_closed(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    bundle = make_bundle()
    bundle["unexpected"] = True
    assert not verifier.verify(bundle, **context)
    bundle = make_bundle()
    bundle["proofs"][0]["unexpected"] = True
    assert not verifier.verify(bundle, **context)


def test_decrypted_payload_map_is_closed(receipt_set):
    verifier, context, _, make_bundle = receipt_set
    receipt = make_bundle()["proofs"][0]
    record, state = verifier.workers[(0, 0)]
    payload = decrypt_payload(
        receipt["proof"],
        verifier.private_key,
        worker_id=record["worker_id"],
        gateway_nonce=context["nonce"],
        model_manifest_id=verifier.model_manifest_id,
    )
    check = {
        "worker_id": record["worker_id"],
        "model_manifest_id": verifier.model_manifest_id,
        "num_raw_terms": record["num_raw_terms"],
        "raw_term_bytes": record["raw_term_bytes"],
        **context,
    }
    assert verify_proof(payload, state, **check)
    assert not verify_proof({**payload, "exec_fingerprint": {}}, state, **check)
    missing = dict(payload)
    del missing["tensor_registry_hash"]
    assert not verify_proof(missing, state, **check)
