# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU verification of sampled runtime-weight receipts.

A valid receipt shows that the worker could read the sampled checkpoint bytes
when the response finished. It does not attest that those weights produced the
response.
"""

from __future__ import annotations

import blake3
import cbor2
import pybase64 as base64
import torch

from aphrodite.proof import PROOF_VERSION

MAX_ENVELOPE_CHARS = 1_048_576
# Receipts are taken once per finished choice; the field is fixed in the format.
SCHEDULER_STEP = 0
# Exactly the payload fields emitted by the native prover (csrc/proof/proof.cpp).
PAYLOAD_FIELDS = frozenset(
    {
        "version",
        "worker_id",
        "request_id",
        "gateway_nonce",
        "model_manifest_id",
        "scheduler_step",
        "tensor_registry_hash",
        "generated_token_hash",
        "terms",
        "weight_proof",
    }
)


def _b64url_decode(value: str) -> bytes:
    value = value.strip()
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _load_x25519_public_key(value: str):
    from cryptography.hazmat.primitives.asymmetric import x25519

    encoded = value.strip().encode("utf-8")
    if value.strip().startswith("-----BEGIN"):
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_public_key(encoded)
    raw = _b64url_decode(value)
    if len(raw) != 32:
        raw = base64.b64decode(value)
    return x25519.X25519PublicKey.from_public_bytes(raw)


def _load_x25519_private_key(value: str):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    encoded = value.strip().encode("utf-8")
    if value.strip().startswith("-----BEGIN"):
        return serialization.load_pem_private_key(encoded, password=None)
    raw = _b64url_decode(value)
    if len(raw) != 32:
        raw = base64.b64decode(value)
    return x25519.X25519PrivateKey.from_private_bytes(raw)


def generate_x25519_keypair() -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    private_key = x25519.X25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64url_encode(private_raw), _b64url_encode(public_raw)


def _derive_key(shared_secret: bytes, aad: bytes):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"dphn-proof-v2" + aad,
    ).derive(shared_secret)


def _aad(worker_id: str, gateway_nonce: str, model_manifest_id: str) -> bytes:
    fields = (worker_id, gateway_nonce, model_manifest_id)
    if any(not isinstance(value, str) or not value or "\0" in value for value in fields):
        raise ValueError("Envelope identifiers must be nonempty strings without NUL")
    return b"\0".join(value.encode() for value in fields)


def _decode_wire(data: bytes):
    # Native v2 uses only definite integers, strings, bytes, arrays and maps.
    # Reject semantic/shared-reference tags before cbor2 can materialize them.
    if not data or len(data) > MAX_ENVELOPE_CHARS:
        raise ValueError("Invalid CBOR message size")
    position = 0
    pending = [1]
    while pending:
        if not pending[-1]:
            pending.pop()
            continue
        if len(pending) > 8 or position >= len(data):
            raise ValueError("Invalid CBOR nesting or truncated message")
        pending[-1] -= 1
        initial = data[position]
        position += 1
        major, additional = initial >> 5, initial & 31
        if major > 5 or additional > 27:
            raise ValueError("Unsupported CBOR type")
        if additional < 24:
            length = additional
        else:
            width = 1 << (additional - 24)
            if position + width > len(data):
                raise ValueError("Truncated CBOR length")
            length = int.from_bytes(data[position : position + width], "big")
            position += width
        if major in (2, 3):
            position += length
            if position > len(data):
                raise ValueError("Truncated CBOR string")
        elif major in (4, 5):
            count = length * (2 if major == 5 else 1)
            if count > len(data) - position:
                raise ValueError("Invalid CBOR container length")
            pending.append(count)
    if position != len(data):
        raise ValueError("Trailing CBOR data")
    try:
        value = cbor2.loads(data)
    except cbor2.CBORDecodeError as exc:
        raise ValueError("Invalid CBOR message") from exc
    if cbor2.dumps(value, canonical=True) != data:
        raise ValueError("CBOR message must be canonical")
    return value


def decrypt_payload(
    envelope_b64: str, private_key: str, *, worker_id: str, gateway_nonce: str, model_manifest_id: str
) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    if not isinstance(envelope_b64, str) or len(envelope_b64) > MAX_ENVELOPE_CHARS:
        raise ValueError("Invalid proof envelope size")
    aad = _aad(worker_id, gateway_nonce, model_manifest_id)
    envelope_bytes = _b64url_decode(envelope_b64)
    envelope = _decode_wire(envelope_bytes)
    if not isinstance(envelope, dict) or set(envelope) != {"v", "kid", "epk", "n", "ct"}:
        raise ValueError("Invalid proof envelope")
    if type(envelope["v"]) is not int or envelope["v"] != PROOF_VERSION:
        raise ValueError("Unsupported proof version")
    if not isinstance(envelope["kid"], str):
        raise ValueError("Invalid key identifier")
    for key, length in (("epk", 32), ("n", 12)):
        if not isinstance(envelope[key], bytes) or len(envelope[key]) != length:
            raise ValueError(f"Invalid envelope {key}")
    if not isinstance(envelope["ct"], bytes) or len(envelope["ct"]) < 16:
        raise ValueError("Invalid ciphertext")
    private = _load_x25519_private_key(private_key)
    ephemeral_public = _load_x25519_public_key(_b64url_encode(envelope["epk"]))
    key = _derive_key(private.exchange(ephemeral_public), aad)
    plaintext = ChaCha20Poly1305(key).decrypt(envelope["n"], envelope["ct"], aad)
    payload = _decode_wire(plaintext)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a CBOR map")
    expected = {"worker_id": worker_id, "gateway_nonce": gateway_nonce, "model_manifest_id": model_manifest_id}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Envelope context mismatch")
    return payload


def _u64_from_seed(seed: bytes, counter: int) -> int:
    h = blake3.blake3(seed)
    h.update(counter.to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little")


def _validate_context(nonce, request_id, model_manifest_id, output_token_ids):
    _aad(request_id, nonce, model_manifest_id)
    if not isinstance(output_token_ids, list) or any(
        type(token) is not int or not 0 <= token < 2**63 for token in output_token_ids
    ):
        raise ValueError("output_token_ids must be a list of nonnegative int64 values")


def _challenge_seed(nonce, request_id, model_manifest_id, output_token_ids):
    token_hash = blake3.blake3(cbor2.dumps(output_token_ids, canonical=True)).hexdigest()
    material = {
        "nonce": nonce,
        "request_id": request_id,
        "model_manifest_id": model_manifest_id,
        "scheduler_step": SCHEDULER_STEP,
        "generated_token_hash": token_hash,
    }
    return blake3.blake3(cbor2.dumps(material, canonical=True)).digest(), token_hash


def _registry_entries(state_dict: dict) -> list[dict]:
    entries = []
    seen = set()
    for name, tensor in state_dict.items():
        if tensor is None:
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("Reference state must map nonempty names to tensors")
        tensor = tensor.detach()
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("Reference tensors must be strided CPU tensors")
        if tensor.is_quantized:
            raise ValueError("Quantized tensor objects require explicit qparameter handling")
        if tensor.numel() == 0 or not tensor.is_contiguous():
            continue
        identity = tensor.data_ptr()
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "num_bytes": tensor.numel() * tensor.element_size(),
            }
        )
    entries.sort(key=lambda entry: entry["name"])
    return entries


def compute_registry_hash(state_dict: dict) -> str:
    return blake3.blake3(cbor2.dumps(_registry_entries(state_dict), canonical=True)).hexdigest()


def _reconstruct_schedule(seed, num_raw_terms, raw_term_bytes, entries):
    terms = []
    for index in range(num_raw_terms):
        entry = entries[_u64_from_seed(seed, index * 2) % len(entries)]
        length = min(raw_term_bytes, entry["num_bytes"])
        offset = _u64_from_seed(seed, index * 2 + 1) % (entry["num_bytes"] - length + 1)
        terms.append({"op": "raw_window_hash", "tensor": entry["name"], "offset": offset, "length": length})
    return terms


def _window_digest(tensor, offset, length):
    byte_view = tensor.detach().reshape(-1).view(torch.uint8)
    return blake3.blake3(byte_view[offset : offset + length].numpy()).hexdigest()


def _matches_context(payload, nonce, request_id, model_manifest_id, token_hash):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        return False
    expected = {
        "version": PROOF_VERSION,
        "gateway_nonce": nonce,
        "request_id": request_id,
        "model_manifest_id": model_manifest_id,
        "scheduler_step": SCHEDULER_STEP,
        "generated_token_hash": token_hash,
    }
    return all(type(payload.get(key)) is type(value) and payload[key] == value for key, value in expected.items())


def verify_proof(
    payload: dict,
    state_dict: dict,
    *,
    worker_id: str,
    nonce: str,
    request_id: str,
    model_manifest_id: str,
    output_token_ids: list[int],
    num_raw_terms: int = 8,
    raw_term_bytes: int = 256,
) -> bool:
    """Check a decrypted receipt payload against trusted CPU weights."""
    _aad(worker_id, nonce, model_manifest_id)
    _validate_context(nonce, request_id, model_manifest_id, output_token_ids)
    if type(num_raw_terms) is not int or not 1 <= num_raw_terms < 2**63:
        raise ValueError("num_raw_terms must be a positive int64")
    if type(raw_term_bytes) is not int or not 16 <= raw_term_bytes < 2**63:
        raise ValueError("raw_term_bytes must be an int64 of at least 16")
    seed, token_hash = _challenge_seed(nonce, request_id, model_manifest_id, output_token_ids)
    if not _matches_context(payload, nonce, request_id, model_manifest_id, token_hash):
        return False
    if payload.get("worker_id") != worker_id:
        return False
    entries = _registry_entries(state_dict)
    if not entries:
        return False
    registry_hash = blake3.blake3(cbor2.dumps(entries, canonical=True)).hexdigest()
    if payload.get("tensor_registry_hash") != registry_hash:
        return False
    actual_terms = payload.get("terms")
    if not isinstance(actual_terms, list) or len(actual_terms) != num_raw_terms:
        return False
    expected_terms = _reconstruct_schedule(seed, num_raw_terms, raw_term_bytes, entries)
    sketches = []
    for expected, actual in zip(expected_terms, actual_terms):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        if any(type(actual[key]) is not type(value) or actual[key] != value for key, value in expected.items()):
            return False
        digest = _window_digest(state_dict[expected["tensor"]], expected["offset"], expected["length"])
        sketches.append({**expected, "digest": digest})
    weight_proof = blake3.blake3(cbor2.dumps(sketches, canonical=True)).hexdigest()
    return payload.get("weight_proof") == weight_proof
