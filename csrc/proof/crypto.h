// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Crypto helpers for the DPHN proof envelope. X25519 + HKDF-SHA256 +
// ChaCha20-Poly1305 (IETF), plus base64url and key parsing. The wire format
// must match sonar.proof.verifier.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dphn_crypto {

// --- Hashing / KDF -------------------------------------------------------

void sha256(const uint8_t* data, size_t len, uint8_t out[32]);

void hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data,
                 size_t data_len, uint8_t out[32]);

// HKDF-SHA256 with an empty (all-zero, hash-length) salt, expanded to
// exactly 32 bytes. Matches cryptography's HKDF(salt=None, length=32).
void hkdf_sha256_32(const uint8_t* ikm, size_t ikm_len, const uint8_t* info,
                    size_t info_len, uint8_t out[32]);

// --- X25519 ---------------------------------------------------------------

void x25519_public_key(uint8_t public_key[32], const uint8_t private_key[32]);

// Returns false when the shared secret is all zeros (a low-order point);
// mirrors cryptography's exchange() failure.
bool x25519_shared_secret(uint8_t shared[32], const uint8_t private_key[32],
                          const uint8_t peer_public[32]);

// --- AEAD -----------------------------------------------------------------

// RFC 8439 ChaCha20-Poly1305 (IETF, 12-byte nonce). Returns
// ciphertext || 16-byte tag, matching cryptography's
// ChaCha20Poly1305.encrypt().
std::vector<uint8_t> chacha20poly1305_ietf_encrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const std::vector<uint8_t>& aad, const std::vector<uint8_t>& plaintext);

// --- Encoding / key parsing ------------------------------------------------

std::string b64url_encode_nopad(const uint8_t* data, size_t len);

// Accepts both the standard and the URL-safe alphabet, with or without
// padding (mirrors the verifier's b64url-then-b64 fallback).
std::vector<uint8_t> b64_decode_flexible(const std::string& value);

// Parses an X25519 public key given as PEM (SubjectPublicKeyInfo) or as a
// base64url/base64 raw 32-byte key. Throws a runtime error on failure,
// mirroring _load_x25519_public_key in sonar.proof.verifier.
std::vector<uint8_t> load_x25519_public_key(const std::string& value);

// --- Randomness ------------------------------------------------------------

// Cryptographically secure random bytes (getrandom / /dev/urandom).
// Throws if no entropy source is available.
void random_bytes(uint8_t* out, size_t len);

}  // namespace dphn_crypto
