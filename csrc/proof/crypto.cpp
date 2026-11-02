// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "crypto.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "monocypher.h"

#if defined(__linux__) || defined(__APPLE__)
  #include <unistd.h>
  #if defined(__linux__)
    #include <sys/random.h>
  #endif
  #include <cerrno>
  #include <cstdio>
#endif

namespace dphn_crypto {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Straightforward public-domain-style implementation;
// only used to build HKDF-SHA256 for the proof envelope key derivation.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256Ctx {
  uint32_t state[8];
  uint64_t total_len = 0;
  uint8_t buffer[64];
  size_t buffer_len = 0;

  Sha256Ctx() {
    static constexpr uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                          0xa54ff53a, 0x510e527f, 0x9b05688c,
                                          0x1f83d9ab, 0x5be0cd19};
    std::memcpy(state, kInit, sizeof(state));
  }

  void process_block(const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(block[4 * i]) << 24) |
             (uint32_t(block[4 * i + 1]) << 16) |
             (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
      uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void update(const uint8_t* data, size_t len) {
    total_len += len;
    if (buffer_len > 0) {
      size_t take = std::min(len, sizeof(buffer) - buffer_len);
      std::memcpy(buffer + buffer_len, data, take);
      buffer_len += take;
      data += take;
      len -= take;
      if (buffer_len == sizeof(buffer)) {
        process_block(buffer);
        buffer_len = 0;
      }
    }
    while (len >= 64) {
      process_block(data);
      data += 64;
      len -= 64;
    }
    if (len > 0) {
      std::memcpy(buffer, data, len);
      buffer_len = len;
    }
  }

  void finish(uint8_t out[32]) {
    uint64_t bit_len = total_len * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (buffer_len != 56) {
      update(&zero, 1);
    }
    uint8_t len_bytes[8];
    for (int i = 0; i < 8; ++i) {
      len_bytes[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    }
    update(len_bytes, 8);
    for (int i = 0; i < 8; ++i) {
      out[4 * i] = static_cast<uint8_t>(state[i] >> 24);
      out[4 * i + 1] = static_cast<uint8_t>(state[i] >> 16);
      out[4 * i + 2] = static_cast<uint8_t>(state[i] >> 8);
      out[4 * i + 3] = static_cast<uint8_t>(state[i]);
    }
  }
};

}  // namespace

void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
  Sha256Ctx ctx;
  ctx.update(data, len);
  ctx.finish(out);
}

void hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data,
                 size_t data_len, uint8_t out[32]) {
  uint8_t key_block[64] = {0};
  if (key_len > 64) {
    sha256(key, key_len, key_block);
  } else {
    std::memcpy(key_block, key, key_len);
  }
  uint8_t ipad[64], opad[64];
  for (int i = 0; i < 64; ++i) {
    ipad[i] = key_block[i] ^ 0x36;
    opad[i] = key_block[i] ^ 0x5c;
  }
  uint8_t inner[32];
  Sha256Ctx inner_ctx;
  inner_ctx.update(ipad, 64);
  inner_ctx.update(data, data_len);
  inner_ctx.finish(inner);
  Sha256Ctx outer_ctx;
  outer_ctx.update(opad, 64);
  outer_ctx.update(inner, 32);
  outer_ctx.finish(out);
}

void hkdf_sha256_32(const uint8_t* ikm, size_t ikm_len, const uint8_t* info,
                    size_t info_len, uint8_t out[32]) {
  // Extract with salt = HashLen zero bytes (RFC 5869; salt=None in
  // cryptography's HKDF).
  uint8_t salt[32] = {0};
  uint8_t prk[32];
  hmac_sha256(salt, sizeof(salt), ikm, ikm_len, prk);
  // Expand with L = 32 = HashLen, so OKM = T(1) = HMAC(PRK, info || 0x01).
  std::vector<uint8_t> block(info, info + info_len);
  block.push_back(0x01);
  hmac_sha256(prk, sizeof(prk), block.data(), block.size(), out);
}

// ---------------------------------------------------------------------------
// X25519 (via vendored Monocypher).
// ---------------------------------------------------------------------------

void x25519_public_key(uint8_t public_key[32], const uint8_t private_key[32]) {
  crypto_x25519_public_key(public_key, private_key);
}

bool x25519_shared_secret(uint8_t shared[32], const uint8_t private_key[32],
                          const uint8_t peer_public[32]) {
  crypto_x25519(shared, private_key, peer_public);
  uint8_t acc = 0;
  for (int i = 0; i < 32; ++i) {
    acc |= shared[i];
  }
  return acc != 0;
}

// ---------------------------------------------------------------------------
// ChaCha20-Poly1305 IETF (RFC 8439), composed from Monocypher primitives.
// ---------------------------------------------------------------------------

std::vector<uint8_t> chacha20poly1305_ietf_encrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const std::vector<uint8_t>& aad, const std::vector<uint8_t>& plaintext) {
  // The Poly1305 one-time key is the first 32 bytes of the ChaCha20
  // keystream at block counter 0.
  uint8_t block0[64];
  crypto_chacha20_ietf(block0, nullptr, sizeof(block0), key, nonce, 0);

  std::vector<uint8_t> out(plaintext.size() + 16);
  crypto_chacha20_ietf(out.data(), plaintext.data(), plaintext.size(), key,
                       nonce, 1);

  // The MAC input is aad || pad16 || ciphertext || pad16 || le64(|aad|) ||
  // le64(|ct|).
  crypto_poly1305_ctx poly;
  crypto_poly1305_init(&poly, block0);
  static const uint8_t kZeros[16] = {0};
  crypto_poly1305_update(&poly, aad.data(), aad.size());
  crypto_poly1305_update(&poly, kZeros, (16 - aad.size() % 16) % 16);
  crypto_poly1305_update(&poly, out.data(), plaintext.size());
  crypto_poly1305_update(&poly, kZeros, (16 - plaintext.size() % 16) % 16);
  uint8_t lengths[16];
  uint64_t aad_len = aad.size();
  uint64_t ct_len = plaintext.size();
  for (int i = 0; i < 8; ++i) {
    lengths[i] = static_cast<uint8_t>(aad_len >> (8 * i));
    lengths[8 + i] = static_cast<uint8_t>(ct_len >> (8 * i));
  }
  crypto_poly1305_update(&poly, lengths, sizeof(lengths));
  crypto_poly1305_final(&poly, out.data() + plaintext.size());

  crypto_wipe(block0, sizeof(block0));
  return out;
}

// ---------------------------------------------------------------------------
// Base64 / key parsing.
// ---------------------------------------------------------------------------

std::string b64url_encode_nopad(const uint8_t* data, size_t len) {
  static const char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  std::string out;
  out.reserve((len + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= len; i += 3) {
    uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) |
                 uint32_t(data[i + 2]);
    out.push_back(kAlphabet[(v >> 18) & 63]);
    out.push_back(kAlphabet[(v >> 12) & 63]);
    out.push_back(kAlphabet[(v >> 6) & 63]);
    out.push_back(kAlphabet[v & 63]);
  }
  if (i + 1 == len) {
    uint32_t v = uint32_t(data[i]) << 16;
    out.push_back(kAlphabet[(v >> 18) & 63]);
    out.push_back(kAlphabet[(v >> 12) & 63]);
  } else if (i + 2 == len) {
    uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
    out.push_back(kAlphabet[(v >> 18) & 63]);
    out.push_back(kAlphabet[(v >> 12) & 63]);
    out.push_back(kAlphabet[(v >> 6) & 63]);
  }
  return out;
}

namespace {

int b64_value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

}  // namespace

std::vector<uint8_t> b64_decode_flexible(const std::string& value) {
  std::vector<uint8_t> out;
  uint32_t acc = 0;
  int bits = 0;
  for (char c : value) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
      continue;
    }
    int v = b64_value(c);
    if (v < 0) {
      throw std::runtime_error("invalid base64 character in input");
    }
    acc = (acc << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>(acc >> bits));
    }
  }
  return out;
}

std::vector<uint8_t> load_x25519_public_key(const std::string& value) {
  // Trim whitespace, mirroring value.strip() in the verifier.
  size_t begin = value.find_first_not_of(" \t\r\n");
  size_t end = value.find_last_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    throw std::runtime_error(
        "DPHN proof public key must decode to 32 raw bytes");
  }
  std::string trimmed = value.substr(begin, end - begin + 1);

  if (trimmed.rfind("-----BEGIN", 0) == 0) {
    // PEM SubjectPublicKeyInfo. The DER prefix for an X25519 public key is
    // SEQUENCE(42) { SEQUENCE(5) { OID 1.3.101.110 } BIT STRING(33) }.
    size_t header_end = trimmed.find('\n');
    size_t footer = trimmed.find("-----END");
    if (header_end == std::string::npos || footer == std::string::npos) {
      throw std::runtime_error("DPHN proof public key PEM is malformed");
    }
    std::string body = trimmed.substr(header_end + 1, footer - header_end - 1);
    std::vector<uint8_t> der = b64_decode_flexible(body);
    static const uint8_t kX25519SpkiPrefix[12] = {
        0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x03, 0x21, 0x00};
    if (der.size() != 44 || std::memcmp(der.data(), kX25519SpkiPrefix,
                                        sizeof(kX25519SpkiPrefix)) != 0) {
      throw std::runtime_error(
          "DPHN proof public key must be an X25519 public key");
    }
    return std::vector<uint8_t>(der.begin() + 12, der.end());
  }

  std::vector<uint8_t> raw = b64_decode_flexible(trimmed);
  if (raw.size() != 32) {
    throw std::runtime_error(
        "DPHN proof public key must decode to 32 raw bytes");
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Randomness.
// ---------------------------------------------------------------------------

void random_bytes(uint8_t* out, size_t len) {
#if defined(__linux__)
  size_t filled = 0;
  while (filled < len) {
    ssize_t n = getrandom(out + filled, len - filled, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;  // fall through to /dev/urandom
    }
    filled += static_cast<size_t>(n);
  }
  if (filled == len) {
    return;
  }
#endif
#if defined(__linux__) || defined(__APPLE__)
  FILE* f = std::fopen("/dev/urandom", "rb");
  if (f != nullptr) {
    size_t n = std::fread(out, 1, len, f);
    std::fclose(f);
    if (n == len) {
      return;
    }
  }
#endif
  throw std::runtime_error("no secure random source available");
}

}  // namespace dphn_crypto
