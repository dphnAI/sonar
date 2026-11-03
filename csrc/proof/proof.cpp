// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Sampled runtime-weight receipts (format v2). The request context and output
// token IDs select byte windows of the live weights; their digests are sealed
// to the verifier's X25519 key. A receipt shows access to the sampled bytes,
// not that those weights produced the output.

#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "canonical_cbor.h"
#include "core/registration.h"
#include "crypto.h"
#include "blake3.h"

namespace {

constexpr int64_t kProofVersion = 2;
// Receipts are taken once per finished choice; the field is fixed in the
// format.
constexpr int64_t kSchedulerStep = 0;

using dphn_cbor::Value;

// --- Helpers ---------------------------------------------------------------

std::vector<uint8_t> blake3_digest(const uint8_t* data, size_t len) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, data, len);
  std::vector<uint8_t> out(BLAKE3_OUT_LEN);
  blake3_hasher_finalize(&hasher, out.data(), out.size());
  return out;
}

std::string hex_encode(const std::vector<uint8_t>& data) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(data.size() * 2);
  for (uint8_t b : data) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xf]);
  }
  return out;
}

std::string blake3_hexdigest(const std::vector<uint8_t>& data) {
  return hex_encode(blake3_digest(data.data(), data.size()));
}

uint64_t u64_from_seed(const std::vector<uint8_t>& seed, uint64_t counter) {
  std::vector<uint8_t> input(seed);
  input.push_back(static_cast<uint8_t>(counter & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 8) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 16) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 24) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 32) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 40) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 48) & 0xff));
  input.push_back(static_cast<uint8_t>((counter >> 56) & 0xff));
  auto d = blake3_digest(input.data(), input.size());
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(d[i]) << (i * 8);
  return v;
}

// --- Tensor registry -----------------------------------------------------

struct TensorEntry {
  std::string name;
  at::Tensor tensor;
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t num_bytes;
};

struct Registry {
  std::vector<TensorEntry> entries;
  std::string registry_hash;
};

static std::string dtype_python_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
    case at::kDouble:
    case at::kInt:
    case at::kLong:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
    case at::kComplexHalf:
    case at::kComplexFloat:
    case at::kComplexDouble:
    case at::ScalarType::UInt16:
    case at::ScalarType::UInt32:
    case at::ScalarType::UInt64:
      break;
    default:
      // Packed/sub-byte and other unsupported formats must not be interpreted
      // as numel() independently addressable elements.
      TORCH_CHECK(c10::isFloat8Type(t), "Unsupported proof tensor dtype: ", t);
  }
  const auto name = c10::getDtypeNames(t).first;
  TORCH_CHECK(!name.empty(), "Unsupported proof tensor dtype: ", t);
  return "torch." + std::string(name);
}

Registry build_registry(at::TensorList tensors,
                        const std::vector<std::string>& names) {
  TORCH_CHECK(tensors.size() == names.size());
  Registry registry;
  std::unordered_set<std::string> seen;
  std::unordered_set<std::string> seen_names;
  for (size_t i = 0; i < tensors.size(); ++i) {
    TORCH_CHECK(!names[i].empty(), "Proof tensor names must be nonempty");
    TORCH_CHECK(seen_names.insert(names[i]).second,
                "Duplicate proof tensor name: ", names[i]);
    const at::Tensor& raw = tensors[i];
    if (!raw.defined()) continue;
    at::Tensor tensor = raw.detach();
    at::DeviceType device = tensor.device().type();
    if (device != at::DeviceType::CPU && device != at::DeviceType::CUDA &&
        device != at::DeviceType::MPS)
      continue;
    if (tensor.layout() != at::kStrided || tensor.numel() == 0 ||
        !tensor.is_contiguous())
      continue;
    std::string key =
        std::to_string(static_cast<int>(device)) + ":" +
        std::to_string(reinterpret_cast<uintptr_t>(tensor.const_data_ptr()));
    if (!seen.insert(std::move(key)).second) continue;
    TensorEntry entry;
    entry.name = names[i];
    entry.tensor = tensor;
    entry.dtype = dtype_python_name(tensor.scalar_type());
    entry.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
    entry.num_bytes =
        tensor.numel() * static_cast<int64_t>(tensor.element_size());
    registry.entries.push_back(std::move(entry));
  }
  std::stable_sort(registry.entries.begin(), registry.entries.end(),
                   [](const TensorEntry& a, const TensorEntry& b) {
                     return a.name < b.name;
                   });
  Value::Array metadata;
  metadata.reserve(registry.entries.size());
  for (const TensorEntry& entry : registry.entries) {
    Value::Array shape;
    for (int64_t dim : entry.shape) shape.push_back(Value::Int(dim));
    metadata.push_back(Value::MakeMap({
        {"name", Value::Str(entry.name)},
        {"dtype", Value::Str(entry.dtype)},
        {"shape", Value::Arr(std::move(shape))},
        {"num_bytes", Value::Int(entry.num_bytes)},
    }));
  }
  registry.registry_hash =
      blake3_hexdigest(dphn_cbor::encode(Value::Arr(std::move(metadata))));
  return registry;
}

// --- Contiguous byte reads -----------------------------------------------

std::vector<uint8_t> read_tensor_bytes(const at::Tensor& tensor, int64_t offset,
                                       int64_t length) {
  TORCH_CHECK(tensor.defined() && tensor.layout() == at::kStrided &&
                  tensor.is_contiguous(),
              "Proof byte reads require a defined, strided, contiguous tensor");
  TORCH_CHECK(offset >= 0 && length >= 0,
              "Proof byte offset and length must be nonnegative");
  const uint64_t total_bytes = tensor.nbytes();
  TORCH_CHECK(static_cast<uint64_t>(offset) <= total_bytes &&
                  static_cast<uint64_t>(length) <=
                      total_bytes - static_cast<uint64_t>(offset),
              "Proof byte window is outside the tensor");
  std::vector<uint8_t> out(static_cast<size_t>(length));
  if (length == 0) return out;
  if (tensor.device().is_cpu()) {
    std::memcpy(out.data(),
                static_cast<const char*>(tensor.const_data_ptr()) + offset,
                static_cast<size_t>(length));
    return out;
  }
  at::Tensor host = tensor.flatten()
                        .view(at::kByte)
                        .slice(0, offset, offset + length)
                        .to(at::kCPU)
                        .contiguous();
  std::memcpy(out.data(), host.const_data_ptr(), static_cast<size_t>(length));
  return out;
}

// --- The v2 op ------------------------------------------------------------

std::string dphn_prove_v2(at::TensorList tensors,
                          std::vector<std::string> names, std::string worker_id,
                          std::string request_id, std::string gateway_nonce,
                          std::string model_manifest_id,
                          std::vector<int64_t> output_token_ids,
                          std::string public_key, std::string key_id,
                          int64_t num_raw_terms, int64_t raw_term_bytes) {
  for (const std::string* field :
       {&worker_id, &request_id, &gateway_nonce, &model_manifest_id}) {
    TORCH_CHECK(!field->empty() && field->find('\0') == std::string::npos,
                "Proof context strings must be nonempty and contain no NUL");
  }
  for (int64_t id : output_token_ids) {
    TORCH_CHECK(id >= 0, "Proof output token IDs must be nonnegative");
  }
  TORCH_CHECK(num_raw_terms >= 1, "v2 requires at least one weight term");
  TORCH_CHECK(raw_term_bytes >= 16);

  Registry registry = build_registry(tensors, names);
  TORCH_CHECK(!registry.entries.empty());

  // generated_token_hash
  Value::Array token_values;
  for (int64_t id : output_token_ids) token_values.push_back(Value::Int(id));
  std::string generated_token_hash =
      blake3_hexdigest(dphn_cbor::encode(Value::Arr(std::move(token_values))));

  // seed = blake3(cbor({nonce, request_id, model_manifest_id, scheduler_step,
  // generated_token_hash}))
  std::vector<uint8_t> seed_material = dphn_cbor::encode(Value::MakeMap({
      {"nonce", Value::Str(gateway_nonce)},
      {"request_id", Value::Str(request_id)},
      {"model_manifest_id", Value::Str(model_manifest_id)},
      {"scheduler_step", Value::Int(kSchedulerStep)},
      {"generated_token_hash", Value::Str(generated_token_hash)},
  }));
  std::vector<uint8_t> seed =
      blake3_digest(seed_material.data(), seed_material.size());

  // Weight terms: all raw_window_hash (deterministic)
  Value::Array terms;
  Value::Array sketches;
  uint64_t counter = 0;
  const auto& entries = registry.entries;
  for (int64_t t = 0; t < num_raw_terms; ++t) {
    const TensorEntry& entry =
        entries[u64_from_seed(seed, counter) % entries.size()];
    counter += 1;
    int64_t length = std::min(raw_term_bytes, entry.num_bytes);
    int64_t max_offset = std::max<int64_t>(0, entry.num_bytes - length);
    int64_t offset = static_cast<int64_t>(
        u64_from_seed(seed, counter) % static_cast<uint64_t>(max_offset + 1));
    counter += 1;
    Value::Map term = {
        {"op", Value::Str("raw_window_hash")},
        {"tensor", Value::Str(entry.name)},
        {"offset", Value::Int(offset)},
        {"length", Value::Int(length)},
    };
    std::vector<uint8_t> raw = read_tensor_bytes(entry.tensor, offset, length);
    Value::Map sketch = term;
    sketch.emplace_back("digest", Value::Str(blake3_hexdigest(raw)));
    terms.push_back(Value::MakeMap(std::move(term)));
    sketches.push_back(Value::MakeMap(std::move(sketch)));
  }
  std::string weight_proof =
      blake3_hexdigest(dphn_cbor::encode(Value::Arr(std::move(sketches))));

  // Payload
  Value::Map payload_map = {
      {"version", Value::Int(kProofVersion)},
      {"worker_id", Value::Str(worker_id)},
      {"request_id", Value::Str(request_id)},
      {"gateway_nonce", Value::Str(gateway_nonce)},
      {"model_manifest_id", Value::Str(model_manifest_id)},
      {"scheduler_step", Value::Int(kSchedulerStep)},
      {"tensor_registry_hash", Value::Str(registry.registry_hash)},
      {"generated_token_hash", Value::Str(generated_token_hash)},
      {"terms", Value::Arr(std::move(terms))},
      {"weight_proof", Value::Str(weight_proof)},
  };

  // Seal the payload to the verifier key (X25519 + HKDF + ChaCha20-Poly1305).
  std::vector<uint8_t> payload_bytes =
      dphn_cbor::encode(Value::MakeMap(std::move(payload_map)));

  uint8_t ephemeral_private[32];
  uint8_t ephemeral_public[32];
  dphn_crypto::random_bytes(ephemeral_private, 32);
  dphn_crypto::x25519_public_key(ephemeral_public, ephemeral_private);

  std::vector<uint8_t> peer_public =
      dphn_crypto::load_x25519_public_key(public_key);
  uint8_t shared[32];
  TORCH_CHECK(dphn_crypto::x25519_shared_secret(shared, ephemeral_private,
                                                peer_public.data()),
              "DPHN proof: X25519 shared secret is all zeros");

  std::vector<uint8_t> aad;
  {
    auto w = worker_id.begin(), we = worker_id.end();
    aad.insert(aad.end(), w, we);
    aad.push_back(0);
    auto n = gateway_nonce.begin(), ne = gateway_nonce.end();
    aad.insert(aad.end(), n, ne);
    aad.push_back(0);
    auto m = model_manifest_id.begin(), me = model_manifest_id.end();
    aad.insert(aad.end(), m, me);
  }

  uint8_t key[32];
  // HKDF info must match Python: b"dphn-proof-v2" + aad
  std::vector<uint8_t> hkdf_info;
  static const char kInfoPrefix[] = "dphn-proof-v2";
  hkdf_info.insert(hkdf_info.end(), kInfoPrefix,
                   kInfoPrefix + sizeof(kInfoPrefix) - 1);
  hkdf_info.insert(hkdf_info.end(), aad.begin(), aad.end());
  dphn_crypto::hkdf_sha256_32(shared, 32, hkdf_info.data(), hkdf_info.size(),
                              key);

  uint8_t nonce[12];
  dphn_crypto::random_bytes(nonce, 12);
  std::vector<uint8_t> ciphertext = dphn_crypto::chacha20poly1305_ietf_encrypt(
      key, nonce, aad, payload_bytes);

  Value envelope = Value::MakeMap({
      {"v", Value::Int(kProofVersion)},
      {"kid", Value::Str(key_id)},
      {"epk", Value::Bytes(std::vector<uint8_t>(ephemeral_public,
                                                ephemeral_public + 32))},
      {"n", Value::Bytes(std::vector<uint8_t>(nonce, nonce + 12))},
      {"ct", Value::Bytes(std::move(ciphertext))},
  });
  std::vector<uint8_t> envelope_bytes = dphn_cbor::encode(envelope);
  return dphn_crypto::b64url_encode_nopad(envelope_bytes.data(),
                                          envelope_bytes.size());
}

}  // namespace

TORCH_LIBRARY(_proof_C, m) {
  m.def(
      "prove(Tensor[] tensors, str[] names, str worker_id, str request_id, "
      "str gateway_nonce, str model_manifest_id, int[] output_token_ids, "
      "str public_key, str key_id, int num_raw_terms, int raw_term_bytes) -> "
      "str",
      &dphn_prove_v2);
}

REGISTER_EXTENSION(_proof_C)
