// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "canonical_cbor.h"

#include <algorithm>

namespace dphn_cbor {

namespace {

void encode_head(uint8_t major, uint64_t value, std::vector<uint8_t>& out) {
  uint8_t mt = static_cast<uint8_t>(major << 5);
  if (value < 24) {
    out.push_back(static_cast<uint8_t>(mt | value));
  } else if (value <= 0xff) {
    out.push_back(mt | 24);
    out.push_back(static_cast<uint8_t>(value));
  } else if (value <= 0xffff) {
    out.push_back(mt | 25);
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
  } else if (value <= 0xffffffffull) {
    out.push_back(mt | 26);
    for (int shift = 24; shift >= 0; shift -= 8) {
      out.push_back(static_cast<uint8_t>(value >> shift));
    }
  } else {
    out.push_back(mt | 27);
    for (int shift = 56; shift >= 0; shift -= 8) {
      out.push_back(static_cast<uint8_t>(value >> shift));
    }
  }
}

void encode_int(int64_t v, std::vector<uint8_t>& out) {
  if (v >= 0) {
    encode_head(0, static_cast<uint64_t>(v), out);
  } else {
    // Negative integer n encodes as major type 1 with argument -1 - n.
    encode_head(1, static_cast<uint64_t>(-(v + 1)), out);
  }
}

void encode_text(const std::string& s, std::vector<uint8_t>& out) {
  encode_head(3, s.size(), out);
  out.insert(out.end(), s.begin(), s.end());
}

}  // namespace

Value Value::Int(int64_t v) {
  Value value;
  value.kind_ = Kind::kInt;
  value.int_ = v;
  return value;
}

Value Value::Str(std::string v) {
  Value value;
  value.kind_ = Kind::kStr;
  value.str_ = std::move(v);
  return value;
}

Value Value::Bytes(std::vector<uint8_t> v) {
  Value value;
  value.kind_ = Kind::kBytes;
  value.bytes_ = std::move(v);
  return value;
}

Value Value::Arr(Array v) {
  Value value;
  value.kind_ = Kind::kArr;
  value.arr_ = std::move(v);
  return value;
}

Value Value::MakeMap(Map v) {
  Value value;
  value.kind_ = Kind::kMap;
  value.map_ = std::move(v);
  return value;
}

void Value::encode(std::vector<uint8_t>& out) const {
  switch (kind_) {
    case Kind::kInt:
      encode_int(int_, out);
      break;
    case Kind::kStr:
      encode_text(str_, out);
      break;
    case Kind::kBytes:
      encode_head(2, bytes_.size(), out);
      out.insert(out.end(), bytes_.begin(), bytes_.end());
      break;
    case Kind::kArr:
      encode_head(4, arr_.size(), out);
      for (const Value& item : arr_) {
        item.encode(out);
      }
      break;
    case Kind::kMap: {
      // Canonical map key ordering (cbor2 canonical=True) sorts by
      // (length of encoded key, encoded key bytes).
      struct Encoded {
        std::vector<uint8_t> key;
        const Value* value;
      };
      std::vector<Encoded> encoded;
      encoded.reserve(map_.size());
      for (const auto& [key, value] : map_) {
        Encoded e;
        encode_text(key, e.key);
        e.value = &value;
        encoded.push_back(std::move(e));
      }
      std::sort(encoded.begin(), encoded.end(),
                [](const Encoded& a, const Encoded& b) {
                  if (a.key.size() != b.key.size()) {
                    return a.key.size() < b.key.size();
                  }
                  return a.key < b.key;
                });
      encode_head(5, encoded.size(), out);
      for (const Encoded& e : encoded) {
        out.insert(out.end(), e.key.begin(), e.key.end());
        e.value->encode(out);
      }
      break;
    }
  }
}

std::vector<uint8_t> encode(const Value& value) {
  std::vector<uint8_t> out;
  value.encode(out);
  return out;
}

}  // namespace dphn_cbor
