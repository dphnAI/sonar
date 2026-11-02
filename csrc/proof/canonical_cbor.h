// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Minimal canonical CBOR encoder matching Python's
// `cbor2.dumps(x, canonical=True)` for the value shapes the DPHN proof
// emits (text-keyed maps, arrays, integers, text and byte strings).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dphn_cbor {

class Value {
 public:
  using Array = std::vector<Value>;
  // Maps are restricted to text-string keys; that is all the proof schema
  // uses. Entries may be given in any order; encoding sorts them
  // canonically.
  using Map = std::vector<std::pair<std::string, Value>>;

  static Value Int(int64_t v);
  static Value Str(std::string v);
  static Value Bytes(std::vector<uint8_t> v);
  static Value Arr(Array v);
  static Value MakeMap(Map v);

  // Appends the canonical encoding of this value to `out`.
  void encode(std::vector<uint8_t>& out) const;

 private:
  enum class Kind { kInt, kStr, kBytes, kArr, kMap };
  Kind kind_ = Kind::kInt;
  int64_t int_ = 0;
  std::string str_;
  std::vector<uint8_t> bytes_;
  Array arr_;
  Map map_;
};

std::vector<uint8_t> encode(const Value& value);

}  // namespace dphn_cbor
