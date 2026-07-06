#pragma once

#include "cutlass/integer_subbyte.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed = false>
struct aphrodite_biased_integer_subbyte : public integer_subbyte<Bits, Signed> {
  using Base = integer_subbyte<Bits, Signed>;

  using Storage = typename Base::Storage;
  using xint_t = typename Base::xint_t;

  using Base::bits_mask_;
  using Base::sign_mask_;
  using Base::storage;

  //
  // Methods
  //

  /// No operation
  aphrodite_biased_integer_subbyte() = default;

  /// Conversion from integer type
  CUTLASS_HOST_DEVICE explicit aphrodite_biased_integer_subbyte(int value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit aphrodite_biased_integer_subbyte(unsigned value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit aphrodite_biased_integer_subbyte(double value)
      : Base(value) {}
};
///////////////////////////////////////////////////////////////////////////////////////////////////

// "GPTQ" types, i.e. symmetric quantization
using aphrodite_uint4b8_t = aphrodite_biased_integer_subbyte<4, 8>;  // u4b8
using aphrodite_uint8b128_t =
    aphrodite_biased_integer_subbyte<8, 128>;  // u8b128

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed>
struct sizeof_bits<aphrodite_biased_integer_subbyte<Bits, Bias, Signed>> {
  static constexpr int value = Bits;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
