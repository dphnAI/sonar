#pragma once

#include <cstddef>
#include "cutlass/numeric_types.h"

namespace aphrodite::cutlass_w4a8_utils {

bool unified_encode_int4b(cutlass::int4b_t const* in, cutlass::int4b_t* out,
                          size_t num_int4_elems);

}  // namespace aphrodite::cutlass_w4a8_utils