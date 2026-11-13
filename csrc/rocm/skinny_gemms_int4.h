/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Declaration of the int4 skinny GEMM, one instantiation per N_in (1..5). The
 * API unit only dispatches on N_in and needs this header; the kernels stay in
 * skinny_gemms_int4.cuh, which the per-N_in units instantiate through
 * SONAR_SKINNY_INT4_INSTANTIATE below.
 *
 * The parameter list is spelled once here and reused by the declaration, the
 * `extern template` lines and the instantiations, so they cannot drift apart
 * into different overloads.
 */

#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>

#define SONAR_SKINNY_INT4_SIG                                                 \
  const at::Tensor &in_a, const at::Tensor &in_b, const at::Tensor &in_scale, \
      const std::optional<at::Tensor>&in_zero_points,                         \
      const std::optional<at::Tensor>&in_bias, const int64_t CuCount,         \
      const int64_t group_size

// One instantiation per supported N_in; EXT is `extern` to name an
// instantiation that another unit provides, and empty in the unit that
// provides it.
#define SONAR_SKINNY_INT4_DECLARE(EXT, N) \
  EXT template torch::Tensor wvSplitK_int4_g_impl<N>(SONAR_SKINNY_INT4_SIG)

#define SONAR_SKINNY_INT4_EXTERN(N) SONAR_SKINNY_INT4_DECLARE(extern, N);
#define SONAR_SKINNY_INT4_INSTANTIATE(N) SONAR_SKINNY_INT4_DECLARE(, N);

template <int N>
torch::Tensor wvSplitK_int4_g_impl(SONAR_SKINNY_INT4_SIG);

SONAR_SKINNY_INT4_EXTERN(1)
SONAR_SKINNY_INT4_EXTERN(2)
SONAR_SKINNY_INT4_EXTERN(3)
SONAR_SKINNY_INT4_EXTERN(4)
SONAR_SKINNY_INT4_EXTERN(5)
