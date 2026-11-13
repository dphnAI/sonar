// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused softplus-sqrt topk launcher: gating_output=float, topk_indices=int32.

#include "topk_softplus_sqrt_kernels.cuh"

namespace sonar::moe {

SONAR_TOPK_SOFTPLUS_SQRT_INSTANTIATE(int, float, int64_t)
SONAR_TOPK_SOFTPLUS_SQRT_INSTANTIATE(int, float, int)

}  // namespace sonar::moe
