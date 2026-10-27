// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cstdint>

void b12x_w4a8_moe_fused_ctrl_init(torch::stable::Tensor& ctrl);

void b12x_w4a8_moe_fused_sm120(
    const torch::stable::Tensor& a_values,
    const torch::stable::Tensor& a_scales, const torch::stable::Tensor& w13,
    const torch::stable::Tensor& w13_scales, torch::stable::Tensor& projections,
    torch::stable::Tensor& intermediate, const torch::stable::Tensor& w2,
    const torch::stable::Tensor& w2_scales,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& alpha1,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& alpha2,
    const torch::stable::Tensor& down_scale,
    torch::stable::Tensor& route_output, torch::stable::Tensor& ctrl,
    int64_t num_pairs, int64_t topk, int64_t rows_capacity, int64_t n,
    double swiglu_limit);
