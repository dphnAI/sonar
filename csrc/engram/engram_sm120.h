// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <optional>

void sm120_engram_pack(const torch::stable::Tensor& hidden,
                       const torch::stable::Tensor& local_projection,
                       const torch::stable::Tensor& q_weight,
                       const torch::stable::Tensor& k_weight,
                       torch::stable::Tensor& packet, int64_t rank,
                       int64_t num_valid_tokens);

void sm120_engram_finish(const torch::stable::Tensor& hidden,
                         const torch::stable::Tensor& gathered_packets,
                         const std::optional<torch::stable::Tensor>& token_mask,
                         torch::stable::Tensor& output,
                         int64_t num_valid_tokens, double eps,
                         double clamp_value);
