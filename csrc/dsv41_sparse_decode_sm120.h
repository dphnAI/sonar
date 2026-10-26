// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <optional>

void dsv41_sparse_decode_sm120(
    const torch::stable::Tensor& q, torch::stable::Tensor& out,
    const torch::stable::Tensor& swa_cache, int64_t swa_page_size,
    const torch::stable::Tensor& swa_indices,
    const torch::stable::Tensor& swa_lengths,
    const std::optional<torch::stable::Tensor>& indexed_cache,
    int64_t indexed_page_size,
    const std::optional<torch::stable::Tensor>& indexed_indices,
    const std::optional<torch::stable::Tensor>& indexed_lengths,
    const std::optional<torch::stable::Tensor>& indexed_page_table,
    const torch::stable::Tensor& attn_sink, double sm_scale,
    torch::stable::Tensor& partial, torch::stable::Tensor& stats);
