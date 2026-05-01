#pragma once

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

#ifndef USE_ROCM
  #include "quantization/exl3/exllamav3_ext/hgemm.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/exl3_gemm.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/exl3_moe.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/hadamard.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/reconstruct.cuh"
#endif

inline void aphrodite_exl3_gemm(const at::Tensor& A, const at::Tensor& B,
                                at::Tensor& C,
                                const std::optional<at::Tensor>& suh,
                                const std::optional<at::Tensor>& A_had,
                                const std::optional<at::Tensor>& svh,
                                int64_t force_shape_idx, bool mcg, bool mul1,
                                int64_t force_num_sms) {
#ifndef USE_ROCM
  exl3_gemm(A, B, C, suh, A_had, svh, static_cast<int>(force_shape_idx), mcg,
            mul1, static_cast<int>(force_num_sms));
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

inline void aphrodite_exl3_mgemm(const at::Tensor& A, const at::Tensor& B,
                                 at::Tensor& C, const at::Tensor& suh,
                                 const at::Tensor& A_had, const at::Tensor& svh,
                                 const std::optional<at::Tensor>& indices,
                                 const std::optional<at::Tensor>& weights,
                                 int64_t k, int64_t force_shape_idx, bool mcg,
                                 bool mul1, int64_t min_index,
                                 int64_t max_index, int64_t force_num_sms) {
#ifndef USE_ROCM
  exl3_mgemm(A, B, C, suh, A_had, svh, indices, weights, static_cast<int>(k),
             static_cast<int>(force_shape_idx), static_cast<uint32_t>(mcg),
             static_cast<uint32_t>(mul1), static_cast<int>(min_index),
             static_cast<int>(max_index), static_cast<int>(force_num_sms));
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

inline void aphrodite_exl3_reconstruct(at::Tensor unpacked, at::Tensor packed,
                                       int64_t k, bool mcg, bool mul1) {
#ifndef USE_ROCM
  reconstruct(unpacked, packed, static_cast<int>(k), mcg, mul1);
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

inline void aphrodite_exl3_had_r_128(
    const at::Tensor& input, const at::Tensor& output,
    const std::optional<at::Tensor>& pre_scale,
    const std::optional<at::Tensor>& post_scale, double scale) {
#ifndef USE_ROCM
  had_r_128(input, output, pre_scale, post_scale, static_cast<float>(scale));
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

inline void aphrodite_exl3_hgemm(at::Tensor a, at::Tensor b, at::Tensor c) {
#ifndef USE_ROCM
  hgemm(a, b, c);
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

inline void aphrodite_exl3_moe(
    const at::Tensor& hidden_state, const at::Tensor& output_state,
    const at::Tensor& expert_count, const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted, const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u, const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u, int64_t act_function, int64_t K_gate,
    int64_t K_up, int64_t K_down, const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh, const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    bool gate_mcg, bool gate_mul1, bool up_mcg, bool up_mul1, bool down_mcg,
    bool down_mul1, double act_limit) {
#ifndef USE_ROCM
  exl3_moe(hidden_state, output_state, expert_count, token_sorted,
           weight_sorted, temp_state_g, temp_state_u, temp_intermediate_g,
           temp_intermediate_u, static_cast<int>(act_function),
           static_cast<int>(K_gate), static_cast<int>(K_up),
           static_cast<int>(K_down), gate_ptrs_trellis, gate_ptrs_suh,
           gate_ptrs_svh, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
           down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh, gate_mcg, gate_mul1,
           up_mcg, up_mul1, down_mcg, down_mul1, static_cast<float>(act_limit));
#else
  TORCH_CHECK(false, "EXL3 is not supported on ROCm");
#endif
}

torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void merge_attn_states(
    torch::Tensor& output, std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
    const std::optional<int64_t> prefill_tokens_with_context,
    const std::optional<torch::Tensor>& output_scale = std::nullopt);
#ifndef USE_ROCM
void convert_vertical_slash_indexes(
    torch::Tensor& block_count,      // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,     // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,         // [BATCH, ]
    torch::Tensor kv_seqlens,        // [BATCH, ]
    torch::Tensor vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int64_t context_size, int64_t block_size_M, int64_t block_size_N,
    bool causal);

void convert_vertical_slash_indexes_mergehead(
    torch::Tensor& block_count,            // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,           // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,               // [BATCH, ]
    torch::Tensor kv_seqlens,              // [BATCH, ]
    torch::Tensor vertical_indexes,        // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,           // [BATCH, N_HEADS, NNZ_S]
    torch::Tensor vertical_indices_count,  // [N_HEADS, ]
    torch::Tensor slash_indices_count, int64_t context_size,
    int64_t block_size_M, int64_t block_size_N, bool causal);
#endif

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void fused_qk_norm_rope(torch::Tensor& qkv, int64_t num_heads_q,
                        int64_t num_heads_k, int64_t num_heads_v,
                        int64_t head_dim, double eps, torch::Tensor& q_weight,
                        torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
                        bool is_neox, torch::Tensor& position_ids,
                        int64_t forced_token_heads_per_warp);

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor& q, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, double eps, int64_t cache_block_size);

void apply_repetition_penalties_(torch::Tensor& logits,
                                 const torch::Tensor& prompt_mask,
                                 const torch::Tensor& output_mask,
                                 const torch::Tensor& repetition_penalties);

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK);

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK);

void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                     torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);

void rms_norm_static_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                               torch::Tensor& weight, torch::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::Tensor& out,
                                         torch::Tensor& input,
                                         torch::Tensor& residual,
                                         torch::Tensor& weight,
                                         torch::Tensor& scale, double epsilon);

void rms_norm_dynamic_per_token_quant(torch::Tensor& out,
                                      torch::Tensor const& input,
                                      torch::Tensor const& weight,
                                      torch::Tensor& scales,
                                      double const epsilon,
                                      std::optional<torch::Tensor> scale_ub,
                                      std::optional<torch::Tensor> residual);

void rms_norm_per_block_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& weight,
                              torch::Tensor& scales, double const epsilon,
                              std::optional<torch::Tensor> scale_ub,
                              std::optional<torch::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);

void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit);

void silu_mul(torch::Tensor& out, torch::Tensor const& gate,
              torch::Tensor const& up);

void make_gate_up_indices(torch::Tensor& out, torch::Tensor const& indices,
                          int64_t offset);

void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,   // (E, T, 2*H)
    const at::Tensor& counts,  // (E)
    at::Tensor& y_q,           // (E, T, H) [OUT]
    at::Tensor& y_s,           // (E, T, H//group_size) [OUT]
    bool use_ue8m0);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

#ifndef USE_ROCM
torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    aphrodite::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits, bool is_a_8bit);

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits,
                                bool is_a_8bit);

torch::Tensor marlin_int4_fp8_preprocess(
    torch::Tensor& qweight, std::optional<torch::Tensor> qzeros_or_none,
    bool inplace);

#endif

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, bool use_v2_format, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape = std::nullopt);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    std::optional<torch::Tensor> const& scale_ub);

void selective_scan_fwd(
    const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A,
    const torch::Tensor& B, const torch::Tensor& C,
    const std::optional<torch::Tensor>& D_,
    const std::optional<torch::Tensor>& z_,
    const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& ssm_states, int64_t null_block_id, int64_t block_size,
    const std::optional<torch::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    const std::optional<torch::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::Tensor>& last_chunk_indices);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind);

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

torch::Tensor hadacore_transform(torch::Tensor& x, bool inplace);

#ifdef USE_ROCM
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);
int64_t qr_max_size();
#endif

#ifndef USE_ROCM
void dsv3_fused_a_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                       torch::Tensor const& mat_b);
#endif

#ifndef USE_ROCM
torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps);
std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps);
#endif
