// Fused-MoE decode GEMM over per-expert Swordfish ABI v1 packed weights.
// One persistent Stream-K kernel covers every (expert block, column, unit)
// of the token-sorted problem that moe_align_block_size(block_size=16)
// prepares: each 16-token block is one m16 tile of exactly one expert, so
// the dense Stream-K decode mainloop applies with the activation rows and
// output rows indirected through sorted_token_ids and the weight and scale
// bases indexed by expert_ids. Work totals depend on the device-side padded
// token count, so per-warp ranges derive in-kernel from
// num_tokens_post_padded rather than on the host.

#include <algorithm>
#include <optional>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"
#include "swordfish_decode.cuh"

namespace swordfish {

namespace {

template <aphrodite::ScalarTypeId type_id, bool W8 = false, int M_TILES = 1>
__global__ void swordfish_decode_moe_kernel(
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ A,
    const int32_t* __restrict__ B,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ S,
    typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ C,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, int top_k, bool mul_topk_weights,
    int total_tokens, int K, int N, int group_size) {
  static_assert(M_TILES == 1 || M_TILES == 2, "moe m-tile fusion is 1 or 2");
  constexpr int kStagesT = M_TILES == 1 ? kStages : 4;
  constexpr int kUnitK = W8 ? 16 : 32;
  using Dtype = marlin::MarlinScalarType<type_id>;
  using scalar_t = typename Dtype::scalar_t;
  using scalar_t2 = typename Dtype::scalar_t2;
  using FragA = typename Dtype::FragA;
  using FragB = typename Dtype::FragB;
  using FragC = typename Dtype::FragC;

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group_id = lane / 4;
  const int tig = lane % 4;
  const int ldsm_row = lane & 15;
  const int ldsm_col = (lane >> 4) << 3;
  const int ia_row = lane >> 1;
  const int ia_c0 = (kUnitK / 2) * (lane & 1);

  const int num_units = K / kUnitK;
  const int nb_cnt = N / kBlockN;
  const int num_kb = K / kBlockK;
  const int ppg = group_size > 0 ? group_size / kUnitK : INT_MAX;
  const scalar_t2 zero2 = Dtype::num2num2(Dtype::float2num(0.0f));
  const uint64_t bpol = l2_evict_first_policy();
  constexpr int64_t kBlockW = W8 ? kBlockInt32_8 : kBlockInt32;

  const int num_blocks = *num_tokens_post_padded / (16 * M_TILES);
  const int64_t total = int64_t(num_blocks) * nb_cnt * num_units;
  const int total_warps = gridDim.x * kDecodeWarps;
  const int64_t per = (total + total_warps - 1) / total_warps;
  int64_t w = int64_t(blockIdx.x * kDecodeWarps + warp) * per;
  const int64_t w_end = w + per < total ? w + per : total;

  __shared__ int4 bstage[kDecodeWarps][kStagesT][2 * 32];
  __shared__ scalar_t astage[kDecodeWarps][kStagesT][M_TILES][16][kUnitK];

  FragC acc[M_TILES][4][2];
  scalar_t2 s_reg[4][2];

  while (w < w_end) {
    const int64_t cg = w / num_units;
    const int p_beg = int(w - cg * num_units);
    const int col = int(cg / num_blocks);
    const int blk = int(cg - int64_t(col) * num_blocks);
    const int p_end =
        int(int64_t(num_units) < p_beg + (w_end - w) ? int64_t(num_units)
                                                     : p_beg + (w_end - w));
    w += p_end - p_beg;

    const int eid = expert_ids[blk];
    if (eid < 0) continue;  // expert-parallel block owned by another rank

    const int m_base = 16 * M_TILES * blk;
    const int col_base = col * kBlockN;

    // Output rows for this lane's C fragments, through the token sort.
    int id0[M_TILES], id1[M_TILES];
    bool r0ok[M_TILES], r1ok[M_TILES];
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
      id0[t] = sorted_token_ids[m_base + 16 * t + group_id];
      id1[t] = sorted_token_ids[m_base + 16 * t + group_id + 8];
      r0ok[t] = id0[t] < total_tokens;
      r1ok[t] = id1[t] < total_tokens;
    }

#pragma unroll
    for (int t = 0; t < M_TILES; t++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[t][j][b][i] = 0.0f;

    const scalar_t* S_e = S + int64_t(eid) * (group_size > 0 ? K / group_size : 1) * N;
    auto fetch_row = [&](int g) -> scalar_t2 {
      return reinterpret_cast<const scalar_t2*>(S_e + int64_t(g) * N +
                                                col_base)[lane];
    };
    auto expand_row = [&](scalar_t2 mine) {
      const uint32_t sel = group_id & 1;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const scalar_t2 v = __shfl_sync(
              0xffffffffu, mine, 8 * j + 4 * b + (group_id >> 1));
          const uint32_t bits = reinterpret_cast<const uint32_t&>(v);
          const uint16_t h16 = sel ? uint16_t(bits >> 16) : uint16_t(bits);
          s_reg[j][b] =
              Dtype::num2num2(reinterpret_cast<const scalar_t&>(h16));
        }
      }
    };
    auto process_slice = [&](const FragA (&fa)[M_TILES], const marlin::I4& bqa,
                             const marlin::I4& bqb) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        FragB frag_b0, frag_b1;
        if constexpr (W8) {
          const marlin::I4& src = j < 2 ? bqa : bqb;
          marlin::dequant<scalar_t2, aphrodite::kU8B128.id(), false>(
              src.elems[(2 * j) & 3], reinterpret_cast<scalar_t2*>(&frag_b0));
          marlin::dequant<scalar_t2, aphrodite::kU8B128.id(), false>(
              src.elems[(2 * j + 1) & 3],
              reinterpret_cast<scalar_t2*>(&frag_b1));
        } else {
          const int b_quant_0 = bqa.elems[j];
          const int b_quant_1 = b_quant_0 >> 8;
          marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
              b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
          marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
              b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
        }
        frag_b0[0] = __hmul2(frag_b0[0], s_reg[j][0]);
        frag_b0[1] = __hmul2(frag_b0[1], s_reg[j][0]);
        frag_b1[0] = __hmul2(frag_b1[0], s_reg[j][1]);
        frag_b1[1] = __hmul2(frag_b1[1], s_reg[j][1]);
#pragma unroll
        for (int t = 0; t < M_TILES; t++) {
          marlin::mma<type_id, false>(fa[t], frag_b0, acc[t][j][0]);
          marlin::mma<type_id, false>(fa[t], frag_b1, acc[t][j][1]);
        }
      }
    };
    auto process_pair = [&](int aslot, const int4* buf) {
      FragA fa0[M_TILES], fa1[M_TILES];
#pragma unroll
      for (int t = 0; t < M_TILES; t++) {
        ldsm4<type_id>(fa0[t], &astage[warp][aslot][t][ldsm_row][ldsm_col]);
        if constexpr (!W8) {
          ldsm4<type_id>(fa1[t],
                         &astage[warp][aslot][t][ldsm_row][ldsm_col + 16]);
        }
      }
      if constexpr (W8) {
        const marlin::I4 bq0 =
            *reinterpret_cast<const marlin::I4*>(&buf[2 * lane]);
        const marlin::I4 bq1 =
            *reinterpret_cast<const marlin::I4*>(&buf[2 * lane + 1]);
        process_slice(fa0, bq0, bq1);
      } else {
        const marlin::I4 bq0 =
            *reinterpret_cast<const marlin::I4*>(&buf[lane]);
        const marlin::I4 bq1 =
            *reinterpret_cast<const marlin::I4*>(&buf[32 + lane]);
        process_slice(fa0, bq0, bq0);
        process_slice(fa1, bq1, bq1);
      }
    };

    const int32_t* ipair_ptr =
        B + (int64_t(eid) * nb_cnt + col) * num_kb * kBlockW +
        p_beg * kPairInt32 + (W8 ? 8 : 4) * lane;
    // Activation rows through the sort; padding rows stage row 0 disabled.
    bool a_okt[M_TILES];
    const scalar_t* ia_ptr[M_TILES];
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
      const int a_id = sorted_token_ids[m_base + 16 * t + ia_row];
      a_okt[t] = a_id < total_tokens;
      ia_ptr[t] = A + (a_okt[t] ? int64_t(a_id / top_k) : 0) * K +
                  kUnitK * p_beg + ia_c0;
    }
    int ipend = p_end - p_beg;

    auto issue_pair = [&](int slot) {
      if (ipend > 0) {
        ipend--;
        if constexpr (W8) {
          cp_async4_evict_first(&bstage[warp][slot][2 * lane], ipair_ptr,
                                bpol);
          cp_async4_evict_first(&bstage[warp][slot][2 * lane + 1],
                                ipair_ptr + 4, bpol);
        } else {
          cp_async4_evict_first(&bstage[warp][slot][lane], ipair_ptr, bpol);
          cp_async4_evict_first(&bstage[warp][slot][32 + lane],
                                ipair_ptr + kPairInt32 / 2, bpol);
        }
        ipair_ptr += kPairInt32;
#pragma unroll
        for (int t = 0; t < M_TILES; t++) {
          if (a_okt[t]) {
            marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0],
                              ia_ptr[t]);
            if constexpr (!W8) {
              marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0 + 8],
                                ia_ptr[t] + 8);
            }
          }
          ia_ptr[t] += kUnitK;
        }
      }
      marlin::cp_async_fence();
    };

    int g = 0;
    int left = INT_MAX;
    const int g_last =
        group_size > 0 && p_beg < p_end ? (kUnitK * (p_end - 1)) / group_size : 0;
    scalar_t2 s_next = zero2;
    if (p_beg < p_end) {
      if (group_size > 0) {
        g = p_beg / ppg;
        left = ppg - p_beg % ppg;
      }
      expand_row(fetch_row(g));
      if (g + 1 <= g_last) s_next = fetch_row(g + 1);
    }

#pragma unroll
    for (int st = 0; st < kStagesT - 1; st++) issue_pair(st);

    int slot = 0;
    int islot = kStagesT - 1;
    for (int p = p_beg; p < p_end; p++) {
      issue_pair(islot);
      if (++islot == kStagesT) islot = 0;
      marlin::cp_async_wait<kStagesT - 2>();
      if (left == 0) {
        ++g;
        expand_row(s_next);
        if (g + 1 <= g_last) s_next = fetch_row(g + 1);
        left = ppg;
      }
      process_pair(slot, bstage[warp][slot]);
      left--;
      if (++slot == kStagesT) slot = 0;
    }

    // Segment flush into the launcher-zeroed C, rows indirected through the
    // token sort, with the router weight folded in when requested.
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
      const float wt0 =
          mul_topk_weights && r0ok[t] ? topk_weights[id0[t]] : 1.0f;
      const float wt1 =
          mul_topk_weights && r1ok[t] ? topk_weights[id1[t]] : 1.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const int cc = col_base + 8 * (2 * j + b) + 2 * tig;
          const float4 v = *reinterpret_cast<float4*>(&acc[t][j][b]);
          if (r0ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(id0[t]) * N + cc),
                Dtype::nums2num2(Dtype::float2num(v.x * wt0),
                                 Dtype::float2num(v.y * wt0)));
          if (r1ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(id1[t]) * N + cc),
                Dtype::nums2num2(Dtype::float2num(v.z * wt1),
                                 Dtype::float2num(v.w * wt1)));
        }
      }
    }
  }
}

template <aphrodite::ScalarTypeId type_id, bool W8>
void launch_moe(const void* a, const int32_t* b, const void* s, void* c,
                const int32_t* sorted_ids, const int32_t* expert_ids,
                const int32_t* num_post_padded, const float* topk_w,
                int top_k, bool mul_topk, int total_tokens, int max_blocks,
                int block_size, int k, int n, int group_size,
                cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  constexpr int kUnitK = W8 ? 16 : 32;
  cudaMemsetAsync(c, 0, size_t(total_tokens) * n * sizeof(scalar_t), stream);
  // Host-side upper bound on work keeps tiny batches from launching idle
  // CTAs; the true total is device-side.
  const int64_t max_total =
      int64_t(max_blocks) * (n / kBlockN) * (k / kUnitK);
  const int64_t max_warps =
      max_total / (2 * kStages) > 0 ? max_total / (2 * kStages) : 1;
  const auto grid = [&](int ctas_per_sm) {
    int ctas = ctas_per_sm * sms;
    if (int64_t(ctas) * kDecodeWarps > max_warps) {
      ctas = int((max_warps + kDecodeWarps - 1) / kDecodeWarps);
    }
    return ctas < 1 ? 1 : ctas;
  };
  if (block_size == 32) {
    // 32-token blocks fuse two m16 tiles per weight fetch, amortizing the
    // dequant for batched shapes.
    static int ctas_per_sm2 = 0;
    if (ctas_per_sm2 == 0) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &ctas_per_sm2, swordfish_decode_moe_kernel<type_id, W8, 2>,
          kDecodeThreads, 0);
      if (ctas_per_sm2 <= 0) ctas_per_sm2 = 2;
    }
    swordfish_decode_moe_kernel<type_id, W8, 2>
        <<<grid(ctas_per_sm2), kDecodeThreads, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(a), b,
            reinterpret_cast<const scalar_t*>(s),
            reinterpret_cast<scalar_t*>(c), sorted_ids, expert_ids,
            num_post_padded, topk_w, top_k, mul_topk, total_tokens, k, n,
            group_size);
    return;
  }
  static int ctas_per_sm = 0;
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, swordfish_decode_moe_kernel<type_id, W8>,
        kDecodeThreads, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 2;
  }
  swordfish_decode_moe_kernel<type_id, W8>
      <<<grid(ctas_per_sm), kDecodeThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<scalar_t*>(c), sorted_ids, expert_ids,
          num_post_padded, topk_w, top_k, mul_topk, total_tokens, k, n,
          group_size);
}

}  // namespace

torch::stable::Tensor swordfish_moe_mm(
    torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
    torch::stable::Tensor& group_scales,
    torch::stable::Tensor& sorted_token_ids, torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& num_tokens_post_padded,
    std::optional<torch::stable::Tensor> const& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t num_bits, int64_t group_size, int64_t size_k, int64_t size_n) {
  STD_TORCH_CHECK(moe_block_size == 16 || moe_block_size == 32,
                  "swordfish_moe_mm supports block sizes 16 and 32");
  STD_TORCH_CHECK(num_bits == 4 || num_bits == 8,
                  "swordfish supports 4-bit and 8-bit weights");
  const bool w8 = num_bits == 8;
  STD_TORCH_CHECK(shape_ok(size_k, size_n), "swordfish ABI v1 requires K % ",
                  kBlockK, " == 0 and N % ", kBlockN, " == 0; got K=", size_k,
                  " N=", size_n);
  STD_TORCH_CHECK(a.dim() == 2 && a.size(1) == size_k && a.stride(1) == 1 &&
                      a.stride(0) == size_k,
                  "a must be contiguous [M, K]");
  const auto a_st = a.scalar_type();
  STD_TORCH_CHECK(a_st == torch::headeronly::ScalarType::Half ||
                      a_st == torch::headeronly::ScalarType::BFloat16,
                  "a must be fp16 or bf16");
  STD_TORCH_CHECK(group_scales.scalar_type() == a_st,
                  "group_scales dtype must match a");

  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  const int64_t words = w8 ? kBlockInt32_8 : kBlockInt32;
  STD_TORCH_CHECK(
      b_packed.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_packed.dim() == 4 && b_packed.size(1) == nb &&
          b_packed.size(2) == kb && b_packed.size(3) == words,
      "b_packed must be int32 [E, ", nb, ", ", kb, ", ", words, "]");
  const int64_t num_experts = b_packed.size(0);

  int64_t num_groups = 1;
  if (group_size != -1) {
    STD_TORCH_CHECK(group_size > 0 && size_k % group_size == 0 &&
                        group_size % (2 * kMarlinTileK) == 0,
                    "group_size must be -1 or a multiple of ",
                    2 * kMarlinTileK, " dividing K; got ", group_size);
    num_groups = size_k / group_size;
  }
  STD_TORCH_CHECK(group_scales.dim() == 3 &&
                      group_scales.size(0) == num_experts &&
                      group_scales.size(1) == num_groups &&
                      group_scales.size(2) == size_n,
                  "group_scales must be [", num_experts, ", ", num_groups,
                  ", ", size_n, "]");
  STD_TORCH_CHECK(
      sorted_token_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          expert_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          num_tokens_post_padded.scalar_type() ==
              torch::headeronly::ScalarType::Int,
      "alignment buffers must be int32");
  if (mul_topk_weights) {
    STD_TORCH_CHECK(topk_weights.has_value() &&
                        topk_weights->scalar_type() ==
                            torch::headeronly::ScalarType::Float,
                    "topk_weights must be fp32 when mul_topk_weights");
  }

  const int64_t total_tokens = a.size(0) * top_k;

  const int32_t device_index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor c = torch::stable::empty({total_tokens, size_n}, a_st,
                                                 std::nullopt, a.device());
  if (total_tokens == 0) return c;

  const auto* b_ptr =
      reinterpret_cast<const int32_t*>(b_packed.const_data_ptr());
  const auto* sid_ptr =
      reinterpret_cast<const int32_t*>(sorted_token_ids.const_data_ptr());
  const auto* eid_ptr =
      reinterpret_cast<const int32_t*>(expert_ids.const_data_ptr());
  const auto* npp_ptr = reinterpret_cast<const int32_t*>(
      num_tokens_post_padded.const_data_ptr());
  const float* tw_ptr =
      mul_topk_weights
          ? reinterpret_cast<const float*>(topk_weights->const_data_ptr())
          : nullptr;
  const int max_blocks =
      int((sorted_token_ids.numel() + moe_block_size - 1) / moe_block_size);

  if (a_st == torch::headeronly::ScalarType::Half) {
    if (w8) {
      launch_moe<aphrodite::kFloat16.id(), true>(
          a.const_data_ptr(), b_ptr, group_scales.const_data_ptr(),
          c.mutable_data_ptr(), sid_ptr, eid_ptr, npp_ptr, tw_ptr,
          int(top_k), mul_topk_weights, int(total_tokens), max_blocks,
          int(moe_block_size), int(size_k), int(size_n), int(group_size),
          stream);
    } else {
      launch_moe<aphrodite::kFloat16.id(), false>(
          a.const_data_ptr(), b_ptr, group_scales.const_data_ptr(),
          c.mutable_data_ptr(), sid_ptr, eid_ptr, npp_ptr, tw_ptr,
          int(top_k), mul_topk_weights, int(total_tokens), max_blocks,
          int(moe_block_size), int(size_k), int(size_n), int(group_size),
          stream);
    }
  } else {
    if (w8) {
      launch_moe<aphrodite::kBFloat16.id(), true>(
          a.const_data_ptr(), b_ptr, group_scales.const_data_ptr(),
          c.mutable_data_ptr(), sid_ptr, eid_ptr, npp_ptr, tw_ptr,
          int(top_k), mul_topk_weights, int(total_tokens), max_blocks,
          int(moe_block_size), int(size_k), int(size_n), int(group_size),
          stream);
    } else {
      launch_moe<aphrodite::kBFloat16.id(), false>(
          a.const_data_ptr(), b_ptr, group_scales.const_data_ptr(),
          c.mutable_data_ptr(), sid_ptr, eid_ptr, npp_ptr, tw_ptr,
          int(top_k), mul_topk_weights, int(total_tokens), max_blocks,
          int(moe_block_size), int(size_k), int(size_n), int(group_size),
          stream);
    }
  }

  return c;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_moe_mm", TORCH_BOX(&swordfish::swordfish_moe_mm));
}
