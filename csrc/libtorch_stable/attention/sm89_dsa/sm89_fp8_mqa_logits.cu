// Prefill/ragged fp8 MQA indexer logits on sm89 (Ada), over a contiguous gathered kv buffer.
//
// logits[m, n] = sum_h w[m, h] * relu((q[m, h, :] . kv[n, :]) * kv_scale[n]),  n in [ks_m, ke_m)
//
// Inputs: q [M, H, 128] e4m3 (H = 32 or 64 indexer heads, templated); kv [N, 128] e4m3
// (contiguous post-gather buffer, what cp_gather_indexer_k_quant_cache produces);
// kv_scales [N] fp32; weights [M, H] fp32; per-row windows cu_seqlen_ks/ke [M] i32
// (clamped to [0, N]).
// Output logits [M, N] fp32 — written ONLY inside [ks, ke); the caller owns the fill value
// outside the window (pass a zeroed buffer for fill=0). No O(M*N) pre-fill traffic.
//
// Grid (x=M, y=ceil(N/CTA_KEYS)), x fastest: all concurrently-resident CTAs work the same
// CTA_KEYS-wide K band, so each K page is pulled from DRAM once and served to every query row
// out of L2 (K-tile reuse from the axis ordering alone, no explicit rasterization).
// CTA = 128 thr / 4 warps = one query token's H heads x CTA_KEYS keys, pipelined in 64-key
// pages through a cp.async ring (3 pages in flight, single barrier per page: pages are issued
// right after the barrier into the stage retired two iterations ago, and the page-store is
// deferred one iteration through double-buffered partials so it overlaps the next page's mma
// work). B fragments via ldmatrix.x4 (LSU-issue relief). mma m16n8k32 e4m3 FP32-accumulate +
// the epilogue math scale -> relu -> per-head weight -> fixed-order reduction.
// Deterministic (no float atomics, fixed reduction order); CUDA-graph-safe (static grid per
// (M, N) shape; windows are read on device; empty-window CTAs exit after the two window loads).

// clang-format off

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

#include <cstdint>

#ifndef USE_ROCM

#include <cuda.h>
#include <cuda_fp8.h>

#define DEVINL __device__ __forceinline__

namespace sm89_dsa {

constexpr int PAGE = 64;  // keys per pipeline stage
constexpr int HEAD_DIM = 128;
constexpr int PAGE_KEY_BYTES = PAGE * HEAD_DIM;    // 8192
constexpr int PAGE_BYTES = PAGE * (HEAD_DIM + 4);  // 8448 (keys + fp32 scales)
constexpr int STAGES = 4;
// 32 pages = 2048 keys per CTA: amortizes the per-CTA fixed cost (q A-fragment loads,
// pipeline fill) 4x better than 8 pages (iso-clock 9.94 -> 8.96 ms on the (2048, 65536)
// probe; 64 pages gave only 2% more and halves grid parallelism for small N).
constexpr int PAGES_PER_CTA = 32;
constexpr int CTA_KEYS = PAGES_PER_CTA * PAGE;  // 2048

DEVINL void cp_async_16(void* smem_dst, const void* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}
DEVINL void cp_async_4(void* smem_dst, const void* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(gmem_src));
}
DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEVINL void mma_fp8(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 4x 8-row x 16B matrices from smem; lane l supplies the row address for matrix l/8, row l%8.
// Result reg r[i] of lane l = bytes [4*(l%4), 4*(l%4)+4) of row l/4 of matrix i — exactly the
// m16n8k32 e4m3 B fragment when the matrix rows are 8 consecutive key rows and the 16B column
// chunk is one half of a 32-deep k-step.
DEVINL void ldmatrix_x4(uint32_t (&r)[4], const void* smem_addr) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// Swizzled smem byte offset of 16B chunk `cc` (0..7) within key row `r` (0..63).
DEVINL int swizzle_chunk(int r, int cc) { return r * HEAD_DIM + ((cc ^ (r & 7)) << 4); }

// Block: 128 threads / 4 warps working query token blockIdx.x. A page's 8 n-tiles (8 keys
// each) are split across NUM_HEADS/16 head groups x 4/(NUM_HEADS/16) key sub-bands: warp w
// owns head group w % HGROUPS and n-tiles [kg*NT, kg*NT + NT), kg = w / HGROUPS.
// NUM_HEADS=64 is the original layout: warp w -> heads [16w, 16w+16), all 8 n-tiles.
template <int NUM_HEADS>
__global__ void __launch_bounds__(128, 2)
mqa_logits_kernel(const uint8_t* __restrict__ q,          // [M, NUM_HEADS, 128] e4m3
                  const uint8_t* __restrict__ kv,         // [N, 128] e4m3
                  const float* __restrict__ kv_scales,    // [N]
                  const float* __restrict__ weights,      // [M, NUM_HEADS]
                  const int32_t* __restrict__ cu_ks,      // [M]
                  const int32_t* __restrict__ cu_ke,      // [M]
                  float* __restrict__ logits,             // [M, N]
                  int N) {
  static_assert(NUM_HEADS == 32 || NUM_HEADS == 64, "unsupported indexer head count");
  constexpr int HGROUPS = NUM_HEADS / 16;  // head groups of 16 (one m16 mma tile each)
  constexpr int NT = 8 * HGROUPS / 4;      // 8-key n-tiles per warp
  extern __shared__ uint8_t smem[];
  float* partials = reinterpret_cast<float*>(smem + STAGES * PAGE_BYTES);  // [2][HGROUPS][64]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // HGROUPS == 4 folds to (warp, 0): the compiler cannot bound `warp` on its own, and the
  // fold keeps the 64-head build's codegen identical to the pre-templated kernel.
  const int hg = HGROUPS == 4 ? warp : warp % HGROUPS;
  const int kg = HGROUPS == 4 ? 0 : warp / HGROUPS;
  const int h_base = hg * 16;
  const int group = lane >> 2;  // 0..7
  const int quad = lane & 3;    // 0..3

  const int m = blockIdx.x;
  const int c0 = blockIdx.y * CTA_KEYS;

  const int win_s = max((int)cu_ks[m], 0);
  const int win_e = min((int)cu_ke[m], N);
  const int lo = max(win_s, c0);
  const int hi = min(win_e, c0 + CTA_KEYS);
  if (lo >= hi) return;  // window misses this CTA's key band (or empty window)

  const float* w_row = weights + (int64_t)m * NUM_HEADS;
  const float w0 = w_row[h_base + group];
  const float w1 = w_row[h_base + group + 8];

  // resident A fragments: q[m, h_base:h_base+16, :], 4 k-steps of m16n8k32
  uint32_t afr[4][4];
  {
    const uint8_t* qb = q + (((int64_t)m * NUM_HEADS) + h_base) * HEAD_DIM;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int kb = ks * 32 + quad * 4;
      afr[ks][0] = *reinterpret_cast<const uint32_t*>(qb + group * HEAD_DIM + kb);
      afr[ks][1] = *reinterpret_cast<const uint32_t*>(qb + (group + 8) * HEAD_DIM + kb);
      afr[ks][2] = *reinterpret_cast<const uint32_t*>(qb + group * HEAD_DIM + kb + 16);
      afr[ks][3] = *reinterpret_cast<const uint32_t*>(qb + (group + 8) * HEAD_DIM + kb + 16);
    }
  }

  // page p covers global keys [c0 + 64p, c0 + 64p + 64)
  const int p_lo = (lo - c0) / PAGE;
  const int p_hi = (hi - c0 + PAGE - 1) / PAGE;  // exclusive

  auto issue_page = [&](int p, int stage) {
    const int gk = c0 + p * PAGE;  // first global key of the page
    uint8_t* dst = smem + stage * PAGE_BYTES;
    const uint8_t* src_keys = kv + (int64_t)gk * HEAD_DIM;
    if (gk + PAGE <= N) {  // interior page: no bounds checks
      for (int c = tid; c < PAGE_BYTES / 16; c += 128) {
        if (c < PAGE_KEY_BYTES / 16) {
          cp_async_16(dst + swizzle_chunk(c >> 3, c & 7), src_keys + c * 16);
        } else {
          const int sc = (c - PAGE_KEY_BYTES / 16) * 4;  // scale index within page
          cp_async_16(dst + PAGE_KEY_BYTES + sc * 4, kv_scales + gk + sc);
        }
      }
    } else {  // tail page: rows/scales past N are skipped (their columns are never stored)
      for (int c = tid; c < PAGE_BYTES / 16; c += 128) {
        if (c < PAGE_KEY_BYTES / 16) {
          const int r = c >> 3;
          if (gk + r < N) cp_async_16(dst + swizzle_chunk(r, c & 7), src_keys + c * 16);
        } else {
          const int sc = (c - PAGE_KEY_BYTES / 16) * 4;
          if (gk + sc + 4 <= N) {
            cp_async_16(dst + PAGE_KEY_BYTES + sc * 4, kv_scales + gk + sc);
          } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
              if (gk + sc + j < N)
                cp_async_4(dst + PAGE_KEY_BYTES + (sc + j) * 4, kv_scales + gk + sc + j);
          }
        }
      }
    }
    cp_async_commit();
  };

  // threads 0..63 each sum one column's head-group partials (fixed order) and store.
  // `buf` selects the double-buffered partials of the page being drained.
  const int64_t row = (int64_t)m * N;
  auto store_page = [&](int it, int buf) {
    if (tid < 64) {
      const float* pt = partials + buf * (HGROUPS * 64);
      float v;
      if constexpr (HGROUPS == 4) {
        v = pt[tid] + pt[64 + tid] + pt[128 + tid] + pt[192 + tid];
      } else {
        v = pt[tid] + pt[64 + tid];
      }
      const int gcol = c0 + (p_lo + it) * PAGE + tid;
      if (gcol >= win_s && gcol < win_e) logits[row + gcol] = v;
    }
  };

  // Single-barrier pipeline. Commit-group invariant: the prologue makes STAGES groups
  // (LOOKAHEAD real + padding empties) and every iteration adds exactly one group (real or
  // empty), so before the wait of iteration `it` there are STAGES + it groups and
  // cp_async_wait<LOOKAHEAD - 1> guarantees page `it`'s group has completed. Pages are
  // issued at the TOP of an iteration, right after the barrier: the stage they overwrite,
  // (it + LOOKAHEAD) % STAGES, was consumed in iteration it + LOOKAHEAD - STAGES < it, and
  // the barrier proves every warp has left it. The page-store is deferred one iteration
  // (double-buffered partials), so the same barrier also orders partials write -> read and
  // the store overlaps the next page's mma work.
  constexpr int LOOKAHEAD = STAGES - 1;  // 3 pages in flight
  const int span = p_hi - p_lo;
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    if (s < LOOKAHEAD && s < span) issue_page(p_lo + s, s);
    else cp_async_commit();
  }

  const int jrow = lane & 7;  // ldmatrix: row within this lane's matrix
  const int mi = lane >> 3;   // ldmatrix: which of the 4 matrices this lane addresses

  for (int it = 0; it < span; ++it) {
    const int stage = it % STAGES;
    cp_async_wait<LOOKAHEAD - 1>();
    __syncthreads();

    {
      const int nxt = it + LOOKAHEAD;
      if (nxt < span) issue_page(p_lo + nxt, nxt % STAGES);
      else cp_async_commit();  // keep the commit-count invariant through the tail
    }
    if (it > 0) store_page(it - 1, (it - 1) & 1);  // overlaps with this page's mmas

    const uint8_t* keys = smem + stage * PAGE_BYTES;
    const float* scales = reinterpret_cast<const float*>(keys + PAGE_KEY_BYTES);

    // B fragments via ldmatrix.x4: 2 LSU instructions per 8-key n-tile instead of 8 LDS.32
    // (v1 was LSU-issue-bound at 70% per ncu). ldmatrix A covers k-steps 0..1 (16B chunks
    // 0..3 of the key rows), B covers 2..3. n-tiles are processed in pairs so consecutive
    // mmas hit independent accumulators (breaks the 4-deep accumulator dependency chain).
    float c[NT][4] = {};
#pragma unroll
    for (int ntp = 0; ntp < NT / 4; ++ntp) {
      const int nb = ntp * 4;  // 4 n-tiles in flight: dependency distance 4 between mmas
      uint32_t bA[4][4], bB[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int keyr = (kg * NT + nb + j) * 8 + jrow;
        ldmatrix_x4(bA[j], keys + swizzle_chunk(keyr, mi));
        ldmatrix_x4(bB[j], keys + swizzle_chunk(keyr, 4 + mi));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_fp8(afr[0], bA[j][0], bA[j][1], c[nb + j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_fp8(afr[1], bA[j][2], bA[j][3], c[nb + j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_fp8(afr[2], bB[j][0], bB[j][1], c[nb + j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_fp8(afr[3], bB[j][2], bB[j][3], c[nb + j]);
    }
    float out_cols[NT][2];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n0 = (kg * NT + t) * 8 + quad * 2;
      const float s0 = scales[n0], s1 = scales[n0 + 1];
      out_cols[t][0] = fmaxf(c[t][0] * s0, 0.f) * w0 + fmaxf(c[t][2] * s0, 0.f) * w1;
      out_cols[t][1] = fmaxf(c[t][1] * s1, 0.f) * w0 + fmaxf(c[t][3] * s1, 0.f) * w1;
    }

    // sum this warp's 16 heads: each column is held by the 8 lanes with equal `quad`
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int d = 4; d < 32; d <<= 1) {
        out_cols[t][0] += __shfl_xor_sync(0xffffffff, out_cols[t][0], d);
        out_cols[t][1] += __shfl_xor_sync(0xffffffff, out_cols[t][1], d);
      }
    }
    if (group == 0) {  // lanes 0..3 hold cols quad*2, quad*2+1 of every n-tile
      float* pw = partials + (it & 1) * (HGROUPS * 64) + hg * 64;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int nt = kg * NT + t;
        pw[nt * 8 + quad * 2] = out_cols[t][0];
        pw[nt * 8 + quad * 2 + 1] = out_cols[t][1];
      }
    }
  }
  __syncthreads();
  store_page(span - 1, (span - 1) & 1);
  cp_async_wait<0>();
}

}  // namespace sm89_dsa

#endif  // USE_ROCM

void sm89_fp8_mqa_logits(const torch::stable::Tensor& q,
                         const torch::stable::Tensor& kv,
                         const torch::stable::Tensor& kv_scales,
                         const torch::stable::Tensor& weights,
                         const torch::stable::Tensor& cu_seqlen_ks,
                         const torch::stable::Tensor& cu_seqlen_ke,
                         torch::stable::Tensor& logits) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() &&
                      weights.is_cuda() && cu_seqlen_ks.is_cuda() &&
                      cu_seqlen_ke.is_cuda() && logits.is_cuda(),
                  "all tensors must be CUDA");
  STD_TORCH_CHECK(q.dim() == 3 && (q.size(1) == 32 || q.size(1) == 64) && q.size(2) == 128,
                  "q must be [M, {32|64}, 128]");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(1) == q.size(1),
                  "weights must be [M, num_heads]");
  STD_TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 128, "kv must be [N, 128]");
  STD_TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() &&
                      weights.is_contiguous() && logits.is_contiguous(),
                  "q, kv, kv_scales, weights, and logits must be contiguous");
  STD_TORCH_CHECK(cu_seqlen_ks.is_contiguous() && cu_seqlen_ke.is_contiguous(),
                  "cu_seqlen_ks and cu_seqlen_ke must be contiguous");
  const int64_t M = q.size(0);
  const int num_heads = q.size(1);
  const int64_t N = kv.size(0);
  STD_TORCH_CHECK(kv_scales.numel() == N && logits.size(0) == M && logits.size(1) == N,
                  "kv_scales/logits shape mismatch");
  STD_TORCH_CHECK(cu_seqlen_ks.numel() == M && cu_seqlen_ke.numel() == M,
                  "cu_seqlen_ks/ke must have M entries");
  if (M == 0 || N == 0) return;
  const int64_t grid_y = (N + sm89_dsa::CTA_KEYS - 1) / sm89_dsa::CTA_KEYS;
  STD_TORCH_CHECK(grid_y <= 65535, "N too large for grid.y");

  const int smem = sm89_dsa::STAGES * sm89_dsa::PAGE_BYTES +
                   2 * (num_heads / 16) * 64 * sizeof(float);
  const cudaStream_t stream = get_current_cuda_stream();
  dim3 grid((unsigned)M, (unsigned)grid_y);
  auto launch = [&](auto* kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, 128, smem, stream>>>(
        q.const_data_ptr<uint8_t>(), kv.const_data_ptr<uint8_t>(),
        kv_scales.const_data_ptr<float>(), weights.const_data_ptr<float>(),
        cu_seqlen_ks.const_data_ptr<int32_t>(), cu_seqlen_ke.const_data_ptr<int32_t>(),
        logits.mutable_data_ptr<float>(), (int)N);
  };
  if (num_heads == 32) {
    launch(&sm89_dsa::mqa_logits_kernel<32>);
  } else {
    launch(&sm89_dsa::mqa_logits_kernel<64>);
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  STD_TORCH_CHECK(false, "sm89_fp8_mqa_logits is not supported on ROCm");
#endif
}

#ifdef APHRODITE_ENABLE_SM89_DSA
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("sm89_fp8_mqa_logits", TORCH_BOX(&sm89_fp8_mqa_logits));
}
#endif

// clang-format on
