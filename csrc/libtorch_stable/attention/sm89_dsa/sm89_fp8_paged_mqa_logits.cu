// Paged fp8 MQA indexer logits for decode on sm89 (Ada), plus the (request,
// page) partition table that schedules it.
//
// logits[m, n] = sum_h w[m, h] * relu((q[m, h, :] . k[n, :]) * k_scale[n]),  n
// in [0, ke_m) h ranges over NUM_HEADS indexer heads (32 or 64, templated).
//
// An indexer K-cache page (block_size=64, 132 B/token, SoA per page) holds 64
// keys x 128 fp8 e4m3, key(token)-major, in bytes [0, 8192), then 64 fp32
// per-token scales in bytes [8192, 8448).
//
// P persistent CTAs; the metadata kernel builds a (P+1, 2) i32 partition table
// splitting the global page-count prefix sum, and CTA p owns work units
// [sched[p], sched[p+1]) in (request, page) order. Per page a 4-stage cp.async
// ring feeds fp8 mma m16n8k32 (fp32 accumulate), then an in-register epilogue
// (scale, relu, weight, fixed-order reduction) and 64 fp32 stores. No float
// atomics; deterministic.

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

constexpr int PAGE = 64;
constexpr int HEAD_DIM = 128;
constexpr int PAGE_KEY_BYTES = PAGE * HEAD_DIM;    // 8192
constexpr int PAGE_BYTES = PAGE * (HEAD_DIM + 4);  // 8448
constexpr int STAGES = 4;

DEVINL void cp_async_16(void* smem_dst, const void* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}
DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEVINL void mma_fp8(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Swizzled smem byte offset of 16B chunk `cc` (0..7) within key row `r` (0..63).
DEVINL int swizzle_chunk(int r, int cc) { return r * HEAD_DIM + ((cc ^ (r & 7)) << 4); }

// ------------------------------------------------------------- paged indexer logits
// Block of NEXT_N*128 threads, 4 warps per token. A page's 8 n-tiles (8 keys each) split
// across NUM_HEADS/16 head groups x 4/(NUM_HEADS/16) key sub-bands, so warp w%4 owns head
// group (w%4) % HGROUPS and n-tiles [kg*NT, kg*NT + NT) with kg = (w%4) / HGROUPS. With
// NUM_HEADS=64 that reduces to warp w%4 -> heads [16*(w%4), 16*(w%4)+16), all 8 n-tiles.
template <int NEXT_N, int NUM_HEADS>
__global__ void __launch_bounds__(NEXT_N * 128, 2)
paged_mqa_logits_kernel(const uint8_t* __restrict__ q,           // [B, NEXT_N, NUM_HEADS, 128] e4m3
                        const uint8_t* __restrict__ pool,        // [pool_pages, 8448]
                        const float* __restrict__ weights,       // [B*NEXT_N, NUM_HEADS]
                        const int32_t* __restrict__ seq_lens,    // [B, NEXT_N]
                        const int32_t* __restrict__ block_table, // [B, max_pages]
                        const int32_t* __restrict__ sched,       // [(P+1), 2]
                        float* __restrict__ logits,              // [B*NEXT_N, max_model_len]
                        int max_pages, int64_t max_model_len, int clean_logits) {
  static_assert(NUM_HEADS == 32 || NUM_HEADS == 64, "unsupported indexer head count");
  constexpr int HGROUPS = NUM_HEADS / 16;  // head groups of 16 (one m16 mma tile each)
  constexpr int NT = 8 * HGROUPS / 4;      // 8-key n-tiles per warp
  extern __shared__ uint8_t smem[];
  float* partials = reinterpret_cast<float*>(smem + STAGES * PAGE_BYTES);  // [NEXT_N][HGROUPS][64]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tok = warp >> 2;
  const int wsub = warp & 3;
  const int hg = wsub % HGROUPS;
  const int kg = wsub / HGROUPS;
  const int h_base = hg * 16;
  const int group = lane >> 2;
  const int quad = lane & 3;

  const int32_t req_s = sched[2 * blockIdx.x];
  const int32_t page_s = sched[2 * blockIdx.x + 1];
  const int32_t req_e = sched[2 * blockIdx.x + 2];
  const int32_t page_e = sched[2 * blockIdx.x + 3];
  if (req_s < 0 || req_e < 0) return;

  for (int req = req_s; req <= req_e; ++req) {
    int ke_max = 0;
    int32_t ke_tok[NEXT_N];
#pragma unroll
    for (int t = 0; t < NEXT_N; ++t) {
      ke_tok[t] = seq_lens[req * NEXT_N + t];
      ke_max = max(ke_max, (int)ke_tok[t]);
    }
    const int n_pages = (ke_max + PAGE - 1) / PAGE;
    const int p_lo = (req == req_s) ? page_s : 0;
    const int p_hi = (req == req_e) ? page_e : n_pages;  // exclusive
    if (p_lo >= p_hi) continue;

    const int m = req * NEXT_N + tok;
    const float* w_row = weights + (int64_t)m * NUM_HEADS;
    const float w0 = w_row[h_base + group];
    const float w1 = w_row[h_base + group + 8];
    const int ke = ke_tok[tok];

    // resident A fragments, q[m, h_base:h_base+16, :] as 4 k-steps
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

    const int32_t* bt = block_table + (int64_t)req * max_pages;
    auto issue_page = [&](int p, int stage) {
      const uint8_t* src = pool + (int64_t)bt[p] * PAGE_BYTES;
      uint8_t* dst = smem + stage * PAGE_BYTES;
      for (int c = tid; c < PAGE_BYTES / 16; c += NEXT_N * 128) {
        int off = (c < PAGE_KEY_BYTES / 16) ? swizzle_chunk(c >> 3, c & 7)
                                            : PAGE_KEY_BYTES + (c - PAGE_KEY_BYTES / 16) * 16;
        cp_async_16(dst + off, src + c * 16);
      }
      cp_async_commit();
    };

    // Exactly STAGES + it commit groups exist before iteration `it` (empty groups pad the
    // queue when span < STAGES or past the tail), so wait_group<STAGES-1> always guarantees
    // the group for iteration `it` has landed.
    const int span = p_hi - p_lo;
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      if (s < span) issue_page(p_lo + s, s);
      else cp_async_commit();
    }

    for (int it = 0; it < span; ++it) {
      const int p = p_lo + it;
      const int stage = it % STAGES;
      cp_async_wait<STAGES - 1>();
      __syncthreads();

      const uint8_t* keys = smem + stage * PAGE_BYTES;
      const float* scales = reinterpret_cast<const float*>(keys + PAGE_KEY_BYTES);

      float out_cols[NT][2];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const int nt = kg * NT + t;
        const int keyb = nt * 8 + group;  // B-frag column (key row in smem)
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          uint32_t b[2];
          const int kb0 = ks * 32 + quad * 4;
          const int kb1 = kb0 + 16;
          b[0] = *reinterpret_cast<const uint32_t*>(keys + swizzle_chunk(keyb, kb0 >> 4) + (kb0 & 15));
          b[1] = *reinterpret_cast<const uint32_t*>(keys + swizzle_chunk(keyb, kb1 >> 4) + (kb1 & 15));
          mma_fp8(afr[ks], b, c);
        }
        const int n0 = nt * 8 + quad * 2;
        const float s0 = scales[n0], s1 = scales[n0 + 1];
        out_cols[t][0] = fmaxf(c[0] * s0, 0.f) * w0 + fmaxf(c[2] * s0, 0.f) * w1;
        out_cols[t][1] = fmaxf(c[1] * s1, 0.f) * w0 + fmaxf(c[3] * s1, 0.f) * w1;
      }

      // sum this warp's 16 heads; each column is held by the 8 lanes with equal `quad`
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int d = 4; d < 32; d <<= 1) {
          out_cols[t][0] += __shfl_xor_sync(0xffffffff, out_cols[t][0], d);
          out_cols[t][1] += __shfl_xor_sync(0xffffffff, out_cols[t][1], d);
        }
      }
      if (group == 0) {  // lanes 0..3 hold cols quad*2, quad*2+1 of every n-tile
        float* pw = partials + (tok * HGROUPS + hg) * 64;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const int nt = kg * NT + t;
          pw[nt * 8 + quad * 2] = out_cols[t][0];
          pw[nt * 8 + quad * 2 + 1] = out_cols[t][1];
        }
      }
      __syncthreads();

      // one warp per token sums the head-group partials in fixed order and stores 64 fp32
      if (wsub == 0) {
        const float* pt = partials + tok * HGROUPS * 64;
        const int64_t row = (int64_t)m * max_model_len;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int n = lane + half * 32;
          float v = pt[n];
#pragma unroll
          for (int g = 1; g < HGROUPS; ++g) v += pt[g * 64 + n];
          const int64_t gcol = (int64_t)p * PAGE + n;
          if (gcol < ke) {
            logits[row + gcol] = v;
          } else if (clean_logits && it == span - 1 && gcol < max_model_len) {
            logits[row + gcol] = __int_as_float(0xff800000);  // -inf
          }
        }
      }
      __syncthreads();

      const int nxt = it + STAGES;
      if (nxt < span) issue_page(p_lo + nxt, stage);
      else cp_async_commit();  // keep the commit-count invariant through the tail
    }
    cp_async_wait<0>();
    __syncthreads();
  }
}

// ------------------------------------------------------------- partition metadata
// sched[p] = (req, page_offset) of global work unit floor(p*total/P); sched[P] = end of work.
__global__ void paged_meta_kernel(const int32_t* __restrict__ seq_lens,  // [B, next_n]
                                  int32_t* __restrict__ sched,           // [(P+1), 2]
                                  int B, int next_n, int P) {
  extern __shared__ int32_t s_prefix[];  // [B+1] inclusive page-count prefix
  const int tid = threadIdx.x;
  for (int b = tid; b < B; b += blockDim.x) {
    int ke = 0;
    for (int t = 0; t < next_n; ++t) ke = max(ke, (int)seq_lens[b * next_n + t]);
    s_prefix[b + 1] = (ke + PAGE - 1) / PAGE;
  }
  if (tid == 0) s_prefix[0] = 0;
  __syncthreads();
  if (tid == 0)
    for (int b = 1; b <= B; ++b) s_prefix[b] += s_prefix[b - 1];
  __syncthreads();

  const int total = s_prefix[B];
  for (int p = tid; p <= P; p += blockDim.x) {
    int32_t req, off;
    if (total == 0) {
      req = -1;
      off = 0;
    } else {
      const int64_t target = ((int64_t)p * total) / P;  // p==P -> total
      if (target >= total) {
        req = B - 1;
        off = total - s_prefix[B - 1];  // == n_pages(B-1), the exclusive end
      } else {
        int lo = 0, hi = B - 1;
        while (lo < hi) {  // first b with prefix[b+1] > target
          const int mid = (lo + hi) >> 1;
          if (s_prefix[mid + 1] > target) hi = mid; else lo = mid + 1;
        }
        req = lo;
        off = (int32_t)(target - s_prefix[lo]);
      }
    }
    sched[2 * p] = req;
    sched[2 * p + 1] = off;
  }
}

}  // namespace sm89_dsa

#endif  // USE_ROCM

void sm89_fp8_paged_mqa_logits(const torch::stable::Tensor& q,
                               const torch::stable::Tensor& pool,
                               const torch::stable::Tensor& weights,
                               const torch::stable::Tensor& seq_lens,
                               const torch::stable::Tensor& block_table,
                               const torch::stable::Tensor& sched,
                               torch::stable::Tensor& logits, bool clean_logits) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(q.is_cuda() && pool.is_cuda() && weights.is_cuda() &&
                      seq_lens.is_cuda() && block_table.is_cuda() &&
                      sched.is_cuda() && logits.is_cuda(),
                  "all tensors must be CUDA");
  STD_TORCH_CHECK(q.dim() == 4 && (q.size(2) == 32 || q.size(2) == 64) && q.size(3) == 128,
                  "q must be [B, next_n, {32|64}, 128]");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(1) == q.size(2),
                  "weights must be [B*next_n, num_heads]");
  STD_TORCH_CHECK(q.is_contiguous() && pool.is_contiguous() && logits.is_contiguous(),
                  "q, pool, and logits must be contiguous");
  STD_TORCH_CHECK(weights.is_contiguous() && seq_lens.is_contiguous() &&
                      block_table.is_contiguous() && sched.is_contiguous(),
                  "weights, seq_lens, block_table, and sched must be contiguous");
  STD_TORCH_CHECK(sched.dim() == 2 && sched.size(1) == 2 && sched.size(0) >= 2,
                  "sched must be [(P+1), 2]");
  const int next_n = q.size(1);
  const int num_heads = q.size(2);
  const int P = sched.size(0) - 1;
  const int max_pages = block_table.size(1);
  const int64_t max_model_len = logits.size(1);
  const cudaStream_t stream = get_current_cuda_stream();

  const int smem = sm89_dsa::STAGES * sm89_dsa::PAGE_BYTES +
                   next_n * (num_heads / 16) * 64 * sizeof(float);
  auto launch = [&](auto* kernel, int threads) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<P, threads, smem, stream>>>(
        q.const_data_ptr<uint8_t>(), pool.const_data_ptr<uint8_t>(),
        weights.const_data_ptr<float>(), seq_lens.const_data_ptr<int32_t>(),
        block_table.const_data_ptr<int32_t>(), sched.const_data_ptr<int32_t>(),
        logits.mutable_data_ptr<float>(), max_pages, max_model_len, (int)clean_logits);
  };
  STD_TORCH_CHECK(next_n == 1 || next_n == 2, "next_n must be 1 or 2");
  if (next_n == 1) {
    if (num_heads == 32) launch(&sm89_dsa::paged_mqa_logits_kernel<1, 32>, 128);
    else launch(&sm89_dsa::paged_mqa_logits_kernel<1, 64>, 128);
  } else {
    if (num_heads == 32) launch(&sm89_dsa::paged_mqa_logits_kernel<2, 32>, 256);
    else launch(&sm89_dsa::paged_mqa_logits_kernel<2, 64>, 256);
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  STD_TORCH_CHECK(false, "sm89_fp8_paged_mqa_logits is not supported on ROCm");
#endif
}

void sm89_paged_mqa_logits_metadata(const torch::stable::Tensor& seq_lens,
                                    torch::stable::Tensor& sched, int64_t next_n) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(seq_lens.is_cuda() && sched.is_cuda(), "all tensors must be CUDA");
  STD_TORCH_CHECK(seq_lens.is_contiguous() && sched.is_contiguous(),
                  "seq_lens and sched must be contiguous");
  STD_TORCH_CHECK(sched.dim() == 2 && sched.size(1) == 2 && sched.size(0) >= 2,
                  "sched must be [(P+1), 2]");
  STD_TORCH_CHECK(next_n > 0 && seq_lens.numel() % next_n == 0,
                  "seq_lens numel must be a multiple of next_n");
  const int P = sched.size(0) - 1;
  const int B = seq_lens.numel() / next_n;
  const cudaStream_t stream = get_current_cuda_stream();
  sm89_dsa::paged_meta_kernel<<<1, 1024, (B + 1) * sizeof(int32_t), stream>>>(
      seq_lens.const_data_ptr<int32_t>(), sched.mutable_data_ptr<int32_t>(), B, (int)next_n, P);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  STD_TORCH_CHECK(false, "sm89_paged_mqa_logits_metadata is not supported on ROCm");
#endif
}

#ifdef APHRODITE_ENABLE_SM89_DSA
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("sm89_fp8_paged_mqa_logits", TORCH_BOX(&sm89_fp8_paged_mqa_logits));
  ops.impl("sm89_paged_mqa_logits_metadata",
           TORCH_BOX(&sm89_paged_mqa_logits_metadata));
}
#endif

// clang-format on
