// Sparse MLA forward attention for decode on sm89 (Ada), gathering top-k KV
// directly from the fp8_ds_mla 656B pool by slot.
//
//   out[t, i] = softmax_k(q[t, i, :576] . kv[idx[t, k], :576] * sm_scale) @
//   kv[idx[t, k], :512]
//
// A 656B AoS pool row is 512 fp8 e4m3 "nope" bytes at [0, 512), 4 fp32
// per-128-tile scales at [512, 528), and 64 bf16 rope values at [528, 656).
//
// Grid (T, ceil(h/16)); 128 threads / 4 warps; BI=32 keys per iteration.
//   2-stage cp.async staging ring of raw 656B rows (by slot; -1 uses slot 0 +
//   score mask)
//   -> dequant nope fp8->bf16 (x tile scale) + rope passthrough into sKV [32 x
//   584]
//   -> QK bf16 mma m16n8k16 (warp w owns keys [8w, 8w+8); 36 k-steps over 576
//   dims)
//   -> masked scaled scores to smem; redundant per-lane online softmax (fixed
//   order)
//   -> PV bf16 mma (warp w owns out cols [128w, 128w+128); B-frags via
//   ldmatrix.x4.trans)
// l==0 rows (all -1 indices) write 0 output / -inf LSE. No atomics and fixed
// reduction orders, so bitwise run-to-run deterministic. Static shapes and -1
// handling keep it CUDA-graph-safe.
//
// Optional topk_lens[T] (requires front-compacted index rows, valid slots first
// then -1) clamps the key loop to ceil(len/BI) tiles. The skipped tiles are
// all-(-1) no-ops in the full loop (alpha == 1, sum == 0, P == 0), so the
// result is identical; DCP passes its per-rank valid counts here to skip the
// ~(1 - 1/dcp_world_size) non-owned candidates. The loop bound is read from
// device memory under a fixed grid, so it stays graph-safe.
//
// cp.async pipeline invariant. The prologue commits exactly STAGES groups
// (empty commits pad when n_iters < STAGES) and each loop iteration commits
// exactly one (empty past the tail), so cp_async_wait<STAGES-1> always
// guarantees iteration `it`'s group has landed.

// clang-format off

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

#include <cstdint>
#include <optional>

#ifndef USE_ROCM

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEVINL __device__ __forceinline__

namespace sm89_dsa {

constexpr int D = 576;   // 512 nope + 64 rope
constexpr int DN = 512;  // value dims (MLA absorb, V == K nope)
constexpr int ROW_BYTES = 656;
constexpr int ROW_CHUNKS = ROW_BYTES / 16;  // 41
constexpr int BI = 32;                      // keys per pipeline iteration
constexpr int STAGES = 2;
constexpr int HT = 16;                    // head tile (one m16 mma tile)
constexpr int SKV_STRIDE = D + 8;         // in halves; 1168B rows stay 16B-aligned, 4-word bank skew
constexpr int SQ_STRIDE = D + 8;
constexpr int SS_STRIDE = BI + 1;         // fp32; +1 pad -> per-row bank skew

constexpr int SMEM_STAGING = STAGES * BI * ROW_BYTES;  // 41984
constexpr int SMEM_SKV = BI * SKV_STRIDE * 2;          // 37376
constexpr int SMEM_SQ = HT * SQ_STRIDE * 2;            // 18688
constexpr int SMEM_SS = HT * SS_STRIDE * 4;            // 2112
constexpr int SMEM_TOTAL = SMEM_STAGING + SMEM_SKV + SMEM_SQ + SMEM_SS;  // 100160 <= 99KB opt-in

DEVINL void cp_async_16(void* smem_dst, const void* gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}
DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEVINL void mma_bf16(const uint32_t* a, const uint32_t* b, float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEVINL void ldmatrix_x4_trans(uint32_t* r, const void* p) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

DEVINL uint32_t bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

DEVINL float2 fp8x2_to_float2(uint16_t v) {
  __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}

// HAS_LENS == false compiles the len clamp away, leaving the full-topk loop
// untouched (the extra kernel param is never read).
template <bool HAS_LENS>
__global__ void __launch_bounds__(128, 1)
sparse_mla_fwd_kernel(const __nv_bfloat16* __restrict__ q,   // [T, h, 576]
                      const uint8_t* __restrict__ pool,      // [S, 656]
                      const int32_t* __restrict__ indices,   // [T, topk], -1 padded
                      __nv_bfloat16* __restrict__ out,       // [T, h, 512]
                      float* __restrict__ lse,                // [T, h]
                      const int32_t* __restrict__ topk_lens,  // [T]; read iff HAS_LENS
                      int h, int topk, float sm_scale) {
  extern __shared__ uint8_t smem[];
  uint8_t* staging = smem;
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_STAGING);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_STAGING + SMEM_SKV);
  float* sS = reinterpret_cast<float*>(smem + SMEM_STAGING + SMEM_SKV + SMEM_SQ);

  const int t = blockIdx.x;
  const int h_base = blockIdx.y * HT;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = lane >> 2;
  const int quad = lane & 3;
  const int r0 = group, r1 = group + 8;
  const float NEG_INF = __int_as_float(0xff800000);

  const int32_t* idx_row = indices + (int64_t)t * topk;
  auto key_valid = [&](int k) { return k < topk && idx_row[k] >= 0; };

  // ---- q tile -> sQ (rows past h zeroed; pad cols never read)
  for (int c = tid; c < HT * (D / 8); c += 128) {
    const int r = c / (D / 8), cc = c % (D / 8);
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (h_base + r < h)
      v = *reinterpret_cast<const uint4*>(q + ((int64_t)t * h + h_base + r) * D + cc * 8);
    *reinterpret_cast<uint4*>(sQ + r * SQ_STRIDE + cc * 8) = v;
  }

  auto issue_iter = [&](int it, int stage) {
    uint8_t* dst = staging + stage * (BI * ROW_BYTES);
    const int kbase = it * BI;
    for (int c = tid; c < BI * ROW_CHUNKS; c += 128) {
      const int j = c / ROW_CHUNKS, cc = c % ROW_CHUNKS;
      const int k = kbase + j;
      const int32_t slot = (k < topk) ? max(idx_row[k], 0) : 0;  // -1/tail -> slot 0 (masked)
      cp_async_16(dst + j * ROW_BYTES + cc * 16, pool + (int64_t)slot * ROW_BYTES + cc * 16);
    }
    cp_async_commit();
  };

  int n_iters = (topk + BI - 1) / BI;
  if constexpr (HAS_LENS) {
    // Rows are front-compacted, so every key at/past len is -1 and the
    // dropped tiles are exact no-ops. Clamp defensively; the per-key -1 mask
    // still guards every processed key, so a wrong len can only change how
    // many all-(-1) tiles get scanned, never corrupt the output.
    const int len = min(max(topk_lens[t], 0), topk);
    n_iters = min(n_iters, (len + BI - 1) / BI);
  }
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    if (s < n_iters) issue_iter(s, s);
    else cp_async_commit();
  }

  float m_st[2] = {NEG_INF, NEG_INF};  // online softmax state for rows r0, r1
  float l_st[2] = {0.f, 0.f};
  float acc[16][4];  // 16 n8-tiles covering out cols [128*warp, 128*warp+128)
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int it = 0; it < n_iters; ++it) {
    const int stage = it % STAGES;
    cp_async_wait<STAGES - 1>();
    __syncthreads();

    // ---- dequant staging[stage] -> sKV bf16 [32][584]; thread (row j=tid/4, tile p=tid%4)
    {
      const uint8_t* srow = staging + stage * (BI * ROW_BYTES) + (tid >> 2) * ROW_BYTES;
      const int p = tid & 3;
      const float scale = reinterpret_cast<const float*>(srow + DN)[p];
      __nv_bfloat16* drow = sKV + (tid >> 2) * SKV_STRIDE;
#pragma unroll
      for (int d = 0; d < 128; d += 8) {
        const uint2 raw = *reinterpret_cast<const uint2*>(srow + p * 128 + d);
        const uint16_t* b2 = reinterpret_cast<const uint16_t*>(&raw);
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = fp8x2_to_float2(b2[i]);
          o[i] = bf16x2(f.x * scale, f.y * scale);
        }
        *reinterpret_cast<uint4*>(drow + p * 128 + d) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      // rope passthrough (64 bf16 = 8 uint4); thread p copies 2
      const uint4* rs = reinterpret_cast<const uint4*>(srow + DN + 16);
      uint4* rd = reinterpret_cast<uint4*>(drow + DN);
      rd[2 * p] = rs[2 * p];
      rd[2 * p + 1] = rs[2 * p + 1];
    }
    __syncthreads();  // sKV ready; staging[stage] consumed -> safe to refill

    if (it + STAGES < n_iters) issue_iter(it + STAGES, stage);
    else cp_async_commit();  // keep the commit-count invariant through the tail

    // ---- QK mma; warp owns keys [8*warp, 8*warp+8), 36 k-steps over 576 dims
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    {
      const __nv_bfloat16* bk = sKV + (8 * warp + group) * SKV_STRIDE + 2 * quad;
      const __nv_bfloat16* qa0 = sQ + r0 * SQ_STRIDE + 2 * quad;
      const __nv_bfloat16* qa1 = sQ + r1 * SQ_STRIDE + 2 * quad;
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        uint32_t a[4], b[2];
        a[0] = *reinterpret_cast<const uint32_t*>(qa0 + ks * 16);
        a[1] = *reinterpret_cast<const uint32_t*>(qa1 + ks * 16);
        a[2] = *reinterpret_cast<const uint32_t*>(qa0 + ks * 16 + 8);
        a[3] = *reinterpret_cast<const uint32_t*>(qa1 + ks * 16 + 8);
        b[0] = *reinterpret_cast<const uint32_t*>(bk + ks * 16);
        b[1] = *reinterpret_cast<const uint32_t*>(bk + ks * 16 + 8);
        mma_bf16(a, b, c);
      }
    }

    // ---- masked, scaled scores -> sS
    const int kbase = it * BI;
    {
      const int col = 8 * warp + 2 * quad;
      const bool v0 = key_valid(kbase + col), v1 = key_valid(kbase + col + 1);
      sS[r0 * SS_STRIDE + col] = v0 ? c[0] * sm_scale : NEG_INF;
      sS[r0 * SS_STRIDE + col + 1] = v1 ? c[1] * sm_scale : NEG_INF;
      sS[r1 * SS_STRIDE + col] = v0 ? c[2] * sm_scale : NEG_INF;
      sS[r1 * SS_STRIDE + col + 1] = v1 ? c[3] * sm_scale : NEG_INF;
    }
    __syncthreads();

    // ---- online softmax; every lane redundantly processes its two rows in fixed order
    uint32_t pf[2][4];  // P bf16 A-frags for the 2 PV k-steps
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const float* srow = sS + (rr ? r1 : r0) * SS_STRIDE;
      float mx = m_st[rr];
#pragma unroll
      for (int k = 0; k < BI; ++k) mx = fmaxf(mx, srow[k]);
      float alpha = 1.f, sum = 0.f;
      if (mx > NEG_INF) {
        alpha = (m_st[rr] > NEG_INF) ? __expf(m_st[rr] - mx) : 0.f;
#pragma unroll
        for (int k = 0; k < BI; ++k) sum += __expf(srow[k] - mx);  // -inf entries -> 0
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int kb = 16 * s + 2 * quad;
          pf[s][rr] = bf16x2(__expf(srow[kb] - mx), __expf(srow[kb + 1] - mx));
          pf[s][rr + 2] = bf16x2(__expf(srow[kb + 8] - mx), __expf(srow[kb + 9] - mx));
        }
      } else {  // no valid key seen yet in this row
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          pf[s][rr] = 0u;
          pf[s][rr + 2] = 0u;
        }
      }
      l_st[rr] = l_st[rr] * alpha + sum;
      m_st[rr] = mx;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        acc[nt][2 * rr] *= alpha;
        acc[nt][2 * rr + 1] *= alpha;
      }
    }

    // ---- PV mma; warp owns out cols [128*warp, 128*warp+128), B-frags via ldmatrix.x4.trans
    const int mr = lane & 15;             // key row within 16-key k-step tile
    const int mc = (lane >> 4) * 8;       // col half of the 16-col chunk
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {  // 16-col chunks
        uint32_t r[4];
        const __nv_bfloat16* src = sKV + (16 * s + mr) * SKV_STRIDE + 128 * warp + 16 * j + mc;
        ldmatrix_x4_trans(r, src);
        mma_bf16(pf[s], r, acc[2 * j]);          // cols [.. +0, +8)
        mma_bf16(pf[s], r + 2, acc[2 * j + 1]);  // cols [.. +8, +16)
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue; normalize and store bf16 out + fp32 lse (l==0 writes 0 / -inf)
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int hh = h_base + (rr ? r1 : r0);
    if (hh >= h) continue;
    const float l = l_st[rr];
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    __nv_bfloat16* orow = out + ((int64_t)t * h + hh) * DN;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int col = 128 * warp + 8 * nt + 2 * quad;
      *reinterpret_cast<uint32_t*>(orow + col) =
          bf16x2(acc[nt][2 * rr] * inv, acc[nt][2 * rr + 1] * inv);
    }
    if (warp == 0 && quad == 0)
      lse[(int64_t)t * h + hh] = (l > 0.f) ? m_st[rr] + logf(l) : NEG_INF;
  }
}

}  // namespace sm89_dsa

#endif  // USE_ROCM

void sm89_sparse_mla_fwd(const torch::stable::Tensor& q,
                         const torch::stable::Tensor& pool,
                         const torch::stable::Tensor& indices,
                         torch::stable::Tensor& out, torch::stable::Tensor& lse,
                         double sm_scale,
                         std::optional<torch::stable::Tensor> topk_lens) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(q.is_cuda() && pool.is_cuda() && indices.is_cuda() &&
                      out.is_cuda() && lse.is_cuda(),
                  "all tensors must be CUDA");
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 576, "q must be [T, h, 576]");
  STD_TORCH_CHECK(q.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "q must be bf16");
  STD_TORCH_CHECK(pool.dim() == 2 && pool.size(1) == 656 &&
                      pool.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "pool must be [S, 656] u8");
  STD_TORCH_CHECK(indices.dim() == 2 &&
                      indices.scalar_type() == torch::headeronly::ScalarType::Int,
                  "indices must be [T, topk] i32");
  const int T = q.size(0), h = q.size(1), topk = indices.size(1);
  STD_TORCH_CHECK(T > 0 && indices.size(0) == T, "indices must have T rows");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == T && out.size(1) == h &&
                      out.size(2) == 512 &&
                      out.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "out must be [T, h, 512] bf16");
  STD_TORCH_CHECK(lse.dim() == 2 && lse.size(0) == T && lse.size(1) == h &&
                      lse.scalar_type() == torch::headeronly::ScalarType::Float,
                  "lse must be [T, h] f32");
  STD_TORCH_CHECK(q.is_contiguous() && pool.is_contiguous() && indices.is_contiguous() &&
                      out.is_contiguous() && lse.is_contiguous(),
                  "all tensors must be contiguous");
  const int32_t* lens_ptr = nullptr;
  if (topk_lens.has_value()) {
    const torch::stable::Tensor& lens = topk_lens.value();
    STD_TORCH_CHECK(lens.is_cuda() && lens.dim() == 1 && lens.size(0) == T &&
                        lens.scalar_type() == torch::headeronly::ScalarType::Int &&
                        lens.is_contiguous(),
                    "topk_lens must be a contiguous CUDA [T] i32 tensor");
    lens_ptr = lens.const_data_ptr<int32_t>();
  }

  const cudaStream_t stream = get_current_cuda_stream();
  dim3 grid(T, (h + sm89_dsa::HT - 1) / sm89_dsa::HT);
  if (lens_ptr != nullptr) {
    cudaFuncSetAttribute(sm89_dsa::sparse_mla_fwd_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, sm89_dsa::SMEM_TOTAL);
    sm89_dsa::sparse_mla_fwd_kernel<true><<<grid, 128, sm89_dsa::SMEM_TOTAL, stream>>>(
        static_cast<const __nv_bfloat16*>(q.const_data_ptr()), pool.const_data_ptr<uint8_t>(),
        indices.const_data_ptr<int32_t>(), static_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        lse.mutable_data_ptr<float>(), lens_ptr, h, topk, (float)sm_scale);
  } else {
    cudaFuncSetAttribute(sm89_dsa::sparse_mla_fwd_kernel<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, sm89_dsa::SMEM_TOTAL);
    sm89_dsa::sparse_mla_fwd_kernel<false><<<grid, 128, sm89_dsa::SMEM_TOTAL, stream>>>(
        static_cast<const __nv_bfloat16*>(q.const_data_ptr()), pool.const_data_ptr<uint8_t>(),
        indices.const_data_ptr<int32_t>(), static_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        lse.mutable_data_ptr<float>(), nullptr, h, topk, (float)sm_scale);
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  STD_TORCH_CHECK(false, "sm89_sparse_mla_fwd is not supported on ROCm");
#endif
}

#ifdef APHRODITE_ENABLE_SM89_DSA
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("sm89_sparse_mla_fwd", TORCH_BOX(&sm89_sparse_mla_fwd));
}
#endif

// clang-format on
