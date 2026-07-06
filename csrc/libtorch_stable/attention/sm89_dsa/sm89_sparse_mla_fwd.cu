// Sparse MLA forward attention for decode on sm89 (Ada), gathering top-k KV directly from the
// fp8_ds_mla 656B pool by slot.
//
//   out[t, i] = softmax_k(q[t, i, :576] . kv[idx[t, k], :576] * sm_scale) @ kv[idx[t, k], :512]
//
// fp8_ds_mla pool row (656 B, AoS):
//   [0, 512)   : 512 fp8 e4m3 "nope"
//   [512, 528) : 4 x fp32 per-128-tile scales
//   [528, 656) : 64 x bf16 rope
//
// Grid (T, ceil(h/16)); 128 threads / 4 warps; BI=32 keys per iteration.
//   2-stage cp.async staging ring of raw 656B rows (by slot; -1 -> slot 0 + score mask)
//   -> dequant nope fp8->bf16 (x tile scale) + rope passthrough into sKV [32 x 584]
//   -> QK bf16 mma m16n8k16 (warp w owns keys [8w, 8w+8); 36 k-steps over 576 dims)
//   -> masked scaled scores to smem; redundant per-lane online softmax (fixed order)
//   -> PV bf16 mma (warp w owns out cols [128w, 128w+128); B-frags via ldmatrix.x4.trans)
// l==0 rows (all -1 indices) write 0 output / -inf LSE. No atomics; fixed reduction orders
// -> bitwise run-to-run deterministic. Static shapes; -1 handling makes it CUDA-graph-safe.
//
// cp.async invariant: exactly STAGES commit groups in the prologue (empty
// commits pad when n_iters < STAGES) and exactly one commit per loop iteration (empty past
// the tail), so cp_async_wait<STAGES-1> always guarantees iteration `it`'s group landed.
//
// Variants (all sharing the staging/dequant pipeline, masking, and softmax numerics):
//   - sparse_mla_fwd_kernel<SPLIT>: the validated 16-head-tile kernel. SPLIT adds a
//     grid.z key-range split (flash-decoding): each CTA runs the same online softmax
//     over its BI-aligned key slice and emits unnormalized fp32 partial O plus
//     per-head (m, l) instead of the final output.
//   - sparse_mla_fwd_h8_kernel<SPLIT>: 8-head tile for ranks with h <= 8 (e.g. 64
//     heads at TP8). The mma m16 dimension is the head axis, so an h=8 tile through
//     the 16-head kernel would still burn full m16 tiles; instead this computes
//     S^T = K.Q^T and O^T = V^T.P^T so m holds keys / value dims and the mma count
//     halves. QK splits the 576-dim k-loop across the 4 warps (9 k-steps each) with
//     a fixed-order smem reduction; PV reuses the ldmatrix.x4.trans loads as
//     A-fragments (reg order {0, 2, 1, 3}).
//   - sparse_mla_combine_kernel: merges the SPLIT partials per (t, h) row in fixed
//     split order: m* = max_s m_s; L = sum_s exp(m_s - m*) * l_s;
//     O = sum_s exp(m_s - m*) * O_s / L. Slices with l == 0 (empty split or all -1
//     keys) are skipped; all-empty rows produce 0 output / -inf LSE exactly like the
//     single-pass epilogue. No atomics anywhere -> run-to-run deterministic.
// sm89_sparse_mla_fwd launches the unmodified <false> 16-head instantiation;
// sm89_sparse_mla_fwd_v2 selects h8_tile / num_splits (env-gated in the backend,
// both default off).

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
constexpr int DN = 512;  // value dims (MLA absorb: V == K nope)
constexpr int ROW_BYTES = 656;
constexpr int ROW_CHUNKS = ROW_BYTES / 16;  // 41
constexpr int BI = 32;                      // keys per pipeline iteration
constexpr int STAGES = 2;
constexpr int HT = 16;                    // head tile (one m16 mma tile)
constexpr int SKV_STRIDE = D + 8;         // halves; rows 1168B: 16B-aligned, 4-word bank skew
constexpr int SQ_STRIDE = D + 8;
constexpr int SS_STRIDE = BI + 1;         // fp32; +1 pad -> per-row bank skew

constexpr int HT8 = 8;                    // 8-head tile (transposed-mma variant)
constexpr int MAX_SPLITS = 8;             // grid.z cap for the split-KV variant

constexpr int SMEM_STAGING = STAGES * BI * ROW_BYTES;  // 41984
constexpr int SMEM_SKV = BI * SKV_STRIDE * 2;          // 37376
constexpr int SMEM_SQ = HT * SQ_STRIDE * 2;            // 18688
constexpr int SMEM_SS = HT * SS_STRIDE * 4;            // 2112
constexpr int SMEM_TOTAL = SMEM_STAGING + SMEM_SKV + SMEM_SQ + SMEM_SS;  // 100160 <= 99KB opt-in

constexpr int SMEM_SQ8 = HT8 * SQ_STRIDE * 2;             // 9344
constexpr int SMEM_SS8 = HT8 * SS_STRIDE * 4;             // 1056
constexpr int SMEM_SP8 = 4 * 2 * 32 * 4 * 4;              // 4096: per-warp QK partial C tiles
constexpr int SMEM_TOTAL_H8 =
    SMEM_STAGING + SMEM_SKV + SMEM_SQ8 + SMEM_SS8 + SMEM_SP8;  // 93856

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

// ------------------------------------------------------------- sparse MLA forward
// SPLIT: grid.z CTAs each cover a BI-aligned slice of the top-k keys and write
// unnormalized fp32 partial O to part_o [S, T, h, 512] and per-head (m, l) to
// part_ml [S, T, h, 2] for sparse_mla_combine_kernel; out/lse are untouched.
template <bool SPLIT>
__global__ void __launch_bounds__(128, 1)
sparse_mla_fwd_kernel(const __nv_bfloat16* __restrict__ q,   // [T, h, 576]
                      const uint8_t* __restrict__ pool,      // [S, 656]
                      const int32_t* __restrict__ indices,   // [T, topk], -1 padded
                      __nv_bfloat16* __restrict__ out,       // [T, h, 512]
                      float* __restrict__ lse,                // [T, h]
                      float* __restrict__ part_o,             // SPLIT only
                      float* __restrict__ part_ml,            // SPLIT only
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
  const int group = lane >> 2;  // 0..7
  const int quad = lane & 3;    // 0..3
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

  // SPLIT: this CTA owns iterations [it0, itN) of the key loop; empty slices
  // (it0 == itN) fall through to the epilogue with m = -inf / l = 0 / acc = 0.
  const int n_iters_total = (topk + BI - 1) / BI;
  int it0 = 0, itN = n_iters_total;
  if constexpr (SPLIT) {
    const int ips = (n_iters_total + (int)gridDim.z - 1) / (int)gridDim.z;
    it0 = min((int)blockIdx.z * ips, n_iters_total);
    itN = min(it0 + ips, n_iters_total);
  }
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    if (it0 + s < itN) issue_iter(it0 + s, s);
    else cp_async_commit();
  }

  float m_st[2] = {NEG_INF, NEG_INF};  // online softmax state for rows r0, r1
  float l_st[2] = {0.f, 0.f};
  float acc[16][4];  // 16 n8-tiles: out cols [128*warp, 128*warp+128)
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int it = it0; it < itN; ++it) {
    const int stage = (it - it0) % STAGES;
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

    if (it + STAGES < itN) issue_iter(it + STAGES, stage);
    else cp_async_commit();  // keep the commit-count invariant through the tail

    // ---- QK: warp owns keys [8*warp, 8*warp+8); 36 k-steps over 576 dims
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

    // ---- online softmax: every lane redundantly processes its two rows in fixed order
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

    // ---- PV: warp owns out cols [128*warp, 128*warp+128); B via ldmatrix.x4.trans
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

  // ---- epilogue: normalize, store bf16 out + fp32 lse (l==0 -> 0 / -inf);
  // SPLIT stores the raw fp32 (acc, m, l) partials for the combine kernel instead
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int hh = h_base + (rr ? r1 : r0);
    if (hh >= h) continue;
    const float l = l_st[rr];
    if constexpr (SPLIT) {
      const int64_t prow = (int64_t)blockIdx.z * gridDim.x * h + (int64_t)t * h + hh;
      float* orow = part_o + prow * DN;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const int col = 128 * warp + 8 * nt + 2 * quad;
        *reinterpret_cast<float2*>(orow + col) =
            make_float2(acc[nt][2 * rr], acc[nt][2 * rr + 1]);
      }
      if (warp == 0 && quad == 0)
        *reinterpret_cast<float2*>(part_ml + prow * 2) = make_float2(m_st[rr], l);
    } else {
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
}

// ------------------------------------------------- 8-head-tile sparse MLA forward
// Transposed-mma variant for h <= 8 heads per rank. The m16 dimension of
// mma.m16n8k16 is the head axis in the kernel above, so 8 heads pad half of every
// tile; here the products are computed transposed so m is always dense:
//   QK: S^T = K.Q^T   A = K [16 keys x 16 dims], B = Q [8 heads], C = [keys x heads]
//   PV: O^T = V^T.P^T A = V^T [16 cols x 16 keys], B = P^T [8 heads], C = [cols x heads]
// QK's 36 k-steps are split across the 4 warps (9 each, fixed-order smem reduce so
// determinism holds); PV's A-fragments come from the same ldmatrix.x4.trans loads
// the 16-head kernel uses as B-fragments, permuted {r0, r2, r1, r3}. Staging,
// dequant, masking, and the online-softmax numerics are identical to the kernel
// above. SPLIT behaves exactly as in sparse_mla_fwd_kernel.
template <bool SPLIT>
__global__ void __launch_bounds__(128, 1)
sparse_mla_fwd_h8_kernel(const __nv_bfloat16* __restrict__ q,   // [T, h, 576]
                         const uint8_t* __restrict__ pool,      // [S, 656]
                         const int32_t* __restrict__ indices,   // [T, topk], -1 padded
                         __nv_bfloat16* __restrict__ out,       // [T, h, 512]
                         float* __restrict__ lse,                // [T, h]
                         float* __restrict__ part_o,             // SPLIT only
                         float* __restrict__ part_ml,            // SPLIT only
                         int h, int topk, float sm_scale) {
  extern __shared__ uint8_t smem[];
  uint8_t* staging = smem;
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_STAGING);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_STAGING + SMEM_SKV);
  float* sS = reinterpret_cast<float*>(smem + SMEM_STAGING + SMEM_SKV + SMEM_SQ8);
  float* sSp =
      reinterpret_cast<float*>(smem + SMEM_STAGING + SMEM_SKV + SMEM_SQ8 + SMEM_SS8);

  const int t = blockIdx.x;
  const int h_base = blockIdx.y * HT8;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = lane >> 2;  // 0..7
  const int quad = lane & 3;    // 0..3
  const float NEG_INF = __int_as_float(0xff800000);

  const int32_t* idx_row = indices + (int64_t)t * topk;
  auto key_valid = [&](int k) { return k < topk && idx_row[k] >= 0; };

  // ---- q tile -> sQ (rows past h zeroed; pad cols never read)
  for (int c = tid; c < HT8 * (D / 8); c += 128) {
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

  const int n_iters_total = (topk + BI - 1) / BI;
  int it0 = 0, itN = n_iters_total;
  if constexpr (SPLIT) {
    const int ips = (n_iters_total + (int)gridDim.z - 1) / (int)gridDim.z;
    it0 = min((int)blockIdx.z * ips, n_iters_total);
    itN = min(it0 + ips, n_iters_total);
  }
#pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    if (it0 + s < itN) issue_iter(it0 + s, s);
    else cp_async_commit();
  }

  // Online-softmax state. Head `group` feeds this lane's P fragments (the mma B
  // operand); heads 2*quad(+1) own this lane's accumulator columns, rescale
  // factors, and epilogue (m, l). Every row is folded redundantly from sS in the
  // same fixed order, so all lanes derive bitwise-identical (m, l) per head and
  // the P values stay consistent with the acc rescales they meet in the mma.
  float m_pv = NEG_INF;                // running max for head `group`
  float m_st[2] = {NEG_INF, NEG_INF};  // heads 2*quad, 2*quad + 1
  float l_st[2] = {0.f, 0.f};
  float acc[8][4];  // 8 m16-tiles: out cols [128*warp, 128*warp+128); C cols = heads
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int it = it0; it < itN; ++it) {
    const int stage = (it - it0) % STAGES;
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

    if (it + STAGES < itN) issue_iter(it + STAGES, stage);
    else cp_async_commit();  // keep the commit-count invariant through the tail

    // ---- QK (S^T): A = K tile [16 keys x 16 dims], B = Q [8 heads x 16 dims];
    // warp w owns k-steps [9w, 9w+9) over the 576 dims for both 16-key m-tiles
    {
      float cq[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) cq[m][j] = 0.f;
      const __nv_bfloat16* bq = sQ + group * SQ_STRIDE + 2 * quad;
#pragma unroll
      for (int ks = 9 * warp; ks < 9 * warp + 9; ++ks) {
        uint32_t b[2];
        b[0] = *reinterpret_cast<const uint32_t*>(bq + ks * 16);
        b[1] = *reinterpret_cast<const uint32_t*>(bq + ks * 16 + 8);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const __nv_bfloat16* ka0 = sKV + (16 * m + group) * SKV_STRIDE + 2 * quad;
          const __nv_bfloat16* ka1 = ka0 + 8 * SKV_STRIDE;
          uint32_t a[4];
          a[0] = *reinterpret_cast<const uint32_t*>(ka0 + ks * 16);
          a[1] = *reinterpret_cast<const uint32_t*>(ka1 + ks * 16);
          a[2] = *reinterpret_cast<const uint32_t*>(ka0 + ks * 16 + 8);
          a[3] = *reinterpret_cast<const uint32_t*>(ka1 + ks * 16 + 8);
          mma_bf16(a, b, cq[m]);
        }
      }
      // per-warp partial C tiles -> sSp
#pragma unroll
      for (int m = 0; m < 2; ++m)
        *reinterpret_cast<float4*>(sSp + (warp * 2 + m) * 128 + lane * 4) =
            make_float4(cq[m][0], cq[m][1], cq[m][2], cq[m][3]);
    }
    __syncthreads();

    // ---- cross-warp reduce (fixed w order), mask + scale -> sS[head][key];
    // 256 (key, head) entries over 128 threads
    const int kbase = it * BI;
#pragma unroll
    for (int ee = 0; ee < 2; ++ee) {
      const int e = tid + 128 * ee;
      const int k = e >> 3, hh = e & 7;
      const int mt = k >> 4, kr = k & 15;
      // partial C layout: c[0]=(key g, head 2q) c[1]=(g, 2q+1) c[2]=(g+8, 2q) c[3]=(g+8, 2q+1)
      const int src = ((kr & 7) * 4 + (hh >> 1)) * 4 + (kr >> 3) * 2 + (hh & 1);
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) v += sSp[(w * 2 + mt) * 128 + src];
      sS[hh * SS_STRIDE + k] = key_valid(kbase + k) ? v * sm_scale : NEG_INF;
    }
    __syncthreads();

    // ---- online softmax: head `group` -> P^T B-frags; heads 2*quad(+1) -> acc state
    uint32_t pf[2][2];  // P^T bf16 B-frags for the 2 PV k-steps
    {
      const float* srow = sS + group * SS_STRIDE;
      float mx = m_pv;
#pragma unroll
      for (int k = 0; k < BI; ++k) mx = fmaxf(mx, srow[k]);
      if (mx > NEG_INF) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int kb = 16 * s + 2 * quad;
          pf[s][0] = bf16x2(__expf(srow[kb] - mx), __expf(srow[kb + 1] - mx));
          pf[s][1] = bf16x2(__expf(srow[kb + 8] - mx), __expf(srow[kb + 9] - mx));
        }
      } else {  // no valid key seen yet for this head
#pragma unroll
        for (int s = 0; s < 2; ++s) pf[s][0] = pf[s][1] = 0u;
      }
      m_pv = mx;
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const float* srow = sS + (2 * quad + rr) * SS_STRIDE;
      float mx = m_st[rr];
#pragma unroll
      for (int k = 0; k < BI; ++k) mx = fmaxf(mx, srow[k]);
      float alpha = 1.f, sum = 0.f;
      if (mx > NEG_INF) {
        alpha = (m_st[rr] > NEG_INF) ? __expf(m_st[rr] - mx) : 0.f;
#pragma unroll
        for (int k = 0; k < BI; ++k) sum += __expf(srow[k] - mx);  // -inf entries -> 0
      }
      l_st[rr] = l_st[rr] * alpha + sum;
      m_st[rr] = mx;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j][rr] *= alpha;
        acc[j][rr + 2] *= alpha;
      }
    }

    // ---- PV (O^T): A = V^T via ldmatrix.x4.trans (B-frag regs permuted to A order)
    const int mr = lane & 15;        // key row within the 16-key k-step tile
    const int mc = (lane >> 4) * 8;  // col half of the 16-col chunk
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {  // 16-col chunks of [128*warp, 128*warp+128)
        uint32_t r[4];
        const __nv_bfloat16* src = sKV + (16 * s + mr) * SKV_STRIDE + 128 * warp + 16 * j + mc;
        ldmatrix_x4_trans(r, src);
        const uint32_t a[4] = {r[0], r[2], r[1], r[3]};
        mma_bf16(a, pf[s], acc[j]);
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue: C rows = out cols (g, g+8 per 16-col tile), C cols = heads
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int hh = h_base + 2 * quad + rr;
    if (hh >= h) continue;
    const float l = l_st[rr];
    if constexpr (SPLIT) {
      const int64_t prow = (int64_t)blockIdx.z * gridDim.x * h + (int64_t)t * h + hh;
      float* orow = part_o + prow * DN;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = 128 * warp + 16 * j + group;
        orow[col] = acc[j][rr];
        orow[col + 8] = acc[j][rr + 2];
      }
      if (warp == 0 && group == 0)
        *reinterpret_cast<float2*>(part_ml + prow * 2) = make_float2(m_st[rr], l);
    } else {
      const float inv = (l > 0.f) ? 1.f / l : 0.f;
      __nv_bfloat16* orow = out + ((int64_t)t * h + hh) * DN;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = 128 * warp + 16 * j + group;
        orow[col] = __float2bfloat16(acc[j][rr] * inv);
        orow[col + 8] = __float2bfloat16(acc[j][rr + 2] * inv);
      }
      if (warp == 0 && group == 0)
        lse[(int64_t)t * h + hh] = (l > 0.f) ? m_st[rr] + logf(l) : NEG_INF;
    }
  }
}

// ------------------------------------------------------------- split-KV combine
// Standard flash-decoding merge of the S split partials per (t, h) row, in fixed
// split order (no atomics -> deterministic):
//   m* = max_s m_s;  L = sum_s exp(m_s - m*) * l_s;  O = sum_s exp(m_s - m*) * O_s / L
// Slices with l == 0 (empty split / all -1 keys) are skipped; if every slice is
// empty the row degenerates to 0 output / -inf LSE like the single-pass epilogue.
__global__ void __launch_bounds__(128, 1)
sparse_mla_combine_kernel(const float* __restrict__ part_o,   // [S, T, h, 512]
                          const float* __restrict__ part_ml,  // [S, T, h, 2]
                          __nv_bfloat16* __restrict__ out,    // [T, h, 512]
                          float* __restrict__ lse,            // [T, h]
                          int h, int num_splits) {
  const int t = blockIdx.x, hh = blockIdx.y;
  const int64_t row = (int64_t)t * h + hh;
  const int64_t row_stride = (int64_t)gridDim.x * h;
  const float NEG_INF = __int_as_float(0xff800000);

  float m = NEG_INF;
  for (int s = 0; s < num_splits; ++s)
    m = fmaxf(m, part_ml[(s * row_stride + row) * 2]);

  const int c0 = threadIdx.x * 4;  // 128 threads x 4 cols = 512
  float lsum = 0.f;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < num_splits; ++s) {
    const float2 ml = *reinterpret_cast<const float2*>(part_ml + (s * row_stride + row) * 2);
    if (!(ml.y > 0.f)) continue;  // empty slice; uniform branch across the block
    const float w = __expf(ml.x - m);
    lsum += w * ml.y;
    const float4 v =
        *reinterpret_cast<const float4*>(part_o + (s * row_stride + row) * DN + c0);
    o[0] += w * v.x;
    o[1] += w * v.y;
    o[2] += w * v.z;
    o[3] += w * v.w;
  }
  const float inv = (lsum > 0.f) ? 1.f / lsum : 0.f;
  *reinterpret_cast<uint2*>(out + row * DN + c0) =
      make_uint2(bf16x2(o[0] * inv, o[1] * inv), bf16x2(o[2] * inv, o[3] * inv));
  if (threadIdx.x == 0) lse[row] = (lsum > 0.f) ? m + logf(lsum) : NEG_INF;
}

}  // namespace sm89_dsa

#endif  // USE_ROCM

#ifndef USE_ROCM
namespace {

// Shared q/pool/indices/out/lse validation for both entry points; fills (T, h, topk).
void check_sparse_mla_fwd_args(const torch::stable::Tensor& q,
                               const torch::stable::Tensor& pool,
                               const torch::stable::Tensor& indices,
                               const torch::stable::Tensor& out,
                               const torch::stable::Tensor& lse, int& T, int& h,
                               int& topk) {
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
  T = q.size(0), h = q.size(1), topk = indices.size(1);
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
}

}  // namespace
#endif  // USE_ROCM

void sm89_sparse_mla_fwd(const torch::stable::Tensor& q,
                         const torch::stable::Tensor& pool,
                         const torch::stable::Tensor& indices,
                         torch::stable::Tensor& out, torch::stable::Tensor& lse,
                         double sm_scale) {
#ifndef USE_ROCM
  int T, h, topk;
  check_sparse_mla_fwd_args(q, pool, indices, out, lse, T, h, topk);

  const cudaStream_t stream = get_current_cuda_stream();
  cudaFuncSetAttribute(sm89_dsa::sparse_mla_fwd_kernel<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, sm89_dsa::SMEM_TOTAL);
  dim3 grid(T, (h + sm89_dsa::HT - 1) / sm89_dsa::HT);
  sm89_dsa::sparse_mla_fwd_kernel<false><<<grid, 128, sm89_dsa::SMEM_TOTAL, stream>>>(
      static_cast<const __nv_bfloat16*>(q.const_data_ptr()), pool.const_data_ptr<uint8_t>(),
      indices.const_data_ptr<int32_t>(), static_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      lse.mutable_data_ptr<float>(), nullptr, nullptr, h, topk, (float)sm_scale);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
#else
  STD_TORCH_CHECK(false, "sm89_sparse_mla_fwd is not supported on ROCm");
#endif
}

// Gated decode variants of sm89_sparse_mla_fwd (same contract; see the file
// header). h8_tile selects the transposed 8-head-tile kernel; num_splits > 1
// selects split-KV, partitioning the top-k keys over grid.z CTAs whose fp32
// partials (part_o [S, T, h, 512], part_ml [S, T, h, 2]) are merged by the
// combine kernel. num_splits == 1 with h8_tile == false is bitwise identical to
// sm89_sparse_mla_fwd.
void sm89_sparse_mla_fwd_v2(const torch::stable::Tensor& q,
                            const torch::stable::Tensor& pool,
                            const torch::stable::Tensor& indices,
                            torch::stable::Tensor& out, torch::stable::Tensor& lse,
                            const std::optional<torch::stable::Tensor>& part_o,
                            const std::optional<torch::stable::Tensor>& part_ml,
                            double sm_scale, int64_t num_splits, bool h8_tile) {
#ifndef USE_ROCM
  int T, h, topk;
  check_sparse_mla_fwd_args(q, pool, indices, out, lse, T, h, topk);
  STD_TORCH_CHECK(num_splits >= 1 && num_splits <= sm89_dsa::MAX_SPLITS,
                  "num_splits must be in [1, 8]");
  float* part_o_ptr = nullptr;
  float* part_ml_ptr = nullptr;
  if (num_splits > 1) {
    STD_TORCH_CHECK(part_o.has_value() && part_ml.has_value(),
                    "num_splits > 1 requires part_o and part_ml workspaces");
    torch::stable::Tensor po = part_o.value(), pml = part_ml.value();
    STD_TORCH_CHECK(po.is_cuda() && po.dim() == 4 && po.size(0) == num_splits &&
                        po.size(1) == T && po.size(2) == h && po.size(3) == 512 &&
                        po.scalar_type() == torch::headeronly::ScalarType::Float &&
                        po.is_contiguous(),
                    "part_o must be [num_splits, T, h, 512] f32 contiguous");
    STD_TORCH_CHECK(pml.is_cuda() && pml.dim() == 4 && pml.size(0) == num_splits &&
                        pml.size(1) == T && pml.size(2) == h && pml.size(3) == 2 &&
                        pml.scalar_type() == torch::headeronly::ScalarType::Float &&
                        pml.is_contiguous(),
                    "part_ml must be [num_splits, T, h, 2] f32 contiguous");
    part_o_ptr = po.mutable_data_ptr<float>();
    part_ml_ptr = pml.mutable_data_ptr<float>();
  }

  const cudaStream_t stream = get_current_cuda_stream();
  const auto* qp = static_cast<const __nv_bfloat16*>(q.const_data_ptr());
  const uint8_t* poolp = pool.const_data_ptr<uint8_t>();
  const int32_t* idxp = indices.const_data_ptr<int32_t>();
  auto* outp = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  float* lsep = lse.mutable_data_ptr<float>();

  const int ht = h8_tile ? sm89_dsa::HT8 : sm89_dsa::HT;
  const int smem = h8_tile ? sm89_dsa::SMEM_TOTAL_H8 : sm89_dsa::SMEM_TOTAL;
  dim3 grid(T, (h + ht - 1) / ht, (unsigned)num_splits);
  auto launch = [&](auto* kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, 128, smem, stream>>>(qp, poolp, idxp, outp, lsep, part_o_ptr,
                                        part_ml_ptr, h, topk, (float)sm_scale);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (h8_tile) {
    if (num_splits > 1) launch(sm89_dsa::sparse_mla_fwd_h8_kernel<true>);
    else launch(sm89_dsa::sparse_mla_fwd_h8_kernel<false>);
  } else {
    if (num_splits > 1) launch(sm89_dsa::sparse_mla_fwd_kernel<true>);
    else launch(sm89_dsa::sparse_mla_fwd_kernel<false>);
  }
  if (num_splits > 1) {
    sm89_dsa::sparse_mla_combine_kernel<<<dim3(T, h), 128, 0, stream>>>(
        part_o_ptr, part_ml_ptr, outp, lsep, h, (int)num_splits);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }
#else
  STD_TORCH_CHECK(false, "sm89_sparse_mla_fwd_v2 is not supported on ROCm");
#endif
}

#ifdef APHRODITE_ENABLE_SM89_DSA
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("sm89_sparse_mla_fwd", TORCH_BOX(&sm89_sparse_mla_fwd));
  ops.impl("sm89_sparse_mla_fwd_v2", TORCH_BOX(&sm89_sparse_mla_fwd_v2));
}
#endif

// clang-format on
