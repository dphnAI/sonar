// SPDX-License-Identifier: Apache-2.0
//
// The decode MoE of b12x's compact W4A8 micro path on SM120 in one persistent
// kernel: FC1 (gate/up projection) -> SwiGLU + MXFP8 activation -> FC2 (down
// projection), with the same projections, intermediate and route_output bits
// as b12x's three kernels. FC2 of an expert starts as soon as its activation is
// done, so DRAM stays busy across the phases.
//
// Adapted from b12x_moe_fused_sm120.cu in vLLM-Moet
// (https://github.com/kacper-daftcode/vLLM-Moet,
// tools/dsv41_sm120/b12x_moe_fused), Copyright 2026 the vLLM-Moet authors,
// licensed under the Apache License, Version 2.0. Changes: the device code is
// unchanged; the host entry points use the stable ABI and launch on the current
// stream with the served defaults (two CTAs per SM, four K64 slices prefetched,
// 512 ns backoff cap, no trace).
//
// The kernel keeps a small control block (work counter, per-expert flags) that
// its last CTA resets, so a launch replays from a CUDA graph. Work items are
// claimed in order and FC1 items never wait, so a waiting CTA always waits on a
// running one.

#include "b12x_moe_fused_sm120.h"
#include "torch_utils.h"

#if defined(APHRODITE_ENABLE_SM120_MOE_FUSED) && !defined(USE_ROCM)

  #include <cuda_bf16.h>
  #include <cuda_runtime.h>
  #include <torch/csrc/stable/accelerator.h>
  #include <torch/csrc/stable/library.h>
  #include <torch/headeronly/core/ScalarType.h>

  #include <algorithm>
  #include <cstdint>

namespace b12x_moe_fused {

constexpr int kThreads = 128;
constexpr int kTileN = 128;
constexpr int kMaxRows = 8;  // pairs per expert (tokens <= 8)
constexpr int kMaxPairs = 64;
constexpr int kMaxExperts = 64;  // unique experts of a launch (<= pairs)
constexpr int kMaxExpertIds = 384;

// FC1 ring: K64 stages; the rows' K/32 scale bytes (160 B per row) are staged
// once per tile after the ring
constexpr int kF1Stages = 8;
constexpr int kF1A = kMaxRows * 64;
constexpr int kF1B = 2 * 4 * 32 * 16;  // K32(2) x N32(4) x lane x 16 B = 4 KB
constexpr int kF1Sfb = 128 * 4;
constexpr int kF1Stage = kF1A + kF1B + kF1Sfb;  // 5,120 B
constexpr int kF1RowScales = 256;  // >= K/32 = 160 B per row, 16 B granules
constexpr int kF1Smem =
    kF1Stages * kF1Stage + kMaxRows * kF1RowScales;  // 43,008 B

// FC2: 64-column items (half of a 128-column w2 tile), two buffers so the next
// item's weights stream in while the current one is computed; the tail slice
// holds two K32 columns. The A fragments are read straight into registers.
constexpr int kSlices = 5;
constexpr int kTotalK32 = 4 * kSlices - 2;  // 18
constexpr int kF2Cols = 64;
constexpr int kF2BItem =
    kTotalK32 * kF2Cols * 16;  // 18 K32 x 2 n32 chunks x 512 B = 18,432 B
constexpr int kF2SfbItem =
    (kSlices - 1) * kF2Cols * 4 + kF2Cols * 2;  // 1,152 B
constexpr int kF2Buf = kF2BItem + kF2SfbItem;   // 19,584 B
constexpr int kF2Smem = 2 * kF2Buf;             // 39,168 B

constexpr int kSmem = kF1Smem > kF2Smem ? kF1Smem : kF2Smem;
static_assert(2 * (kSmem + 5 * 1024 + 1024) <= 100 * 1024,
              "two CTAs per SM within sm_120 100 KB (static tables ~5 KB + 1 "
              "KB reserved each)");

// control block (int32)
constexpr int kCtrlNext = 0, kCtrlFinished = 1, kCtrlNReady = 2;
constexpr int kCtrlSliceDone =
    4;  // [u * 5 + s] FC1 tiles (gate, up) of the slice done
constexpr int kCtrlActDone =
    kCtrlSliceDone + kMaxExperts * kSlices;  // [u] slices activated
constexpr int kCtrlReady =
    kCtrlActDone + kMaxExperts;  // [u] expert ready for FC2
constexpr int kCtrlReadyList =
    kCtrlReady +
    kMaxExperts;  // [k] k-th ready expert among the dynamic ones, -1
constexpr int kCtrlInts = kCtrlReadyList + kMaxExperts;
constexpr int kTraceItems = 16;

struct Args {
  const uint32_t* a_values;  // [cap][K] e4m3 (b12x rows-quant)
  const uint8_t* a_scales;   // [cap][K/32] ue8m0
  const uint32_t* w13;     // prepared n64-repacked [experts][2][N][K/32][16 B]
  const uint32_t* w13_sf;  // [experts][2][N][K/32] bytes
  __nv_bfloat16* projections;  // scratch [cap*topk][2N] bf16: gate | up
  uint32_t* intermediate;      // scratch: rows of 640 B, then scale words
  const uint32_t*
      w2;  // n64-repacked [experts][hidden/128][18 K32][128 cols][16 B]
  const uint8_t* w2_sf;        // [experts][hidden/128][128 cols][18]
  const int32_t* ids;          // [pairs] int32
  const float* token_weights;  // [pairs]
  const float* alpha1;
  const float* input_scale;
  const float* alpha2;
  const float* down_scale;
  __nv_bfloat16* out;  // route_output [rows_capacity][hidden]
  int* ctrl;
  long long* trace;  // optional [grid][kTraceItems][4]
  float limit;
  int is_per_expert;
  int ds_per_expert;
  int num_pairs;
  int topk;
  int rows_capacity;
  int k;        // hidden (5120)
  int n;        // intermediate (576)
  int n_tiles;  // ceil(n / 128) = 5
  int out_tiles;
  int num_experts;
  int pf_depth;    // FC1: K64 slices prefetched into L2 ahead of the ring (0 =
                   // off)
  int phase_mask;  // debug: 1 = FC1 + activation only, 2 = FC2 only (no wait;
                   // intermediate as given), 3 = all
  unsigned spin_ns_max;  // cap of the exponential backoff while waiting for an
                         // expert
};

__device__ __forceinline__ long long globaltimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
               "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void st_shared_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v));
}
__device__ __forceinline__ void st_shared_zero16(uint32_t addr) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(addr), "r"(0u));
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
// the weights are streamed once: an evict_first L2 policy keeps them from
// displacing the activations and the rows (FC1 -0.6 us per layer on the
// recorded routing, FC2 within noise)
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src,
                                              uint64_t pol) {
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst),
      "l"(src), "l"(pol));
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}

// b12x's e2m1x8_to_qmma_e2m1x8: eight FP4 nibbles into E2M1 byte containers
// (nibble in bits 5:2)
__device__ __forceinline__ void spread_e2m1x8(uint32_t packed, uint32_t& lo,
                                              uint32_t& hi) {
  asm("{\n"
      ".reg .b32 s, t;\n"
      "shl.b32 s, %2, 2;\n"
      "shr.u32 t, %2, 2;\n"
      "prmt.b32 %0, s, t, 0x5140;\n"
      "prmt.b32 %1, s, t, 0x7362;\n"
      "and.b32 %0, %0, 0x3C3C3C3C;\n"
      "and.b32 %1, %1, 0x3C3C3C3C;\n"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(packed));
}

template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1,
                                     uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row."
      "col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, "
      "{%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
        "r"(sfb), "n"(KB), "n"(0));
}
template <int KB>
__device__ __forceinline__ void qmma0(float (&d)[4], uint32_t a0, uint32_t a1,
                                      uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1, uint32_t sfa, uint32_t sfb) {
  const float z = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row."
      "col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%14, %14, %14, %14}, "
      "{%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
        "r"(sfb), "n"(KB), "n"(0), "f"(z));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint16_t to_bf16_rn(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}

// b12x's compact activation for 4 columns of one row (lane of an 8-lane K32
// group): SwiGLU with limit, bf16 boundary, MXFP8 quantization with the group's
// ue8m0 scale. Returns the e4m3x4 word; *scale_byte gets the group's scale.
// `g2`/`u2` are the gate / up bf16x4 of the columns (ignored when !valid).
__device__ __forceinline__ uint32_t act4(uint2 g2, uint2 u2, float limit,
                                         bool valid, uint32_t* scale_byte) {
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (valid) {
    const uint32_t gw[2] = {g2.x, g2.y}, uw[2] = {u2.x, u2.y};
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float g = __uint_as_float(
          (i & 1 ? gw[i >> 1] & 0xffff0000u : gw[i >> 1] << 16));
      float u = __uint_as_float(
          (i & 1 ? uw[i >> 1] & 0xffff0000u : uw[i >> 1] << 16));
      g = g >= limit ? limit : g;
      u = u >= limit ? limit : u;
      u = u <= -limit ? -limit : u;
      float t, e, sg;
      asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
      asm("add.rn.f32 %0, %1, 0f3F800000;" : "=f"(sg) : "f"(e));
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sg) : "f"(sg));
      const float m = __fmul_rn(g, sg);
      const float x = __fmul_rn(u, m);
      unsigned short h;
      asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
      v[i] = __uint_as_float((uint32_t)h << 16);
    }
  }
  float amax;
  asm("abs.f32 %0, %1;" : "=f"(amax) : "f"(v[0]));
  #pragma unroll
  for (int i = 1; i < 4; ++i) {
    float ab;
    asm("abs.f32 %0, %1;" : "=f"(ab) : "f"(v[i]));
    asm("max.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(ab));
  }
  #pragma unroll
  for (int sh = 1; sh <= 4; sh <<= 1) {
    const float o = __shfl_xor_sync(0xffffffffu, amax, sh);
    asm("max.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(o));
  }
  float sc;
  asm("mul.rn.f32 %0, %1, 0f3B124925;" : "=f"(sc) : "f"(amax));
  uint32_t bits = __float_as_uint(sc);
  if (bits & 0x007FFFFFu) bits = (bits + 0x00800000u) & 0x7F800000u;
  const uint32_t byte = (bits >> 23) & 0xFFu;
  float inv;
  {
    const float ef = (float)(127 - (int)byte);
    asm("ex2.approx.f32 %0, %1;" : "=f"(inv) : "f"(ef));
    if (byte == 0) inv = 0.f;
  }
  if (!valid) {
    *scale_byte = 0;
    return 0u;
  }
  *scale_byte = byte;
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
      : "=h"(lo)
      : "f"(__fmul_rn(inv, v[0])), "f"(__fmul_rn(inv, v[1])));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
      : "=h"(hi)
      : "f"(__fmul_rn(inv, v[2])), "f"(__fmul_rn(inv, v[3])));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

struct Tables {
  int16_t expert[kMaxExperts];
  uint8_t rows[kMaxExperts][kMaxRows];
  uint8_t tok[kMaxExperts][kMaxRows];  // rows / topk
  uint8_t nrows[kMaxExperts];
  uint8_t pad[kMaxPairs];
  int16_t slot_of[kMaxExpertIds];
  int first[kMaxExpertIds];  // first pair of each expert id (prologue scratch)
  int cnt[kMaxExpertIds];    // pairs of each expert id (prologue scratch)
  int16_t ids[kMaxPairs];
  int w0_leaders;
  int w0_pads;
  int n_unique;
  int n_pad;
  int item;
  int next;
  int flag;
  int n_trace;
  int pf_depth;
};

// ---------------------------------------------------------------- FC1: one
// 128-column projection tile of an expert
__device__ __forceinline__ void fc1_tile(const Args& a, Tables& T,
                                         uint8_t* smem, int u, int ptile) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nrows = T.nrows[u];
  const int e = T.expert[u];
  const int K = a.k, N = a.n;
  const int projection = ptile / a.n_tiles,
            otile = ptile - projection * a.n_tiles;
  const int n_cols = otile == a.n_tiles - 1
                         ? (N - otile * kTileN)
                         : kTileN;  // 64 in the n64 tail tile
  const size_t wexp = (size_t)e * N * (K / 32) * 8 +
                      (size_t)projection * N * (K / 32) * 4 +
                      (size_t)otile * kTileN * (K / 32) * 4;
  const size_t sexp = (size_t)e * N * (K / 32) / 2 +
                      (size_t)projection * N * (K / 32) / 4 +
                      (size_t)otile * kTileN * (K / 32) / 4;
  const int k64_tiles = K / 64;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int n32_chunks = n_cols / 32;
  const uint32_t* wtile = a.w13 + wexp;
  const uint32_t* stile = a.w13_sf + sexp;
  const int tile_words_per_k32 =
      n32_chunks * 128;  // 2 KB (full) or 1 KB (tail) per K32 block
  const uint32_t s_rowsc = sbase + kF1Stages * kF1Stage;  // [row][K/32 bytes]
  const int sc_vecs = K / 32 / 16;  // 16-byte granules of a row's scales (10)
  const uint64_t pol = policy_evict_first();
  if (tid < nrows * 16) {
    const int row = tid >> 4, v = tid & 15;
    if (v < sc_vecs)
      cp_async16(s_rowsc + row * kF1RowScales + v * 16,
                 a.a_scales + (size_t)T.tok[u][row] * (K / 32) + v * 16);
  }

  auto issue = [&](int j) {
    const uint32_t st = sbase + (j % kF1Stages) * kF1Stage;
    if (tid < nrows * 4) {
      const int row = tid >> 2, v = tid & 3;
      cp_async16(st + row * 64 + v * 16,
                 a.a_values + (size_t)T.tok[u][row] * (K / 4) + j * 16 + v * 4);
    }
  #pragma unroll
    for (int i = 0; i < 2; ++i) {  // kb
      const int idx = tid + i * kThreads;
      const int l = idx & 31, w = (idx >> 5) & 3, kb = idx >> 7;
      const uint32_t dst = st + kF1A + ((kb * 4 + w) * 32 + l) * 16;
      if (w < n32_chunks)
        cp_async16_ef(
            dst,
            wtile + (size_t)(j * 2 + kb) * tile_words_per_k32 + w * 128 + l * 4,
            pol);
      else
        st_shared_zero16(dst);
    }
    if (tid < 32) {  // 16 B = 4 columns' scale words
      const uint32_t dst = st + kF1A + kF1B + tid * 16;
      if (tid < n_cols / 4)
        cp_async16_ef(dst, stile + (size_t)(j >> 1) * n_cols + tid * 4, pol);
      else
        st_shared_zero16(dst);
    }
    const int pj = j + T.pf_depth;
    if (T.pf_depth > 0 && pj < k64_tiles) {
      const int lines =
          tile_words_per_k32 * 2 * 4 / 128;  // 32 (full) or 16 (tail)
      if (tid < lines)
        prefetch_l2(reinterpret_cast<const uint8_t*>(
                        wtile + (size_t)pj * 2 * tile_words_per_k32) +
                    tid * 128);
      if ((pj & 1) == 0 && tid >= 32 && tid < 32 + n_cols * 4 / 128)
        prefetch_l2(reinterpret_cast<const uint8_t*>(stile + (size_t)(pj >> 1) *
                                                                 n_cols) +
                    (tid - 32) * 128);
    }
  };

  #pragma unroll 1
  for (int j = 0; j < kF1Stages - 1; ++j) {
    issue(j);
    cp_async_commit();
  }
  const int q = lane >> 2, c = lane & 3;
  const float vs = __fmul_rn(__ldg(a.alpha1 + e),
                             __ldg(a.input_scale + (a.is_per_expert ? e : 0)));
  float acc[4][4];
  #pragma unroll
  for (int nt = 0; nt < 4; ++nt)
  #pragma unroll
    for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;

  for (int j = 0; j < k64_tiles; ++j) {
    cp_async_wait<kF1Stages - 2>();
    __syncthreads();
    if (j + kF1Stages - 1 < k64_tiles) issue(j + kF1Stages - 1);
    cp_async_commit();
    const uint32_t st = sbase + (j % kF1Stages) * kF1Stage;
    const uint32_t shift = (j & 1) * 16;
    uint32_t asc;
    asm volatile("ld.shared.u32 %0, [%1];"
                 : "=r"(asc)
                 : "r"(s_rowsc + q * kF1RowScales + (j >> 1) * 4));
    asc >>= shift;
  #pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      uint32_t a0, a2;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(a0), "=r"(a2)
                   : "r"(st + q * 64 + kb * 32 + c * 8));
      uint32_t w[4];
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                   : "r"(st + kF1A + ((kb * 4 + warp) * 32 + lane) * 16));
  #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(w[nt], b0, b1);
        uint32_t sfb;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(sfb)
                     : "r"(st + kF1A + kF1B + ((warp * 4 + nt) * 8 + q) * 4));
        sfb >>= shift;
        float x[4];
        if (kb == 0)
          qmma0<0>(x, a0, a0, a2, a2, b0, b1, asc, sfb);
        else
          qmma0<1>(x, a0, a0, a2, a2, b0, b1, asc, sfb);
  #pragma unroll
        for (int i = 0; i < 4; ++i) acc[nt][i] = __fadd_rn(acc[nt][i], x[i]);
      }
    }
  }
  cp_async_wait<0>();

  if (q < nrows) {
    const int pair = T.rows[u][q];
    int dst = otile * kTileN;
    if (ptile < a.n_tiles)
      dst += N;  // the prepared layout holds up first; projections hold gate,
                 // then up
    uint16_t* row =
        reinterpret_cast<uint16_t*>(a.projections + (size_t)pair * 2 * N);
    const int col_base = warp * 32 + (c << 1);
  #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int tc = col_base + nt * 8;
      if (otile * kTileN + tc < N) {
        const uint32_t v =
            (uint32_t)to_bf16_rn(__fmul_rn(vs, acc[nt][0])) |
            ((uint32_t)to_bf16_rn(__fmul_rn(vs, acc[nt][1])) << 16);
        *reinterpret_cast<uint32_t*>(row + dst + tc) = v;
      }
    }
  }
}

// ---------------------------------------------- activation of one K128 slice
// of an expert's rows (once per slice) by the CTA that finished the second of
// the slice's two FC1 tiles (gate, up); reads both halves through L2.
__device__ __forceinline__ void activation_slice(const Args& a, Tables& T,
                                                 int u, int s) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nrows = T.nrows[u];
  const int words_per_row = kSlices * 32;
  const int col = s * 128 + lane * 4;
  const bool valid = col < a.n;
  // a warp handles rows warp and warp + 4: all loads first (one round trip),
  // then the compute and stores
  uint2 g2[2] = {make_uint2(0u, 0u), make_uint2(0u, 0u)},
        u2[2] = {make_uint2(0u, 0u), make_uint2(0u, 0u)};
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = warp + i * (kThreads / 32);
    if (valid && r < nrows) {
      const __nv_bfloat16* prow =
          a.projections + (size_t)T.rows[u][r] * 2 * a.n;
      g2[i] = __ldcg(reinterpret_cast<const uint2*>(prow + col));
      u2[i] = __ldcg(reinterpret_cast<const uint2*>(prow + a.n + col));
    }
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = warp + i * (kThreads / 32);
    if (r >= nrows) break;
    const int pair = T.rows[u][r];
    uint32_t sb;
    const uint32_t w = act4(g2[i], u2[i], a.limit, valid, &sb);
    a.intermediate[(size_t)pair * words_per_row + s * 32 + lane] = w;
    if ((lane & 7) == 0) {
      uint8_t* sw = reinterpret_cast<uint8_t*>(
          a.intermediate + (size_t)a.rows_capacity * words_per_row +
          (size_t)s * a.rows_capacity + pair);
      sw[lane >> 3] = (uint8_t)sb;
    }
  }
}

// ----------------------------------------------------------------- FC2: one
// 64-column output block of an expert part 1 (no dependency on the expert's
// activation): the w2 half tile and its scales into buffer `buf`, one commit
// group
__device__ __forceinline__ void fc2_issue_w(const Args& a, uint8_t* smem, int e,
                                            int item, int buf) {
  const int tid = threadIdx.x;
  const int tile = item >> 1, half = item & 1;
  const size_t expert_words = (size_t)a.out_tiles * kTileN * kTotalK32 * 4;
  const uint32_t* wsrc = a.w2 + (size_t)e * expert_words +
                         (size_t)tile * (kTileN * kTotalK32 * 4) + half * 256;
  const uint8_t* sfsrc =
      a.w2_sf + ((size_t)e * a.out_tiles + tile) * kTileN * kTotalK32;
  const uint32_t sb =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + buf * kF2Buf;
  const uint64_t pol = policy_evict_first();
  // B: per K32 block the half tile's two n32 chunks (1 KB contiguous); 18
  // blocks = 18 KB
  const int l = tid & 31, n32 = (tid >> 5) & 1,
            kbo = tid >> 6;  // 128 threads: two K32 blocks per pass
  #pragma unroll 1
  for (int kb = kbo; kb < kTotalK32; kb += 2)
    cp_async16_ef(sb + (kb * 2 + n32) * 512 + l * 16,
                  wsrc + (size_t)kb * 512 + n32 * 128 + l * 4, pol);
  // scales: slices 0-3: 64 columns x 4 B (256 B each); tail: 64 columns x 2 B
  // (128 B)
  if (tid < 4 * 16 + 8) {
    const int s = tid >> 4, v = tid & 15;
    cp_async16_ef(
        sb + kF2BItem + s * 256 + v * 16,
        sfsrc + (size_t)s * 512 + half * (s == 4 ? 128 : 256) + v * 16, pol);
  }
  cp_async_commit();
}

// part 2a (after the expert is ready, before waiting for the buffer): this
// thread's A fragments from `intermediate` (through L2): 8 bytes of row `pair`
// per K32 block (the logical vector vec = kb*2 + (c>>1) of each slice-row), the
// row's scale word per slice, and the epilogue's scalars
struct Fc2Regs {
  uint2 af[kTotalK32];
  uint32_t asc[kSlices];
  float scale;
  int pair;
};
__device__ __forceinline__ void fc2_load_a(const Args& a, Tables& T, int u,
                                           Fc2Regs& R) {
  const int lane = threadIdx.x & 31;
  const int nrows = T.nrows[u];
  const int e = T.expert[u];
  const int words_per_row = kSlices * 32;
  const int q = lane >> 2, c = lane & 3;
  R.pair = T.rows[u][q < nrows ? q : 0];
  const uint32_t* arow = a.intermediate + (size_t)R.pair * words_per_row;
  #pragma unroll
  for (int k32 = 0; k32 < kTotalK32; ++k32) {
    const int s = k32 >> 2, kb = k32 & 3;
    R.af[k32] = __ldcg(reinterpret_cast<const uint2*>(
        arow + s * 32 + (kb * 2 + (c >> 1)) * 4 + (c & 1) * 2));
  }
  #pragma unroll
  for (int s = 0; s < kSlices; ++s)
    R.asc[s] = __ldcg(a.intermediate + (size_t)a.rows_capacity * words_per_row +
                      (size_t)s * a.rows_capacity + R.pair);
  const float down_scale =
      __ldg(a.alpha2 + e) * __ldg(a.down_scale + (a.ds_per_expert ? e : 0));
  R.scale = down_scale * __ldg(a.token_weights + R.pair);
}

// part 2b (buffer landed): MMAs and epilogue
__device__ __forceinline__ void fc2_mma(const Args& a, Tables& T, uint8_t* smem,
                                        int u, int item, int buf,
                                        const Fc2Regs& R) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nrows = T.nrows[u];
  const uint32_t sb =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + buf * kF2Buf;
  const int q = lane >> 2, c = lane & 3;
  float acc[2][4];
  #pragma unroll
  for (int nt = 0; nt < 2; ++nt)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[nt][j] = 0.f;
  // warp w: local columns [16w, 16w+16) = n32 chunk w>>1, n8 tiles 2(w&1),
  // 2(w&1)+1
  const uint32_t bbase = sb + ((warp >> 1) * 32 + lane) * 16 + (warp & 1) * 8;
  const int lc0 = warp * 16 + q;  // local column of nt = 0 (nt adds 8)

  #pragma unroll
  for (int s = 0; s < kSlices; ++s) {
    const bool tail = s == kSlices - 1;
  #pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const int k32 = s * 4 + kb;
      uint32_t w0 = 0u, w1 = 0u;
      if (!tail || kb < 2)  // the tail slice's K32 columns 2-3 have zero B, as
                            // in b12x's phase 2
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                     : "=r"(w0), "=r"(w1)
                     : "r"(bbase + k32 * 1024));
      const uint32_t a0 = (!tail || kb < 2)
                              ? R.af[k32].x
                              : 0u;  // the padding columns are zero, as in b12x
      const uint32_t a2 = (!tail || kb < 2) ? R.af[k32].y : 0u;
  #pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uint32_t b0, b1;
        spread_e2m1x8(nt ? w1 : w0, b0, b1);
        uint32_t sfb;
        if (tail) {
          uint16_t h;
          asm volatile("ld.shared.u16 %0, [%1];"
                       : "=h"(h)
                       : "r"(sb + kF2BItem + 4 * 256 + (lc0 + nt * 8) * 2));
          sfb = h;
        } else {
          asm volatile("ld.shared.u32 %0, [%1];"
                       : "=r"(sfb)
                       : "r"(sb + kF2BItem + s * 256 + (lc0 + nt * 8) * 4));
        }
        if (kb == 0)
          qmma<0>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else if (kb == 1)
          qmma<1>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else if (kb == 2)
          qmma<2>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
        else
          qmma<3>(acc[nt], a0, a0, a2, a2, b0, b1, R.asc[s], sfb);
      }
    }
  }

  if (q < nrows) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(a.out + (size_t)R.pair * a.k);
    const int col_base = item * kF2Cols + warp * 16 + (c << 1);
  #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int col = col_base + nt * 8;
      asm volatile(
          "st.global.u32 [%0], %1;" ::"l"(dst + (col >> 1)),
          "r"(pack_bf16x2(R.scale * acc[nt][0], R.scale * acc[nt][1])));
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// the kernel
__global__ void __launch_bounds__(kThreads, 2) moe_fused_kernel(Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Tables T;
  const int tid = threadIdx.x;
  long long t_entry = 0;
  if (a.trace != nullptr && tid == 0) t_entry = globaltimer();

  // the expert table: unique experts in order of their first pair (slot order),
  // each with its pairs (rows); the row order within an expert comes from
  // shared-memory atomics (it does not affect any result: rows are independent)
  if (tid < kMaxPairs)
    T.ids[tid] = (int16_t)(tid < a.num_pairs ? __ldg(a.ids + tid) : -1);
  if (tid == 0) T.n_trace = 1;
  for (int i = tid; i < kMaxExpertIds; i += kThreads) {
    T.first[i] = 0x7fff;
    T.cnt[i] = 0;
  }
  __syncthreads();
  const int my_id = tid < kMaxPairs ? T.ids[tid] : -1;
  const bool valid = my_id >= 0 && my_id < a.num_experts;
  int rank = 0;
  if (valid) {
    atomicMin(&T.first[my_id], tid);
    rank = atomicAdd(&T.cnt[my_id], 1);
  }
  __syncthreads();
  const bool leader = valid && T.first[my_id] == tid;
  const int cnt = valid ? T.cnt[my_id] : 0;
  // leaders numbered in pair order (warps 0 and 1 hold the 64 pairs; warp 1
  // adds warp 0's count); pads likewise
  const unsigned lm = __ballot_sync(0xffffffffu, leader);
  const unsigned pm = __ballot_sync(0xffffffffu, tid < a.num_pairs && !valid);
  if (tid == 0) {
    T.w0_leaders = __popc(lm);
    T.w0_pads = __popc(pm);
  }
  __syncthreads();
  if (tid < kMaxPairs) {
    const int lane = tid & 31;
    const int base_l = tid < 32 ? 0 : T.w0_leaders;
    const int base_p = tid < 32 ? 0 : T.w0_pads;
    const int my_slot = base_l + __popc(lm & ((1u << lane) - 1u));
    const int my_pad = base_p + __popc(pm & ((1u << lane) - 1u));
    if (leader) {
      T.expert[my_slot] = (int16_t)my_id;
      T.nrows[my_slot] = (uint8_t)min(cnt, kMaxRows);
      T.slot_of[my_id] = (int16_t)my_slot;
    }
    if (tid < a.num_pairs && !valid) T.pad[my_pad] = (uint8_t)tid;
    if (tid == kMaxPairs - 1) {
      T.n_unique = base_l + __popc(lm);
      T.n_pad = base_p + __popc(pm);
    }
  }
  __syncthreads();
  if (valid && rank < kMaxRows) {
    T.rows[T.slot_of[my_id]][rank] = (uint8_t)tid;
    T.tok[T.slot_of[my_id]][rank] = (uint8_t)(tid / a.topk);
  }
  if (a.trace != nullptr && tid == 0) {
    long long* tr = a.trace + (size_t)blockIdx.x * kTraceItems * 4;
    tr[0] = -1;
    tr[1] = t_entry;
    tr[2] = globaltimer();
    tr[3] = 0;
  }
  // CTA b starts with item b (blockIdx order spreads the FC1 streams one per
  // SM); the counter hands out the rest
  if (tid == 0) T.next = blockIdx.x;
  __syncthreads();

  const int U = T.n_unique;
  const int per_expert = 2 * a.n_tiles;
  const int fc2_per_expert = a.out_tiles * 2;  // 64-column items
  const int n_fc1 = (a.phase_mask & 1) ? U * per_expert : 0;
  // L2 prefetch depth of the FC1 streams (K64 slices ahead of the ring): 4
  // measured best at 22-36 experts
  if (tid == 0) T.pf_depth = a.pf_depth >= 0 ? a.pf_depth : 4;
  __syncthreads();
  const int n_items = n_fc1 + ((a.phase_mask & 2) ? U * fc2_per_expert : 0);
  const bool no_wait = (a.phase_mask & 1) == 0;
  // FC2 items follow the experts' readiness order (phase_mask 2: the slot
  // order, nothing to wait for); a static share for the CTAs left over during
  // FC1 (weights preloaded while waiting) was measured no better
  const int n_static_experts = no_wait ? U : 0;
  const int n_static = n_static_experts * fc2_per_expert;
  int* slice_done = a.ctrl + kCtrlSliceDone;
  int* act_done = a.ctrl + kCtrlActDone;
  int* ready = a.ctrl + kCtrlReady;
  int* ready_list = a.ctrl + kCtrlReadyList;

  if (U == 0 &&
      blockIdx.x ==
          0) {  // nothing routed: zero every pair's output (padding tokens)
    for (int p = 0; p < T.n_pad; ++p) {
      uint32_t* dst =
          reinterpret_cast<uint32_t*>(a.out + (size_t)T.pad[p] * a.k);
      for (int i = tid; i < a.k / 2; i += kThreads) dst[i] = 0u;
    }
  }

  // FC2 pipeline: the item whose weights are in flight / landed in buffer
  // pend_buf, computed when the next item's weights have been issued (or at the
  // end)
  int pend_item = -1, pend_u = 0, pend_buf = 1;
  bool pend_static = false;
  long long* pend_tr = nullptr;

  auto fc2_finish_pending = [&](int wait_groups) {
    if (pend_item < 0) return;
    if (tid == 0) {
      if (pend_static) {
        unsigned ns = 256;
        while (!no_wait && ld_acquire(ready + pend_u) == 0) {
          __nanosleep(ns);
          ns = min(ns * 2, a.spin_ns_max);
        }
      }
      if (pend_tr) pend_tr[2] = globaltimer();
    }
    if (pend_static)
      __syncthreads();  // the expert's activation is visible (thread 0's
                        // acquire)
    Fc2Regs R;
    fc2_load_a(a, T, pend_u, R);
    if (wait_groups == 0)
      cp_async_wait<0>();
    else
      cp_async_wait<1>();
    __syncthreads();
    if (pend_tr && tid == 0)
      pend_tr[2] =
          globaltimer();  // buffer landed (overwrites the readiness stamp)
    fc2_mma(a, T, smem, pend_u, pend_item, pend_buf, R);
    if (pend_u == 0) {  // zeros for the padding pairs' columns of this item
                        // (b12x's phase 2 does the same)
      for (int p = 0; p < T.n_pad; ++p)
        if (tid < kF2Cols / 2)
          reinterpret_cast<uint32_t*>(a.out + (size_t)T.pad[p] * a.k +
                                      pend_item * kF2Cols)[tid] = 0u;
    }
    if (pend_tr && tid == 0) pend_tr[3] = globaltimer();
    pend_item = -1;
    pend_tr = nullptr;
  };

  for (;;) {
    if (tid == 0) {
      T.item = T.next;
      if (T.next < n_items)
        T.next =
            (int)gridDim.x + atomicAdd(a.ctrl + kCtrlNext,
                                       1);  // the following item, fetched ahead
    }
    __syncthreads();
    const int item = T.item;
    if (item >= n_items) break;
    long long* tr = nullptr;
    if (a.trace != nullptr && tid == 0) {
      const int slot = T.n_trace++;
      if (slot < kTraceItems) {
        tr = a.trace + ((size_t)blockIdx.x * kTraceItems + slot) * 4;
        tr[0] = item;
        tr[1] = globaltimer();
      }
    }
    if (item < n_fc1) {
      const int u = item / per_expert, t = item - u * per_expert;
      const int otile = t % a.n_tiles;
      fc1_tile(a, T, smem, u, t);
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        T.flag = atomicAdd(slice_done + u * kSlices + otile, 1) ==
                 1;  // the slice's gate and up tiles are both done
        if (tr) tr[2] = globaltimer();
      }
      __syncthreads();
      if (T.flag) {
        __threadfence();
        activation_slice(a, T, u, otile);
        __threadfence();
        __syncthreads();
        if (tid == 0 && atomicAdd(act_done + u, 1) == a.n_tiles - 1) {
          st_release(ready + u, 1);
          if (u >= n_static_experts) {
            const int k = atomicAdd(a.ctrl + kCtrlNReady, 1);
            st_release(ready_list + k, u);
          }
        }
      }
      if (tr && tid == 0) tr[3] = globaltimer();
    } else {
      const int j = item - n_fc1;
      int u, li;
      bool is_static;
      if (j < n_static) {
        u = j / fc2_per_expert;
        li = j - u * fc2_per_expert;
        is_static = true;
      } else {
        const int k = (j - n_static) / fc2_per_expert;
        li = (j - n_static) - k * fc2_per_expert;
        if (tid == 0) {
          int v = k;  // no_wait: the slot order
          unsigned ns = 256;
          while (!no_wait && (v = ld_acquire(ready_list + k)) < 0) {
            __nanosleep(ns);
            ns = min(ns * 2, a.spin_ns_max);
          }
          T.flag = v;
        }
        __syncthreads();
        u = T.flag;
        is_static = false;
      }
      const int buf =
          pend_buf ^
          1;  // free: the item before the pending one has been computed
      fc2_issue_w(a, smem, T.expert[u], li, buf);
      fc2_finish_pending(1);
      pend_item = li;
      pend_u = u;
      pend_buf = buf;
      pend_static = is_static;
      pend_tr = tr;
    }
  }
  fc2_finish_pending(0);

  // the last CTA out resets the control block for the next launch (graph
  // replay)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    T.flag = atomicAdd(a.ctrl + kCtrlFinished, 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (T.flag) {
    for (int i = tid; i < kCtrlInts; i += kThreads)
      a.ctrl[i] = i >= kCtrlReadyList ? -1 : 0;
    __threadfence();
  }
}

__global__ void ctrl_init_kernel(int* ctrl) {
  for (int i = threadIdx.x; i < kCtrlInts; i += blockDim.x)
    ctrl[i] = i >= kCtrlReadyList ? -1 : 0;
}

}  // namespace b12x_moe_fused

namespace {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

constexpr int kCtasPerSm = 2;
constexpr int kPrefetchSlices = 4;
constexpr unsigned kSpinNsMax = 512;
constexpr int kMaxDevices = 64;

}  // namespace

void b12x_w4a8_moe_fused_ctrl_init(Tensor& ctrl) {
  STD_TORCH_CHECK(ctrl.scalar_type() == ScalarType::Int &&
                      ctrl.is_contiguous() &&
                      ctrl.numel() >= b12x_moe_fused::kCtrlInts,
                  "ctrl must be a contiguous int32 control block");
  const int device = ctrl.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  b12x_moe_fused::
      ctrl_init_kernel<<<1, 256, 0, get_current_cuda_stream(device)>>>(
          reinterpret_cast<int*>(ctrl.mutable_data_ptr()));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "ctrl init launch failed: ", cudaGetErrorString(err));
}

void b12x_w4a8_moe_fused_sm120(
    const Tensor& a_values, const Tensor& a_scales, const Tensor& w13,
    const Tensor& w13_scales, Tensor& projections, Tensor& intermediate,
    const Tensor& w2, const Tensor& w2_scales, const Tensor& topk_ids,
    const Tensor& topk_weights, const Tensor& alpha1, const Tensor& input_scale,
    const Tensor& alpha2, const Tensor& down_scale, Tensor& route_output,
    Tensor& ctrl, int64_t num_pairs, int64_t topk, int64_t rows_capacity,
    int64_t n, double swiglu_limit) {
  using namespace b12x_moe_fused;
  const int64_t k = route_output.size(1);
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == ScalarType::Int &&
          topk_weights.scalar_type() == ScalarType::Float &&
          topk_ids.is_contiguous() && topk_weights.is_contiguous(),
      "top-k ids must be contiguous int32 and weights contiguous float32");
  STD_TORCH_CHECK(alpha1.scalar_type() == ScalarType::Float &&
                      alpha2.scalar_type() == ScalarType::Float &&
                      input_scale.scalar_type() == ScalarType::Float &&
                      down_scale.scalar_type() == ScalarType::Float,
                  "alphas and scales must be float32");
  STD_TORCH_CHECK(
      route_output.scalar_type() == ScalarType::BFloat16 &&
          route_output.dim() == 2 && route_output.is_contiguous(),
      "route_output must be a contiguous bf16 [rows, hidden] tensor");
  // FC1 copies scales in 16-byte vectors into each fixed-size row slot.
  // The minimum also covers its seven unconditionally prefetched K64 stages.
  STD_TORCH_CHECK(k >= 512 && k <= 32 * kF1RowScales && k % 512 == 0,
                  "hidden must be a multiple of 512 in [512, 8192]");
  STD_TORCH_CHECK(n == 576, "intermediate 576 (five K128 slices, n64 tail)");
  STD_TORCH_CHECK(0 <= num_pairs && num_pairs <= rows_capacity &&
                      rows_capacity <= kMaxPairs &&
                      route_output.size(0) >= rows_capacity,
                  "pairs exceed the fused kernel's capacity");
  STD_TORCH_CHECK(
      ctrl.scalar_type() == ScalarType::Int && ctrl.numel() >= kCtrlInts,
      "ctrl must be an int32 control block");
  STD_TORCH_CHECK(
      alpha1.numel() <= kMaxExpertIds && alpha1.numel() == alpha2.numel(),
      "at most 384 experts");
  STD_TORCH_CHECK(w2.numel() * w2.element_size() == alpha2.numel() * k * n / 2,
                  "w2 is not the n64-repacked storage");
  STD_TORCH_CHECK(
      input_scale.numel() == 1 || input_scale.numel() == alpha1.numel(),
      "input scale must be scalar or per expert");
  STD_TORCH_CHECK(
      down_scale.numel() == 1 || down_scale.numel() == alpha1.numel(),
      "down scale must be scalar or per expert");
  const Tensor* operands[] = {
      &a_values,     &a_scales,   &w13,       &w13_scales, &projections,
      &intermediate, &w2,         &w2_scales, &alpha1,     &alpha2,
      &input_scale,  &down_scale, &ctrl};
  for (const Tensor* t : operands) {
    STD_TORCH_CHECK(t->is_contiguous(),
                    "fused MoE operands must be contiguous");
  }
  Args a;
  a.a_values = reinterpret_cast<const uint32_t*>(a_values.const_data_ptr());
  a.a_scales = reinterpret_cast<const uint8_t*>(a_scales.const_data_ptr());
  a.w13 = reinterpret_cast<const uint32_t*>(w13.const_data_ptr());
  a.w13_sf = reinterpret_cast<const uint32_t*>(w13_scales.const_data_ptr());
  a.projections =
      reinterpret_cast<__nv_bfloat16*>(projections.mutable_data_ptr());
  a.intermediate = reinterpret_cast<uint32_t*>(intermediate.mutable_data_ptr());
  a.w2 = reinterpret_cast<const uint32_t*>(w2.const_data_ptr());
  a.w2_sf = reinterpret_cast<const uint8_t*>(w2_scales.const_data_ptr());
  a.ids = reinterpret_cast<const int32_t*>(topk_ids.const_data_ptr());
  a.token_weights =
      reinterpret_cast<const float*>(topk_weights.const_data_ptr());
  a.alpha1 = reinterpret_cast<const float*>(alpha1.const_data_ptr());
  a.input_scale = reinterpret_cast<const float*>(input_scale.const_data_ptr());
  a.alpha2 = reinterpret_cast<const float*>(alpha2.const_data_ptr());
  a.down_scale = reinterpret_cast<const float*>(down_scale.const_data_ptr());
  a.out = reinterpret_cast<__nv_bfloat16*>(route_output.mutable_data_ptr());
  a.ctrl = reinterpret_cast<int*>(ctrl.mutable_data_ptr());
  a.trace = nullptr;
  a.limit = static_cast<float>(swiglu_limit);
  a.is_per_expert = input_scale.numel() > 1;
  a.ds_per_expert = down_scale.numel() > 1;
  a.num_pairs = static_cast<int>(num_pairs);
  a.topk = static_cast<int>(topk);
  a.rows_capacity = static_cast<int>(rows_capacity);
  a.k = static_cast<int>(k);
  a.n = static_cast<int>(n);
  a.n_tiles = static_cast<int>((n + kTileN - 1) / kTileN);
  a.out_tiles = static_cast<int>(k / kTileN);
  a.num_experts = static_cast<int>(alpha1.numel());
  a.pf_depth = kPrefetchSlices;
  a.phase_mask = 3;
  a.spin_ns_max = kSpinNsMax;
  if (num_pairs == 0) return;

  const int device = route_output.get_device_index();
  STD_TORCH_CHECK(device >= 0 && device < kMaxDevices,
                  "unexpected device index");
  const torch::stable::accelerator::DeviceGuard guard(device);
  // Every CTA must be able to become resident: FC2 items wait for FC1 items of
  // other CTAs. The grid is the per-device resident capacity, computed once.
  static int grid_by_device[kMaxDevices] = {};
  int& grid = grid_by_device[device];
  if (grid == 0) {
    cudaError_t err = cudaFuncSetAttribute(
        moe_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    STD_TORCH_CHECK(
        err == cudaSuccess,
        "fused MoE shared memory opt-in failed: ", cudaGetErrorString(err));
    int sms = 0;
    int resident = 0;
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "SM count query failed: ", cudaGetErrorString(err));
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, moe_fused_kernel, kThreads, kSmem);
    STD_TORCH_CHECK(err == cudaSuccess && resident >= 1,
                    "fused MoE kernel does not fit on this GPU");
    grid = sms * std::min(kCtasPerSm, resident);
  }
  moe_fused_kernel<<<grid, kThreads, kSmem, get_current_cuda_stream(device)>>>(
      a);
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "fused MoE launch failed: ", cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("b12x_w4a8_moe_fused_ctrl_init(Tensor! ctrl) -> ()");
  ops.def(
      "b12x_w4a8_moe_fused_sm120(Tensor a_values, Tensor a_scales, "
      "Tensor w13, Tensor w13_scales, Tensor! projections, "
      "Tensor! intermediate, Tensor w2, Tensor w2_scales, Tensor topk_ids, "
      "Tensor topk_weights, Tensor alpha1, Tensor input_scale, Tensor alpha2, "
      "Tensor down_scale, Tensor! route_output, Tensor! ctrl, int num_pairs, "
      "int topk, int rows_capacity, int n, float swiglu_limit) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("b12x_w4a8_moe_fused_ctrl_init",
           TORCH_BOX(&b12x_w4a8_moe_fused_ctrl_init));
  ops.impl("b12x_w4a8_moe_fused_sm120", TORCH_BOX(&b12x_w4a8_moe_fused_sm120));
}

#endif
