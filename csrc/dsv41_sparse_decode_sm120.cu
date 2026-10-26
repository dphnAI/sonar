// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4.1 sparse MLA decode on SM120: one launch per step, an 8-CTA
// cluster per query row. Each CTA decodes its share of the SWA (FP8 + UE8M0)
// and indexed (E2M1 + E4M3) keys into BF16 shared memory, scores and weights
// them with BF16 tensor-core MMAs, and the cluster merges the eight partial
// softmax states with the attention sink in a fixed order.
//
// Adapted from sparse_decode.cu in DeepSeek-4.1-Flash-1017-BRRR
// (https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR),
// Copyright 2026 Fraser Price, licensed under the Apache License, Version 2.0.
// Changes: the kernel reads the SWA slots, indexed top-k and per-row indexed
// page table the b12x decode path uses, so window and replay semantics are the
// caller's; indexed records load 8 bytes per lane (a 16-byte load at an 8-byte
// offset is misaligned); split score tiles are summed in a fixed warp order
// instead of with shared-memory atomics, so results are deterministic.

#include "dsv41_sparse_decode_sm120.h"
#include "torch_utils.h"

#if defined(APHRODITE_ENABLE_SM120_SPARSE_DECODE) && !defined(USE_ROCM)

  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <torch/csrc/stable/accelerator.h>
  #include <torch/headeronly/core/ScalarType.h>

  #include <cmath>
  #include <cstdint>

namespace {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

constexpr int H = 16;
constexpr int D = 512;
constexpr int CL = 8;
constexpr int MAXK = 80;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int PS = MAXK + 8;
constexpr int SPLIT = WARPS / 2;
constexpr int SWA_BYTES = 528;
constexpr int IDX_BYTES = 288;
constexpr float LOG2E = 1.4426950408889634f;

struct Args {
  const __nv_bfloat16* q;
  __nv_bfloat16* out;
  const uint8_t* swa;
  int64_t swa_page_stride;
  int swa_page;
  const int* swa_indices;
  int64_t swa_indices_stride;
  const int* swa_lengths;
  int swa_width;
  const uint8_t* idx;
  int64_t idx_page_stride;
  int idx_page;
  int64_t idx_pages;
  const int* idx_logical;
  int64_t idx_logical_stride;
  const int* idx_lengths;
  const int* idx_table;
  int64_t idx_table_stride;
  int idx_table_width;
  int idx_width;
  int per_cta;
  const float* sink;
  float scale;
  float* partial;
  float2* stats;
};

struct Smem {
  __nv_bfloat16 q[H * D];
  __nv_bfloat16 k[MAXK * D];
  float red[WARPS][H];
  union {
    // Score sums of the two spill tiles.
    float extra[2][H][8];
    float w[CL][H];
  };
  float m[H];
  float l[H];
  int slots[MAXK];
  int64_t offsets[MAXK];
};

__device__ __forceinline__ int swz(int row, int col) {
  return row * D + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)),
               "l"(src)
               : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0,
                                         uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile(
      "barrier.cluster.arrive.release.aligned;\n"
      "barrier.cluster.wait.acquire.aligned;" ::
          : "memory");
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float e8m0(uint32_t e) {
  return __uint_as_float(e << 23);
}

__device__ __forceinline__ float2 half2_float2(uint32_t h) {
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t v) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
      : "=r"(h)
      : "h"(static_cast<unsigned short>(v)));
  return h;
}

__device__ __forceinline__ float2 e2m1x2(uint8_t v) {
  uint32_t h;
  asm("{\n .reg .b8 t;\n mov.b16 {t, _}, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}"
      : "=r"(h)
      : "h"(static_cast<uint16_t>(v)));
  return half2_float2(h);
}

// Physical slot of key `k` (SWA keys first, then indexed), or -1 when masked.
__device__ __forceinline__ int slot_of(const Args& a, int row, int k) {
  if (k < a.swa_width) {
    const int length = min(a.swa_lengths[row], a.swa_width);
    return k < length ? a.swa_indices[row * a.swa_indices_stride + k] : -1;
  }
  k -= a.swa_width;
  if (k >= a.idx_width || k >= min(a.idx_lengths[row], a.idx_width)) return -1;
  const int64_t logical = a.idx_logical[row * a.idx_logical_stride + k];
  if (logical < 0) return -1;
  const int64_t page = logical / a.idx_page;
  if (page >= a.idx_table_width) return -1;
  const int64_t pid = a.idx_table[row * a.idx_table_stride + page];
  if (pid < 0 || pid >= a.idx_pages) return -1;
  return static_cast<int>(pid * a.idx_page + logical % a.idx_page);
}

__device__ __forceinline__ void decode_keys(const Args& a, Smem& sm, int first,
                                            int warp, int lane) {
  constexpr int PER = MAXK / WARPS;
  const int per = a.per_cta / WARPS;
  uint4 raw[PER];
  uint8_t scale[PER];
  #pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int key = warp * per + i;
    const int slot = i < per ? sm.slots[key] : -1;
    const bool swa = first + key < a.swa_width;
    const uint8_t* rec = (swa ? a.swa : a.idx) + sm.offsets[key];
    raw[i] = make_uint4(0, 0, 0, 0);
    scale[i] = 0;
    if (slot >= 0) {
      if (swa) {
        raw[i] = __ldg(reinterpret_cast<const uint4*>(rec + 16 * lane));
        scale[i] = __ldg(rec + D + lane / 2);
      } else {
        // 16 E2M1 values per lane are 8 bytes, so only 8-byte alignment holds.
        const uint2 v = __ldg(reinterpret_cast<const uint2*>(rec + 8 * lane));
        raw[i] = make_uint4(v.x, v.y, 0, 0);
        scale[i] = __ldg(rec + D / 2 + lane);
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < PER; ++i) {
    if (i >= per) break;
    const int key = warp * per + i;
    uint32_t o[8];
    if (first + key < a.swa_width) {
      const float s = e8m0(scale[i]);
      const uint16_t* v = reinterpret_cast<const uint16_t*>(&raw[i]);
  #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 f = half2_float2(e4m3x2_f16x2(v[j]));
        o[j] = pack_bf16(f.x * s, f.y * s);
      }
    } else {
      const float s = half2_float2(e4m3x2_f16x2(scale[i])).x;
      const uint8_t* v = reinterpret_cast<const uint8_t*>(&raw[i]);
  #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 f = e2m1x2(v[j]);
        o[j] = pack_bf16(f.x * s, f.y * s);
      }
    }
    *reinterpret_cast<uint4*>(sm.k + swz(key, lane * 16)) =
        make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(sm.k + swz(key, lane * 16 + 8)) =
        make_uint4(o[4], o[5], o[6], o[7]);
  }
}

template <class Op>
__device__ __forceinline__ float across_warps(Smem& sm, int warp, int lane,
                                              int tid, float lo, float hi,
                                              float init, Op op) {
  const int g = lane / 4;
  if (lane % 4 == 0) sm.red[warp][g] = lo, sm.red[warp][g + 8] = hi;
  __syncthreads();
  float v = init;
  if (tid < H) {
  #pragma unroll
    for (int w = 0; w < WARPS; ++w) v = op(v, sm.red[w][tid]);
  }
  return v;
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THREADS, 1)
    sparse_decode(Args a) {
  extern __shared__ __align__(128) uint8_t raw_smem[];
  Smem& sm = *reinterpret_cast<Smem*>(raw_smem);
  const int rank = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int keys = a.per_cta, first = rank * keys, tiles = keys / 8;
  asm volatile("griddepcontrol.launch_dependents;");
  // Every input may come from the previous kernel on the stream.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  for (int i = tid; i < H * D / 8; i += THREADS) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    cp_async16(sm.q + swz(h, c), a.q + (int64_t(row) * H + h) * D + c);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  if (tid < keys) {
    const int slot = slot_of(a, row, first + tid);
    const bool swa = first + tid < a.swa_width;
    const int page = swa ? a.swa_page : a.idx_page;
    sm.slots[tid] = slot;
    sm.offsets[tid] =
        slot < 0 ? 0
                 : int64_t(slot / page) *
                           (swa ? a.swa_page_stride : a.idx_page_stride) +
                       int64_t(slot % page) * (swa ? SWA_BYTES : IDX_BYTES);
  }
  __syncthreads();
  decode_keys(a, sm, first, warp, lane);
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  float s[2][4] = {};
  const auto score = [&](float (&acc)[4], int tile, int k0, int k1) {
    float odd[4] = {};
  #pragma unroll 4
    for (int kk = k0; kk < k1; kk += 2) {
      uint32_t qa[4], kb[2], qb[4], kc[2];
      ldsm_x4(qa, sm.q + swz(lane % 16, kk * 16 + (lane / 16) * 8));
      ldsm_x2(kb,
              sm.k + swz(tile * 8 + lane % 8, kk * 16 + ((lane / 8) % 2) * 8));
      ldsm_x4(qb, sm.q + swz(lane % 16, (kk + 1) * 16 + (lane / 16) * 8));
      ldsm_x2(kc, sm.k + swz(tile * 8 + lane % 8,
                             (kk + 1) * 16 + ((lane / 8) % 2) * 8));
      mma_bf16(acc, qa[0], qa[1], qa[2], qa[3], kb[0], kb[1]);
      mma_bf16(odd, qb[0], qb[1], qb[2], qb[3], kc[0], kc[1]);
    }
  #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += odd[i];
  };
  if (warp < tiles) score(s[0], warp, 0, D / 16);
  // Tiles past one per warp are split along D across SPLIT warps each. Their
  // partial scores are added in warp order, one warp per barrier interval.
  const int spill = tiles - WARPS, quarter = D / 16 / SPLIT;
  const int col = 2 * (lane % 4);
  const int t = warp / SPLIT, j = warp % SPLIT;
  const bool spills = spill > 0 && WARPS + t < tiles;
  float part[4] = {};
  if (spills) score(part, WARPS + t, j * quarter, (j + 1) * quarter);
  #pragma unroll
  for (int turn = 0; turn < SPLIT; ++turn) {
    if (spills && j == turn) {
      float* lo = &sm.extra[t][lane / 4][col];
      float* hi = &sm.extra[t][lane / 4 + 8][col];
      lo[0] = turn ? lo[0] + part[0] : part[0];
      lo[1] = turn ? lo[1] + part[1] : part[1];
      hi[0] = turn ? hi[0] + part[2] : part[2];
      hi[1] = turn ? hi[1] + part[3] : part[3];
    }
    __syncthreads();
  }
  if (warp < spill) {
    s[1][0] = sm.extra[warp][lane / 4][col];
    s[1][1] = sm.extra[warp][lane / 4][col + 1];
    s[1][2] = sm.extra[warp][lane / 4 + 8][col];
    s[1][3] = sm.extra[warp][lane / 4 + 8][col + 1];
  }
  const int g = lane / 4;
  const float scale = a.scale * LOG2E;
  float mx0 = -INFINITY, mx1 = -INFINITY;
  #pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = warp + t * WARPS, key = tile * 8 + 2 * (lane % 4);
    const bool live = tile < tiles && (t == 0 || warp < spill);
    const bool v0 = live && sm.slots[key] >= 0;
    const bool v1 = live && sm.slots[key + 1] >= 0;
    s[t][0] = v0 ? s[t][0] * scale : -INFINITY;
    s[t][1] = v1 ? s[t][1] * scale : -INFINITY;
    s[t][2] = v0 ? s[t][2] * scale : -INFINITY;
    s[t][3] = v1 ? s[t][3] * scale : -INFINITY;
    mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
    mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
  }
  const float head_max =
      across_warps(sm, warp, lane, tid, quad_max(mx0), quad_max(mx1), -INFINITY,
                   [](float x, float y) { return fmaxf(x, y); });
  if (tid < H) sm.m[tid] = head_max;
  __syncthreads();
  // The query tile is no longer needed; it holds the BF16 probabilities.
  __nv_bfloat16* p = sm.q;
  const float m0 = sm.m[g], m1 = sm.m[g + 8];
  float l0 = 0.0f, l1 = 0.0f;
  #pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = warp + t * WARPS, key = tile * 8 + 2 * (lane % 4);
    if (tile >= tiles) break;
    const auto e = [](float v, float m) {
      return v == -INFINITY ? 0.0f : exp2f(v - m);
    };
    const uint32_t pb0 = pack_bf16(e(s[t][0], m0), e(s[t][1], m0));
    const uint32_t pb1 = pack_bf16(e(s[t][2], m1), e(s[t][3], m1));
    *reinterpret_cast<uint32_t*>(p + g * PS + key) = pb0;
    *reinterpret_cast<uint32_t*>(p + (g + 8) * PS + key) = pb1;
    // The softmax denominator sums the rounded weights the MMA consumes.
    const __nv_bfloat162 r0 = *reinterpret_cast<const __nv_bfloat162*>(&pb0);
    const __nv_bfloat162 r1 = *reinterpret_cast<const __nv_bfloat162*>(&pb1);
    l0 += __low2float(r0) + __high2float(r0);
    l1 += __low2float(r1) + __high2float(r1);
  }
  __syncthreads();
  const float head_sum =
      across_warps(sm, warp, lane, tid, quad_sum(l0), quad_sum(l1), 0.0f,
                   [](float x, float y) { return x + y; });
  if (tid < H) sm.l[tid] = head_sum;

  float o[8][4] = {};
  for (int ks = 0; ks < keys / 16; ++ks) {
    uint32_t pa[4];
    ldsm_x4(pa, p + (lane % 16) * PS + ks * 16 + (lane / 16) * 8);
  #pragma unroll
    for (int nt = 0; nt < 8; nt += 2) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sm.k + swz(ks * 16 + (lane % 8) + ((lane / 8) % 2) * 8,
                               warp * 64 + nt * 8 + (lane / 16) * 8));
      mma_bf16(o[nt], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
      mma_bf16(o[nt + 1], pa[0], pa[1], pa[2], pa[3], vb[2], vb[3]);
    }
  }
  float* po = a.partial + (int64_t(row) * CL + rank) * H * D;
  #pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int d = warp * 64 + nt * 8 + 2 * (lane % 4);
    *reinterpret_cast<float2*>(po + g * D + d) =
        make_float2(o[nt][0], o[nt][1]);
    *reinterpret_cast<float2*>(po + (g + 8) * D + d) =
        make_float2(o[nt][2], o[nt][3]);
  }
  if (tid < H) {
    a.stats[(int64_t(row) * CL + rank) * H + tid] =
        make_float2(sm.m[tid], sm.l[tid]);
  }
  cluster_sync();

  // Merge the cluster's partial states and the sink in CTA order.
  if (tid < H) {
    float2 ml[CL];
  #pragma unroll
    for (int c = 0; c < CL; ++c) {
      ml[c] = __ldcg(a.stats + (int64_t(row) * CL + c) * H + tid);
    }
    float mx = a.sink[tid] * LOG2E;
    const float sink = mx;
  #pragma unroll
    for (int c = 0; c < CL; ++c) mx = fmaxf(mx, ml[c].x);
    float total = mx == -INFINITY ? 0.0f : exp2f(sink - mx);
  #pragma unroll
    for (int c = 0; c < CL; ++c) {
      const float w = ml[c].x == -INFINITY ? 0.0f : exp2f(ml[c].x - mx);
      sm.w[c][tid] = w;
      total += w * ml[c].y;
    }
    sm.red[0][tid] = total > 0.0f ? 1.0f / total : 0.0f;
  }
  __syncthreads();
  constexpr int SLICE = D / CL / 4;
  const int h = tid / SLICE, d4 = rank * SLICE + tid % SLICE;
  float4 v[CL];
  #pragma unroll
  for (int c = 0; c < CL; ++c) {
    v[c] = __ldcg(reinterpret_cast<const float4*>(
                      a.partial + ((int64_t(row) * CL + c) * H + h) * D) +
                  d4);
  }
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  #pragma unroll
  for (int c = 0; c < CL; ++c) {
    const float w = sm.w[c][h];
    acc.x += w * v[c].x, acc.y += w * v[c].y, acc.z += w * v[c].z,
        acc.w += w * v[c].w;
  }
  const float inv = sm.red[0][h];
  *reinterpret_cast<uint2*>(a.out + (int64_t(row) * H + h) * D + d4 * 4) =
      make_uint2(pack_bf16(acc.x * inv, acc.y * inv),
                 pack_bf16(acc.z * inv, acc.w * inv));
  cluster_sync();
}

void check_int_rows(const Tensor& t, int64_t rows, const char* name) {
  STD_TORCH_CHECK(t.scalar_type() == ScalarType::Int && t.dim() >= 1 &&
                      t.size(0) >= rows && t.stride(t.dim() - 1) == 1,
                  name, " must be int32 rows with a unit inner stride");
}

}  // namespace

void dsv41_sparse_decode_sm120(
    const Tensor& q, Tensor& out, const Tensor& swa_cache,
    int64_t swa_page_size, const Tensor& swa_indices, const Tensor& swa_lengths,
    const std::optional<Tensor>& indexed_cache, int64_t indexed_page_size,
    const std::optional<Tensor>& indexed_indices,
    const std::optional<Tensor>& indexed_lengths,
    const std::optional<Tensor>& indexed_page_table, const Tensor& attn_sink,
    double sm_scale, Tensor& partial, Tensor& stats) {
  const int64_t rows = q.size(0);
  STD_TORCH_CHECK(q.dim() == 3 && q.size(1) == H && q.size(2) == D &&
                      q.scalar_type() == ScalarType::BFloat16 &&
                      q.is_contiguous(),
                  "q must be contiguous BF16 [rows, 16, 512]");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == rows && out.size(1) == H &&
                      out.size(2) == D &&
                      out.scalar_type() == ScalarType::BFloat16 &&
                      out.is_contiguous(),
                  "out must match q");
  STD_TORCH_CHECK(attn_sink.scalar_type() == ScalarType::Float &&
                      attn_sink.numel() == H && attn_sink.is_contiguous(),
                  "attn_sink must be float32 [16]");
  STD_TORCH_CHECK(swa_cache.scalar_type() == ScalarType::Byte &&
                      swa_cache.dim() == 2 && swa_cache.stride(1) == 1 &&
                      swa_cache.size(1) >= swa_page_size * SWA_BYTES &&
                      swa_page_size > 0,
                  "swa_cache must be uint8 [pages, >= page_size * 528]");
  check_int_rows(swa_indices, rows, "swa_indices");
  check_int_rows(swa_lengths, rows, "swa_lengths");
  const bool indexed = indexed_cache.has_value();
  STD_TORCH_CHECK(indexed == indexed_indices.has_value() &&
                      indexed == indexed_lengths.has_value() &&
                      indexed == indexed_page_table.has_value(),
                  "indexed cache, indices, lengths and page table go together");

  Args a{};
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr());
  a.out = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  a.swa = reinterpret_cast<const uint8_t*>(swa_cache.const_data_ptr());
  a.swa_page_stride = swa_cache.stride(0);
  a.swa_page = static_cast<int>(swa_page_size);
  a.swa_indices = reinterpret_cast<const int*>(swa_indices.const_data_ptr());
  a.swa_indices_stride = swa_indices.stride(0);
  a.swa_lengths = reinterpret_cast<const int*>(swa_lengths.const_data_ptr());
  a.swa_width = static_cast<int>(swa_indices.size(1));
  if (indexed) {
    const Tensor& cache = *indexed_cache;
    STD_TORCH_CHECK(cache.scalar_type() == ScalarType::Byte &&
                        cache.dim() == 2 && cache.stride(1) == 1 &&
                        cache.size(1) >= indexed_page_size * IDX_BYTES &&
                        indexed_page_size > 0,
                    "indexed_cache must be uint8 [pages, >= page_size * 288]");
    check_int_rows(*indexed_indices, rows, "indexed_indices");
    check_int_rows(*indexed_lengths, rows, "indexed_lengths");
    check_int_rows(*indexed_page_table, rows, "indexed_page_table");
    a.idx = reinterpret_cast<const uint8_t*>(cache.const_data_ptr());
    a.idx_page_stride = cache.stride(0);
    a.idx_page = static_cast<int>(indexed_page_size);
    a.idx_pages = cache.size(0);
    a.idx_logical =
        reinterpret_cast<const int*>(indexed_indices->const_data_ptr());
    a.idx_logical_stride = indexed_indices->stride(0);
    a.idx_lengths =
        reinterpret_cast<const int*>(indexed_lengths->const_data_ptr());
    a.idx_table =
        reinterpret_cast<const int*>(indexed_page_table->const_data_ptr());
    a.idx_table_stride = indexed_page_table->stride(0);
    a.idx_table_width = static_cast<int>(indexed_page_table->size(1));
    a.idx_width = static_cast<int>(indexed_indices->size(1));
  }
  const int total = a.swa_width + a.idx_width;
  STD_TORCH_CHECK(total > 0 && total % (CL * 16) == 0 && total / CL <= MAXK,
                  "SWA plus indexed width must be a positive multiple of 128 "
                  "and at most 640");
  a.per_cta = total / CL;
  STD_TORCH_CHECK(partial.scalar_type() == ScalarType::Float &&
                      partial.is_contiguous() &&
                      partial.numel() >= rows * CL * H * D,
                  "partial must hold rows * 8 * 16 * 512 float32");
  STD_TORCH_CHECK(stats.scalar_type() == ScalarType::Float &&
                      stats.is_contiguous() &&
                      stats.numel() >= rows * CL * H * 2,
                  "stats must hold rows * 8 * 16 * 2 float32");
  a.partial = reinterpret_cast<float*>(partial.mutable_data_ptr());
  a.stats = reinterpret_cast<float2*>(stats.mutable_data_ptr());
  a.sink = reinterpret_cast<const float*>(attn_sink.const_data_ptr());
  a.scale = static_cast<float>(sm_scale);
  if (rows == 0) return;
  STD_TORCH_CHECK(rows <= 65535, "too many decode rows");

  const int device = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  // The opt-in is a per-device function attribute.
  const cudaError_t opted_in = cudaFuncSetAttribute(
      sparse_decode, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(Smem)));
  STD_TORCH_CHECK(opted_in == cudaSuccess,
                  "sparse decode shared memory opt-in failed");
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(CL, static_cast<unsigned>(rows));
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = sizeof(Smem);
  config.stream = get_current_cuda_stream(device);
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&config, sparse_decode, a);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "sparse decode launch failed: ", cudaGetErrorString(err));
}

#endif
