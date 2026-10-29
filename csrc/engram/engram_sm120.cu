// SPDX-License-Identifier: Apache-2.0

#include "engram_sm120.h"
#include "torch_utils.h"

#if defined(APHRODITE_ENABLE_SM120_ENGRAM) && !defined(USE_ROCM)

  #include <cuda_bf16.h>
  #include <torch/csrc/stable/library.h>

  #include <cmath>
  #include <cstdint>
  #include <limits>
  #include <type_traits>

namespace {
constexpr int H = 5120, L = 1280, PACK = 2624;
static_assert(PACK == 64 + L * sizeof(__nv_bfloat16));
static_assert(PACK % 64 == 0);
using BF = __nv_bfloat16;
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

struct Strides {
  int64_t token, head, feature;
};

__device__ __forceinline__ float mul(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ __forceinline__ float add(float a, float b) {
  return __fadd_rn(a, b);
}
__device__ __forceinline__ float source_div(float a, float b) {
  float c;
  asm("div.full.f32 %0,%1,%2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float source_rsqrt(float a) {
  float c;
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(c) : "f"(a));
  return c;
}
__device__ __forceinline__ float source_sqrt(float a) {
  float c;
  asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(c) : "f"(a));
  return c;
}
__device__ __forceinline__ float source_exp2(float a) {
  float c;
  asm("ex2.approx.f32 %0,%1;" : "=f"(c) : "f"(a));
  return c;
}
__device__ __forceinline__ float source_max(float a, float b) {
  // Triton postprocessing uses maxnumf / max.f32, not max.NaN.f32.
  float c;
  asm("max.f32 %0,%1,%2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

// Preserve the source PTX's per-lane FMA boundaries and XOR reduction order.
__device__ __forceinline__ float lane_sum(const float* a, const float* b) {
  float p[8], q[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int r = (j / 2) * 8 + (j % 2) * 4;
    p[j] = __fmaf_rn(a[r], b[r], mul(a[r + 2], b[r + 2]));
    q[j] = __fmaf_rn(a[r + 1], b[r + 1], mul(a[r + 3], b[r + 3]));
  }
  const float even = add(add(add(p[0], p[1]), add(p[2], p[3])),
                         add(add(p[4], p[5]), add(p[6], p[7])));
  const float odd = add(add(add(q[0], q[1]), add(q[2], q[3])),
                        add(add(q[4], q[5]), add(q[6], q[7])));
  float s = add(even, odd);
  #pragma unroll
  for (int delta = 16; delta; delta /= 2)
    s = add(s, __shfl_xor_sync(0xffffffff, s, delta));
  return s;
}

__device__ __forceinline__ int local_feature(int d) {
  return (d / 2048) * 512 + ((d % 2048) / 1024) * 256 + d % 256;
}

template <bool Contiguous>
__device__ __forceinline__ int64_t hidden_offset(int64_t t, int hc, int d,
                                                 Strides strides) {
  if constexpr (Contiguous) return (t * 4 + hc) * H + d;
  return t * strides.token + hc * strides.head + d * strides.feature;
}

template <bool Contiguous>
__global__ void sm120_engram_pack_kernel(const BF* hidden, const BF* local,
                                         const BF* q, const BF* k,
                                         unsigned char* packet, int rank,
                                         int64_t valid,
                                         Strides hidden_strides) {
  const int64_t t = blockIdx.x;
  const int hc = blockIdx.y, half = blockIdx.z, lane = threadIdx.x;
  const int warp = rank + 4 * half;
  float key[32], coeff[32];
  #pragma unroll
  for (int round = 0; round < 4; ++round) {
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int i = round * 8 + j;
      const int d = round * 2048 + warp * 256 + lane * 8 + j;
      const int f = round * 512 + half * 256 + lane * 8 + j;
      float v = 0, h = 0, qv = 0, kv = 0;
      if (d < H) {
        h = __bfloat162float(
            hidden[hidden_offset<Contiguous>(t, hc, d, hidden_strides)]);
        qv = __bfloat162float(q[hc * H + d]);
        kv = __bfloat162float(k[hc * H + d]);
        if (t < valid) v = __bfloat162float(local[t * 6400 + hc * L + f]);
        if (hc == 0) {
          BF value = __float2bfloat16_rn(0.f);
          if (t < valid) value = local[t * 6400 + 4 * L + f];
          reinterpret_cast<BF*>(packet + t * PACK + 64)[f] = value;
        }
      }
      key[i] = v;
      coeff[i] = mul(mul(h, qv), kv);
    }
  }
  const float square = lane_sum(key, key), dot = lane_sum(coeff, key);
  if (lane == 0) {
    float* header = reinterpret_cast<float*>(packet + t * PACK);
    header[hc * 4 + half * 2] = square;
    header[hc * 4 + half * 2 + 1] = dot;
  }
}

template <bool Contiguous>
__global__ void sm120_engram_finish_kernel(
    const BF* hidden, const unsigned char* packets, const bool* mask,
    BF* output, int64_t tokens, int64_t valid, float eps, float clamp,
    Strides hidden_strides, Strides output_strides, int64_t mask_stride,
    bool has_mask) {
  const int64_t t = blockIdx.x;
  const int hc = blockIdx.y, lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  float h[32];
  #pragma unroll
  for (int round = 0; round < 4; ++round) {
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = round * 2048 + warp * 256 + lane * 8 + j;
      h[round * 8 + j] =
          d < H
              ? __bfloat162float(
                    hidden[hidden_offset<Contiguous>(t, hc, d, hidden_strides)])
              : 0.f;
    }
  }
  const float hs = lane_sum(h, h);
  __shared__ float hidden_warp[8];
  if (lane == 0) hidden_warp[warp] = hs;
  __syncthreads();
  // Match the source shared-load layout, then XOR 4, 2, 1 in every warp.
  const int sw = lane % 8, rank = sw % 4, half = sw / 4;
  const float* header =
      reinterpret_cast<const float*>(packets + (rank * tokens + t) * PACK);
  float square = header[hc * 4 + half * 2];
  float dot = header[hc * 4 + half * 2 + 1], hidden_sum = hidden_warp[sw];
  #pragma unroll
  for (int delta = 4; delta; delta /= 2) {
    hidden_sum =
        add(hidden_sum, __shfl_xor_sync(0xffffffff, hidden_sum, delta));
    square = add(square, __shfl_xor_sync(0xffffffff, square, delta));
    dot = add(dot, __shfl_xor_sync(0xffffffff, dot, delta));
  }
  const float hr = source_rsqrt(add(eps, source_div(hidden_sum, 5120.f)));
  const float kr = source_rsqrt(add(eps, source_div(square, 5120.f)));
  const float normalized =
      mul(mul(mul(hr, kr), __uint_as_float(0x3c64f92e)), dot);
  const float root = source_sqrt(source_max(fabsf(normalized), clamp));
  const float neg_gate_input = normalized < 0.f ? root : -root;
  float gate = source_div(
      1.f,
      add(source_exp2(mul(neg_gate_input, __uint_as_float(0x3fb8aa3b))), 1.f));
  if (has_mask && !(t < valid && mask[t * mask_stride])) gate = 0.f;
  #pragma unroll
  for (int round = 0; round < 4; ++round) {
  #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = round * 2048 + warp * 256 + lane * 8 + j;
      if (d < H) {
        const int owner = (d % 1024) / 256, f = local_feature(d);
        const BF value = reinterpret_cast<const BF*>(
            packets + (owner * tokens + t) * PACK + 64)[f];
        output[hidden_offset<Contiguous>(t, hc, d, output_strides)] =
            __float2bfloat16_rn(
                __fmaf_rn(gate, __bfloat162float(value), h[round * 8 + j]));
      }
    }
  }
}

struct Region {
  uintptr_t begin, end;
};

Region check_tensor(const Tensor& tensor, ScalarType dtype, int32_t device,
                    const char* name) {
  STD_TORCH_CHECK(
      tensor.device().is_cuda() && tensor.get_device_index() == device, name,
      " must be on the hidden CUDA device");
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  int64_t span = 0;
  constexpr int64_t limit = std::numeric_limits<int64_t>::max();
  for (int64_t d = 0; d < tensor.dim(); ++d) {
    const int64_t stride = tensor.stride(d);
    STD_TORCH_CHECK(stride >= 0, name, " must have nonnegative strides");
    const int64_t count = tensor.size(d) > 0 ? tensor.size(d) - 1 : 0;
    STD_TORCH_CHECK(count == 0 || stride <= (limit - span) / count, name,
                    " strides overflow");
    span += count * stride;
  }
  const auto begin = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  const int64_t element_size = tensor.element_size();
  STD_TORCH_CHECK(tensor.numel() == 0 || begin % element_size == 0, name,
                  " is misaligned");
  STD_TORCH_CHECK(span < limit / element_size, name, " byte span overflows");
  const auto bytes = tensor.numel() == 0
                         ? uintptr_t{0}
                         : static_cast<uintptr_t>(span + 1) * element_size;
  STD_TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin, name,
                  " address span overflows");
  return {begin, begin + bytes};
}

void check_disjoint(Region output, Region input) {
  STD_TORCH_CHECK(output.begin == output.end || input.begin == input.end ||
                      output.end <= input.begin || input.end <= output.begin,
                  "Engram destination must not overlap an input");
}

void check_hidden(const Tensor& hidden, int64_t valid) {
  STD_TORCH_CHECK(hidden.device().is_cuda(), "hidden must be CUDA");
  STD_TORCH_CHECK(
      hidden.dim() == 3 && hidden.size(1) == 4 && hidden.size(2) == H,
      "hidden must have shape [T, 4, 5120]");
  STD_TORCH_CHECK(hidden.size(0) <= std::numeric_limits<int>::max(),
                  "Engram token count exceeds the CUDA grid limit");
  STD_TORCH_CHECK(valid >= 0 && valid <= hidden.size(0),
                  "num_valid_tokens must be in [0, T]");
}

void check_packet(const Tensor& packet, int64_t rows, const char* name) {
  STD_TORCH_CHECK(packet.dim() == 2 && packet.size(0) == rows &&
                      packet.size(1) == PACK && packet.is_contiguous(),
                  name, " must be contiguous [", rows, ", 2624]");
  STD_TORCH_CHECK(packet.numel() == 0 ||
                      reinterpret_cast<uintptr_t>(packet.const_data_ptr()) %
                              alignof(float) ==
                          0,
                  name, " FP32 headers must be four-byte aligned");
}

void check_output_strides(const Tensor& output) {
  if (output.numel() == 0 || output.is_contiguous()) return;
  const int64_t stride[3] = {output.stride(0), output.stride(1),
                             output.stride(2)};
  const int64_t size[3] = {output.size(0), 4, H};
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (stride[order[j]] < stride[order[i]]) {
        const int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
  int64_t span = 1;
  bool separated = true;
  for (int d : order) {
    if (size[d] <= 1) continue;
    STD_TORCH_CHECK(stride[d] > 0, "output cannot have broadcast dimensions");
    separated &= stride[d] >= span;
    span += (size[d] - 1) * stride[d];
  }
  if (separated) return;
  // Check uncommon interleavings exactly; sorted spans can overlap harmlessly.
  for (int head_delta = 0; head_delta < 4; ++head_delta) {
    for (int feature_delta = head_delta == 0 ? 1 : 1 - H; feature_delta < H;
         ++feature_delta) {
      const int64_t delta = head_delta * stride[1] + feature_delta * stride[2];
      const bool overlap = size[0] == 1 ? delta == 0
                                        : delta % stride[0] == 0 &&
                                              delta / stride[0] > -size[0] &&
                                              delta / stride[0] < size[0];
      STD_TORCH_CHECK(!overlap, "output must have non-overlapping strides");
    }
  }
}

Strides strides(const Tensor& tensor) {
  return {tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}

void check_sm120(int32_t device) {
  int major = 0, minor = 0;
  cudaError_t err =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "Engram CUDA device query failed: ", cudaGetErrorString(err));
  err =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "Engram CUDA device query failed: ", cudaGetErrorString(err));
  STD_TORCH_CHECK(major == 12 && minor == 0, "Engram requires an SM120 device");
}

void check_launch() {
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "Engram CUDA launch failed: ", cudaGetErrorString(err));
}
}  // namespace

void sm120_engram_pack(const Tensor& hidden, const Tensor& local_projection,
                       const Tensor& q_weight, const Tensor& k_weight,
                       Tensor& packet, int64_t rank, int64_t num_valid_tokens) {
  check_hidden(hidden, num_valid_tokens);
  const int64_t tokens = hidden.size(0);
  const int32_t device = hidden.get_device_index();
  STD_TORCH_CHECK(rank >= 0 && rank < 4, "Engram rank must be in [0, 4)");
  STD_TORCH_CHECK(
      local_projection.dim() == 2 &&
          local_projection.size(0) >= num_valid_tokens &&
          local_projection.size(0) <= tokens &&
          local_projection.size(1) == 6400 && local_projection.is_contiguous(),
      "local_projection must be contiguous [rows, 6400], valid <= rows <= T");
  STD_TORCH_CHECK(q_weight.dim() == 2 && q_weight.size(0) == 4 &&
                      q_weight.size(1) == H && q_weight.is_contiguous() &&
                      k_weight.dim() == 2 && k_weight.size(0) == 4 &&
                      k_weight.size(1) == H && k_weight.is_contiguous(),
                  "q_weight and k_weight must be contiguous [4, 5120]");
  check_packet(packet, tokens, "packet");
  const Region hidden_region =
      check_tensor(hidden, ScalarType::BFloat16, device, "hidden");
  const Region local_region = check_tensor(
      local_projection, ScalarType::BFloat16, device, "local_projection");
  const Region q_region =
      check_tensor(q_weight, ScalarType::BFloat16, device, "q_weight");
  const Region k_region =
      check_tensor(k_weight, ScalarType::BFloat16, device, "k_weight");
  const Region packet_region =
      check_tensor(packet, ScalarType::Byte, device, "packet");
  check_disjoint(packet_region, hidden_region);
  check_disjoint(packet_region, local_region);
  check_disjoint(packet_region, q_region);
  check_disjoint(packet_region, k_region);
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm120(device);
  if (tokens == 0) return;
  const auto stream = get_current_cuda_stream();
  const auto launch = [&](auto contiguous) {
    sm120_engram_pack_kernel<decltype(contiguous)::value>
        <<<dim3(tokens, 4, 2), 32, 0, stream>>>(
            reinterpret_cast<const BF*>(hidden.const_data_ptr()),
            reinterpret_cast<const BF*>(local_projection.const_data_ptr()),
            reinterpret_cast<const BF*>(q_weight.const_data_ptr()),
            reinterpret_cast<const BF*>(k_weight.const_data_ptr()),
            reinterpret_cast<unsigned char*>(packet.mutable_data_ptr()),
            static_cast<int>(rank), num_valid_tokens, strides(hidden));
  };
  if (hidden.is_contiguous())
    launch(std::true_type{});
  else
    launch(std::false_type{});
  check_launch();
}

void sm120_engram_finish(const Tensor& hidden, const Tensor& gathered_packets,
                         const std::optional<Tensor>& token_mask,
                         Tensor& output, int64_t num_valid_tokens, double eps,
                         double clamp_value) {
  check_hidden(hidden, num_valid_tokens);
  const int64_t tokens = hidden.size(0);
  const int32_t device = hidden.get_device_index();
  STD_TORCH_CHECK(output.dim() == 3 && output.size(0) == tokens &&
                      output.size(1) == 4 && output.size(2) == H,
                  "output must have shape [T, 4, 5120]");
  STD_TORCH_CHECK(std::isfinite(eps) && eps > 0 &&
                      std::isfinite(static_cast<float>(eps)) &&
                      static_cast<float>(eps) > 0,
                  "eps must be positive and finite in FP32");
  STD_TORCH_CHECK(std::isfinite(clamp_value) && clamp_value > 0 &&
                      std::isfinite(static_cast<float>(clamp_value)) &&
                      static_cast<float>(clamp_value) > 0,
                  "clamp_value must be positive and finite in FP32");
  check_packet(gathered_packets, 4 * tokens, "gathered_packets");
  const Region hidden_region =
      check_tensor(hidden, ScalarType::BFloat16, device, "hidden");
  const Region gathered_region = check_tensor(
      gathered_packets, ScalarType::Byte, device, "gathered_packets");
  const Region output_region =
      check_tensor(output, ScalarType::BFloat16, device, "output");
  check_output_strides(output);
  check_disjoint(output_region, hidden_region);
  check_disjoint(output_region, gathered_region);
  if (token_mask) {
    STD_TORCH_CHECK(token_mask->dim() == 1 &&
                        token_mask->size(0) >= num_valid_tokens &&
                        token_mask->size(0) <= tokens,
                    "token_mask must have valid..T entries");
    check_disjoint(output_region, check_tensor(*token_mask, ScalarType::Bool,
                                               device, "token_mask"));
  }
  const torch::stable::accelerator::DeviceGuard guard(device);
  check_sm120(device);
  if (tokens == 0) return;
  const auto stream = get_current_cuda_stream();
  const auto launch = [&](auto contiguous) {
    sm120_engram_finish_kernel<decltype(contiguous)::value>
        <<<dim3(tokens, 4), 256, 0, stream>>>(
            reinterpret_cast<const BF*>(hidden.const_data_ptr()),
            reinterpret_cast<const unsigned char*>(
                gathered_packets.const_data_ptr()),
            token_mask
                ? reinterpret_cast<const bool*>(token_mask->const_data_ptr())
                : nullptr,
            reinterpret_cast<BF*>(output.mutable_data_ptr()), tokens,
            num_valid_tokens, static_cast<float>(eps),
            static_cast<float>(clamp_value), strides(hidden), strides(output),
            token_mask ? token_mask->stride(0) : 0, token_mask.has_value());
  };
  if (hidden.is_contiguous() && output.is_contiguous())
    launch(std::true_type{});
  else
    launch(std::false_type{});
  check_launch();
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sm120_engram_pack(Tensor hidden, Tensor local_projection, "
      "Tensor q_weight, Tensor k_weight, Tensor! packet, int rank, "
      "int num_valid_tokens) -> ()");
  ops.def(
      "sm120_engram_finish(Tensor hidden, Tensor gathered_packets, "
      "Tensor? token_mask, Tensor! output, int num_valid_tokens, float eps, "
      "float clamp_value) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("sm120_engram_pack", TORCH_BOX(&sm120_engram_pack));
  ops.impl("sm120_engram_finish", TORCH_BOX(&sm120_engram_finish));
}

#endif
