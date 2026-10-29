// SPDX-License-Identifier: Apache-2.0
#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace aphrodite::pcie_pull {
using Word = unsigned long long;
static_assert(sizeof(Word) == sizeof(int64_t));
constexpr int kRanks = 4;
constexpr uint64_t kLine = 128;
constexpr uint64_t kSlotAlignment = 256;
constexpr uint64_t kReadyBankBytes = 2 * kRanks * kLine;
constexpr unsigned kPairedTileBytes = 16384;
// Per-rank payload limit of the one-shot gather, and its cells per sender: one
// header plus one cell per four payload bytes.
constexpr uint64_t kGatherBytes = 16384;
constexpr uint64_t kGatherCells = kGatherBytes / 4 + 1;
constexpr uint64_t kGatherBankBytes = 2 * kRanks * kGatherCells * 8;

enum Error : Word {
  kReadyTimeout = 1,
  kMetadataMismatch = 2,
  kFutureEpoch = 3,
  kEpochExhausted = 4,
  kPeerFault = 5,
  kDirtyCompletion = 6,
  kUnsupportedDevice = 7,
};
enum Stage : Word { kPublish = 1, kReady = 2, kConsume = 3, kComplete = 4 };
enum Kind : Word { kClose = 1, kPairedR1 = 2, kPairedR2 = 3 };

// Fixed eight-word device-state layout shared with the Python owner.
enum StateWord : int {
  kStateEpoch = 0,
  kStateConsumeTicket = 1,
  kStateError = 2,
  kStateErrorPeer = 3,
  kStateErrorEpoch = 4,
  kStateErrorStage = 5,
  kStateTimeout = 6,
  kStatePublishTicket = 7,
  kStateWords = 8,
};

struct alignas(kLine) Ready {
  Word epoch;
  Word count;
  Word kind;
  unsigned char padding[kLine - 3 * sizeof(Word)];
};
struct alignas(kLine) Fault {
  Word code;
  Word peer;
  Word epoch;
  Word stage;
  unsigned char padding[kLine - 4 * sizeof(Word)];
};
static_assert(sizeof(Ready) == kLine && sizeof(Fault) == kLine);

struct Layout {
  uint64_t payload_offset;
  uint64_t slot_stride;
  uint64_t bytes;
  uint64_t fault_offset;
  uint64_t ready_offset;
  uint64_t r2_offset;
  uint64_t gather_offset;
};

Layout layout(int64_t capacity, int64_t blocks, int64_t transport) {
  STD_TORCH_CHECK(capacity > 0, "PCIe pull capacity must be positive");
  STD_TORCH_CHECK(transport == 2,
                  "PCIe transport must be paired single-CTA (2)");
  const uint64_t limit = std::numeric_limits<int64_t>::max();
  STD_TORCH_CHECK(blocks == 1,
                  "PCIe paired single-CTA transport requires blocks=1");
  // R1, R2 and CLOSE each have two parity banks of four origin cells.
  // Fault records follow these banks; gather cells follow the payload slots.
  const uint64_t fault_offset = 3 * kReadyBankBytes;
  const uint64_t offset = fault_offset + kRanks * kLine;
  STD_TORCH_CHECK(
      static_cast<uint64_t>(capacity) <=
          (limit - offset - kGatherBankBytes - 4 * (kSlotAlignment - 1)) / 12,
      "PCIe paired allocation size overflows int64");
  const uint64_t r1 =
      (2 * static_cast<uint64_t>(capacity) + kSlotAlignment - 1) &
      ~(kSlotAlignment - 1);
  const uint64_t r2 =
      (4 * static_cast<uint64_t>(capacity) + kSlotAlignment - 1) &
      ~(kSlotAlignment - 1);
  const uint64_t stride = r1 + r2;
  return {offset,
          stride,
          offset + 2 * stride + kGatherBankBytes,
          fault_offset,
          2 * kReadyBankBytes,
          r1,
          offset + 2 * stride};
}

struct Params {
  char* slabs[kRanks];
  // This GPU's private allocation; peers access only the IPC slabs.
  Word* state;
  uint64_t payload_offset;
  uint64_t slot_stride;
  uint64_t fault_offset;
  uint64_t ready_offset;
  uint64_t r2_offset;
  uint64_t gather_offset;
  int rank;
};

struct Context {
  Params params;
  torch::stable::Tensor state_owner;
  uint64_t capacity;
  uint64_t slab_bytes;
  int device;
  int threads;
  cudaStream_t stream;
};

__device__ __forceinline__ Word read_state(const Word* address) {
  Word value;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}
__device__ __forceinline__ void write_local(Word* address, Word value) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ __forceinline__ Word acquire(const Word* address) {
  Word value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}
__device__ __forceinline__ Word relaxed(const Word* address) {
  Word value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}
__device__ __forceinline__ void release(Word* address, Word value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}
__device__ __forceinline__ void store(Word* address, Word value) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}
__device__ __forceinline__ void device_fence() {
  asm volatile("fence.acq_rel.gpu;" : : : "memory");
}
__device__ __forceinline__ Word timer_ns() {
  Word value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ Ready* ready(const Params& p, int receiver,
                                        int sender, Word epoch) {
  const uint64_t index = (epoch & 1) * kRanks + sender;
  return reinterpret_cast<Ready*>(p.slabs[receiver] + p.ready_offset) + index;
}
__device__ __forceinline__ Ready* paired_ready(const Params& p, int receiver,
                                               int sender, Word epoch,
                                               int bank) {
  const uint64_t index = (epoch & 1) * kRanks + sender;
  return reinterpret_cast<Ready*>(p.slabs[receiver] + bank * kReadyBankBytes) +
         index;
}
__device__ __forceinline__ Fault* fault(const Params& p, int receiver,
                                        int origin) {
  return reinterpret_cast<Fault*>(p.slabs[receiver] + p.fault_offset) + origin;
}
__device__ __forceinline__ unsigned short* payload(const Params& p,
                                                   int receiver, Word epoch) {
  return reinterpret_cast<unsigned short*>(
      p.slabs[receiver] + p.payload_offset + (epoch & 1) * p.slot_stride);
}

// Only a local CAS winner writes this origin's immutable records. No peer RMW.
__device__ __forceinline__ void fail(const Params& p, Word code, Word peer,
                                     Word epoch, Word stage) {
  if (atomicCAS(p.state + kStateError, 0ULL, code) != 0) return;
  store(p.state + kStateErrorPeer, peer);
  store(p.state + kStateErrorEpoch, epoch);
  store(p.state + kStateErrorStage, stage);
#pragma unroll
  for (int receiver = 0; receiver < kRanks; ++receiver) {
    Fault* f = fault(p, receiver, p.rank);
    store(&f->peer, peer);
    store(&f->epoch, epoch);
    store(&f->stage, stage);
    release(&f->code, code);
  }
}

__device__ __forceinline__ bool observe_fault(const Params& p, Word stage) {
  if (read_state(p.state + kStateError) != 0) return true;
#pragma unroll
  for (int origin = 0; origin < kRanks; ++origin) {
    const Fault* local = fault(p, p.rank, origin);
    if (relaxed(&local->code) != 0) {
      const Word confirmed = acquire(&local->code);
      fail(p, kPeerFault, origin, confirmed != 0 ? relaxed(&local->epoch) : 0,
           stage);
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ uint4 load_vector(const void* address) {
  return __ldcg(static_cast<const uint4*>(address));
}
__device__ __forceinline__ void store_vector(void* address, uint4 value) {
  *static_cast<uint4*>(address) = value;
}
__device__ __forceinline__ unsigned short load_scalar(
    const unsigned short* address) {
  return __ldcg(address);
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float value;
  // Explicit non-FTZ instructions keep the contract independent of fast-math.
  asm("add.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
  return value;
}

__global__ void initialize_state(Word* state, Word timeout_ns) {
  for (int i = 0; i < kStateWords; ++i)
    state[i] = i == kStateTimeout ? timeout_ns : 0;
}

__device__ __forceinline__ Word complete_ticket(Word* ticket) {
  Word previous;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(previous)
               : "l"(ticket), "l"(1ULL)
               : "memory");
  return previous;
}
__global__ void publish(const __grid_constant__ Params p, Word* snapshot) {
  __shared__ Word epoch;
  __shared__ int bad;
  __shared__ int last;
  if (threadIdx.x == 0) {
    // A missing consumer completion must never expose an uninitialized zero.
    if (blockIdx.x == 0 && snapshot != nullptr) {
      write_local(snapshot, std::numeric_limits<Word>::max());
    }
    const Word prior = read_state(p.state + kStateEpoch);
    epoch = prior + 1;
    bad = observe_fault(p, kPublish);
    if (!bad && prior == std::numeric_limits<Word>::max()) {
      fail(p, kEpochExhausted, p.rank, prior, kPublish);
      bad = 1;
    }
  }
  __syncthreads();
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    // Drain each CTA before the last CTA publishes CLOSE.
    asm volatile("fence.acq_rel.sys;" : : : "memory");
    observe_fault(p, kPublish);
    last = complete_ticket(p.state + kStatePublishTicket) ==
           static_cast<Word>(gridDim.x - 1);
  }
  __syncthreads();
  if (last) {
    if (threadIdx.x == 0) bad = observe_fault(p, kPublish);
    __syncthreads();
    if (threadIdx.x < kRanks && !bad &&
        read_state(p.state + kStateError) == 0) {
      Ready* flag = ready(p, threadIdx.x, p.rank, epoch);
      store(&flag->count, 0);
      store(&flag->kind, kClose);
      release(&flag->epoch, epoch);
    }
    __syncthreads();
    if (threadIdx.x == 0) write_local(p.state + kStatePublishTicket, 0);
  }
}

__device__ __forceinline__ bool wait_ready(const Params& p, int sender,
                                           Word epoch, uint64_t count,
                                           Word kind, Word started,
                                           Word timeout_ns, int bank = -1) {
  const Ready* flag = bank < 0 ? ready(p, p.rank, sender, epoch)
                               : paired_ready(p, p.rank, sender, epoch, bank);
  unsigned misses = 0;
  for (;;) {
    const Word seen = relaxed(&flag->epoch);
    if (seen == epoch) {
      if (acquire(&flag->epoch) != epoch || relaxed(&flag->count) != count ||
          relaxed(&flag->kind) != kind) {
        fail(p, kMetadataMismatch, sender, epoch, kReady);
        return false;
      }
      return true;
    }
    if ((++misses & 63U) == 0 && observe_fault(p, kReady)) return false;
    if (seen > epoch) {
      fail(p, kFutureEpoch, sender, epoch, kReady);
      return false;
    }
    if (timer_ns() - started >= timeout_ns) {
      fail(p, kReadyTimeout, sender, epoch, kReady);
      return false;
    }
    __nanosleep(64);
  }
}

// The source may be only two-byte aligned. Inbox and staging words are aligned.
__device__ __forceinline__ unsigned paired_input_word(
    const unsigned short* source) {
  return static_cast<unsigned>(load_scalar(source)) |
         (static_cast<unsigned>(load_scalar(source + 1)) << 16);
}

__device__ __forceinline__ uint4 paired_load8(const unsigned short* source,
                                              bool aligned) {
  if (aligned) return load_vector(source);
  return make_uint4(paired_input_word(source), paired_input_word(source + 2),
                    paired_input_word(source + 4),
                    paired_input_word(source + 6));
}

struct PairedPacked8 {
  uint4 lo;
  uint4 hi;
};

__device__ __forceinline__ PairedPacked8 paired_pack8(int rank, uint4 own,
                                                      uint4 peer) {
  const uint4 first = (rank & 1) ? peer : own;
  const uint4 second = (rank & 1) ? own : peer;
  const unsigned a[4] = {first.x, first.y, first.z, first.w};
  const unsigned b[4] = {second.x, second.y, second.z, second.w};
  unsigned words[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (rank < 2) {
      words[2 * j] = __float_as_uint(
          add_rn(__uint_as_float(a[j] << 16), __uint_as_float(b[j] << 16)));
      words[2 * j + 1] =
          __float_as_uint(add_rn(__uint_as_float(a[j] & 0xffff0000U),
                                 __uint_as_float(b[j] & 0xffff0000U)));
    } else {
      words[2 * j] = (a[j] & 0xffffU) | (b[j] << 16);
      words[2 * j + 1] = (a[j] >> 16) | (b[j] & 0xffff0000U);
    }
  }
  return {make_uint4(words[0], words[1], words[2], words[3]),
          make_uint4(words[4], words[5], words[6], words[7])};
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
// Tile loads are prefetched into registers one tile ahead, so their latency
// overlaps staging and the TMA issue of the previous tile.
template <int Round>
__device__ __forceinline__ void paired_bulk_payload(const Params& p,
                                                    const unsigned short* input,
                                                    const unsigned short* r1,
                                                    Word epoch, uint64_t count,
                                                    uint4* tiles) {
  constexpr unsigned kTileVectors = kPairedTileBytes / sizeof(uint4);
  // Items per lane and tile for the smallest supported CTA (128 threads).
  constexpr int kItems = kTileVectors / 128;
  constexpr int kPartnerMask = Round == 1 ? 1 : 2;
  const uint64_t bytes = count * (Round == 1 ? 2 : 4);
  char* destination =
      reinterpret_cast<char*>(payload(p, p.rank ^ kPartnerMask, epoch)) +
      (Round == 1 ? 0 : p.r2_offset);
  const bool aligned = (reinterpret_cast<uintptr_t>(input) & 15) == 0;
  uint4 own[kItems];
  uint4 peer[kItems];
  // A Round 1 item is one 16-byte vector; a Round 2 item is one 32-byte group.
  const auto tile_items = [&](uint64_t offset) {
    const uint64_t tile_bytes =
        bytes - offset < kPairedTileBytes ? bytes - offset : kPairedTileBytes;
    return static_cast<unsigned>(tile_bytes / (Round == 1 ? 16 : 32));
  };
  const auto load_tile = [&](uint64_t offset) {
    const unsigned items = tile_items(offset);
  #pragma unroll
    for (int k = 0; k < kItems; ++k) {
      const unsigned item = threadIdx.x + k * blockDim.x;
      if (item < items) {
        if constexpr (Round == 1) {
          own[k] = paired_load8(input + offset / 2 + item * 8, aligned);
        } else {
          const uint64_t i = offset / 4 + item * 8;
          own[k] = paired_load8(input + i, aligned);
          peer[k] = load_vector(r1 + i);
        }
      }
    }
  };
  if (bytes > 0) load_tile(0);
  for (uint64_t offset = 0, index = 0; offset < bytes;
       offset += kPairedTileBytes, ++index) {
    // Only the leader issues TMA. Its oldest of two shared tiles is no longer
    // read by the async proxy before any generic writer reuses that tile.
    if (index >= 2 && threadIdx.x == 0) {
      asm volatile("cp.async.bulk.wait_group.read 1;" : : : "memory");
      asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
    }
    __syncthreads();
    const unsigned items = tile_items(offset);
    const unsigned tile_bytes = items * (Round == 1 ? 16 : 32);
    uint4* staging = tiles + (index & 1) * kTileVectors;
  #pragma unroll
    for (int k = 0; k < kItems; ++k) {
      const unsigned item = threadIdx.x + k * blockDim.x;
      if (item < items) {
        if constexpr (Round == 1) {
          staging[item] = own[k];
        } else {
          const PairedPacked8 packed = paired_pack8(p.rank, own[k], peer[k]);
          staging[2 * item] = packed.lo;
          staging[2 * item + 1] = packed.hi;
        }
      }
    }
    if (offset + kPairedTileBytes < bytes) load_tile(offset + kPairedTileBytes);
    // Fence every writer before the leader can issue the async read.
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned shared =
          static_cast<unsigned>(__cvta_generic_to_shared(staging));
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :
                   : "l"(destination + offset), "r"(shared), "r"(tile_bytes)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;" : : : "memory");
    }
  }
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
    // The leader is the only remote-store issuer; others only write shared.
    __threadfence_system();
  }
  __syncthreads();
}
#endif

__device__ __forceinline__ unsigned short paired_finish(float ab, float c,
                                                        float d) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(add_rn(add_rn(ab, c), d)));
}

__device__ __forceinline__ uint4 paired_result8(int rank, uint4 own, uint4 peer,
                                                uint4 r2_lo, uint4 r2_hi) {
  const uint4 first = (rank & 1) ? peer : own;
  const uint4 second = (rank & 1) ? own : peer;
  const unsigned a[4] = {first.x, first.y, first.z, first.w};
  const unsigned b[4] = {second.x, second.y, second.z, second.w};
  const unsigned remote[8] = {r2_lo.x, r2_lo.y, r2_lo.z, r2_lo.w,
                              r2_hi.x, r2_hi.y, r2_hi.z, r2_hi.w};
  unsigned words[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float a0 = __uint_as_float(a[j] << 16);
    const float a1 = __uint_as_float(a[j] & 0xffff0000U);
    const float b0 = __uint_as_float(b[j] << 16);
    const float b1 = __uint_as_float(b[j] & 0xffff0000U);
    unsigned lo;
    unsigned hi;
    if (rank < 2) {
      lo = paired_finish(add_rn(a0, b0), __uint_as_float(remote[2 * j] << 16),
                         __uint_as_float(remote[2 * j] & 0xffff0000U));
      hi = paired_finish(add_rn(a1, b1),
                         __uint_as_float(remote[2 * j + 1] << 16),
                         __uint_as_float(remote[2 * j + 1] & 0xffff0000U));
    } else {
      lo = paired_finish(__uint_as_float(remote[2 * j]), a0, b0);
      hi = paired_finish(__uint_as_float(remote[2 * j + 1]), a1, b1);
    }
    words[j] = lo | (hi << 16);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

// Exchanges the payload with both partners and records the epoch. The
// reduction itself runs in paired_combine on the same stream.
__global__ void paired_single(const __grid_constant__ Params p,
                              const unsigned short* input, uint64_t count) {
  __shared__ __align__(16) uint4 tiles[2 * kPairedTileBytes / sizeof(uint4)];
  __shared__ Word epoch;
  __shared__ Word timeout_ns;
  __shared__ int bad;
  if (threadIdx.x == 0) {
    const Word prior = read_state(p.state + kStateEpoch);
    epoch = prior + 1;
    timeout_ns = read_state(p.state + kStateTimeout);
    bad = observe_fault(p, kPublish);
    if (!bad && prior == std::numeric_limits<Word>::max()) {
      fail(p, kEpochExhausted, p.rank, prior, kPublish);
      bad = 1;
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ != 1200
    // No TMA PTX is emitted for any other project architecture. Executing an
    // unsupported device-code image is a sticky collective fault, not fallback.
    if (!bad) {
      fail(p, kUnsupportedDevice, p.rank, epoch, kPublish);
      bad = 1;
    }
#endif
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  const unsigned short* r1 = payload(p, p.rank, epoch);
  if (!bad) {
    paired_bulk_payload<1>(p, input, r1, epoch, count, tiles);
    if (threadIdx.x == 0) {
      bad = observe_fault(p, kPublish);
      if (!bad) {
        Ready* flag = paired_ready(p, p.rank ^ 1, p.rank, epoch, 0);
        store(&flag->count, count);
        store(&flag->kind, kPairedR1);
        release(&flag->epoch, epoch);
        bad = !wait_ready(p, p.rank ^ 1, epoch, count, kPairedR1, timer_ns(),
                          timeout_ns, 0);
        if (!bad) bad = observe_fault(p, kReady);
      }
    }
    // Thread 0's successful acquire makes the entire R1 inbox visible here.
    __syncthreads();
  }
  if (!bad) {
    paired_bulk_payload<2>(p, input, r1, epoch, count, tiles);
    if (threadIdx.x == 0) {
      bad = observe_fault(p, kPublish);
      if (!bad) {
        Ready* flag = paired_ready(p, p.rank ^ 2, p.rank, epoch, 1);
        store(&flag->count, count);
        store(&flag->kind, kPairedR2);
        release(&flag->epoch, epoch);
        bad = !wait_ready(p, p.rank ^ 2, epoch, count, kPairedR2, timer_ns(),
                          timeout_ns, 1);
        if (!bad) bad = observe_fault(p, kReady);
      }
    }
    __syncthreads();
  }
#endif

  device_fence();
  __syncthreads();
  if (threadIdx.x == 0 && !bad) {
    // DATA leaves both consume and publish tickets at zero.
    // Before E+2 can overwrite a parity, E+1 must have consumed READY from both
    // write partners, which follows their completed E. paired_combine reads
    // this epoch's inboxes before the next paired_single on the same stream
    // can publish READY(E+1), so recording E here keeps R1 and R2 intact.
    write_local(p.state + kStateEpoch, epoch);
  }
}

// Reduces the inboxes of the epoch paired_single just completed. It runs on
// many CTAs: a single CTA is limited by one SM's memory bandwidth. A sticky
// fault, local or propagated from a peer, before or after the reduction
// zeroes this CTA's output range instead.
__global__ void paired_combine(const __grid_constant__ Params p,
                               const unsigned short* input,
                               unsigned short* output, uint64_t count) {
  __shared__ Word epoch;
  __shared__ int bad;
  if (threadIdx.x == 0) {
    bad = observe_fault(p, kConsume);
    epoch = read_state(p.state + kStateEpoch);
  }
  __syncthreads();
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t first =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bad) {
    for (uint64_t i = first; i < count; i += stride) output[i] = 0;
    return;
  }
  const unsigned short* r1 = payload(p, p.rank, epoch);
  const unsigned* r2 = reinterpret_cast<const unsigned*>(
      reinterpret_cast<const char*>(r1) + p.r2_offset);
  const bool input_aligned = (reinterpret_cast<uintptr_t>(input) & 15) == 0;
  const bool output_aligned = (reinterpret_cast<uintptr_t>(output) & 15) == 0;
  for (uint64_t vector = first; vector < count / 8; vector += stride) {
    const uint64_t i = vector * 8;
    const uint4 own = paired_load8(input + i, input_aligned);
    const uint4 peer = load_vector(r1 + i);
    const uint4 r2_lo =
        load_vector(reinterpret_cast<const unsigned short*>(r2 + i));
    const uint4 r2_hi =
        load_vector(reinterpret_cast<const unsigned short*>(r2 + i + 4));
    const uint4 result = paired_result8(p.rank, own, peer, r2_lo, r2_hi);
    if (output_aligned) {
      store_vector(output + i, result);
    } else {
      const unsigned words[4] = {result.x, result.y, result.z, result.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        output[i + 2 * j] = static_cast<unsigned short>(words[j]);
        output[i + 2 * j + 1] = static_cast<unsigned short>(words[j] >> 16);
      }
    }
  }
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0) bad = observe_fault(p, kComplete);
  __syncthreads();
  if (bad) {
    for (uint64_t i = first; i < count; i += stride) output[i] = 0;
  }
}

// A one-shot small all-gather in LL form. Each 8-byte cell carries four payload
// bytes and a nonzero epoch tag, so one posted store per cell needs neither a
// fence nor a separate READY; cell 0 carries the byte count. Receivers clear
// every consumed cell and slabs start zeroed, so a nonzero tag can only be this
// epoch's. No rank reaches E+2 before every rank has published E+1, which
// follows its completed E, so a parity bank is never rewritten while read.
// A fault zeroes the whole output.
__device__ __forceinline__ Word gather_tag(Word epoch) {
  return (epoch % 0xffffffffULL + 1) << 32;
}
__device__ __forceinline__ Word* gather_cells(const Params& p, int receiver,
                                              int sender, Word epoch) {
  return reinterpret_cast<Word*>(p.slabs[receiver] + p.gather_offset) +
         ((epoch & 1) * kRanks + sender) * kGatherCells;
}

__global__ void gather_small(const __grid_constant__ Params p,
                             const unsigned char* input, unsigned char* output,
                             uint64_t bytes) {
  __shared__ Word epoch;
  __shared__ Word timeout_ns;
  __shared__ int bad;
  if (threadIdx.x == 0) {
    const Word prior = read_state(p.state + kStateEpoch);
    epoch = prior + 1;
    timeout_ns = read_state(p.state + kStateTimeout);
    bad = observe_fault(p, kPublish);
    if (!bad && prior == std::numeric_limits<Word>::max()) {
      fail(p, kEpochExhausted, p.rank, prior, kPublish);
      bad = 1;
    }
  }
  __syncthreads();
  const uint64_t words = (bytes + 3) / 4;
  const Word tag = gather_tag(epoch);
  int failed = bad;
  if (!failed) {
    for (uint64_t cell = threadIdx.x; cell <= words; cell += blockDim.x) {
      unsigned value = static_cast<unsigned>(bytes);
      if (cell > 0) {
        const uint64_t first = 4 * (cell - 1);
        value = 0;
        for (uint64_t b = 0; b < 4 && first + b < bytes; ++b) {
          const unsigned char byte = __ldcg(input + first + b);
          output[p.rank * bytes + first + b] = byte;
          value |= static_cast<unsigned>(byte) << (8 * b);
        }
      }
#pragma unroll
      for (int receiver = 0; receiver < kRanks; ++receiver) {
        if (receiver != p.rank) {
          store(gather_cells(p, receiver, p.rank, epoch) + cell, tag | value);
        }
      }
    }
    const Word started = timer_ns();
    unsigned misses = 0;
    for (int sender = 0; sender < kRanks && !failed; ++sender) {
      if (sender == p.rank) continue;
      Word* cells = gather_cells(p, p.rank, sender, epoch);
      for (uint64_t cell = threadIdx.x; cell <= words && !failed;
           cell += blockDim.x) {
        Word seen = relaxed(cells + cell);
        while (seen == 0) {
          if ((++misses & 63U) == 0 && observe_fault(p, kReady)) {
            failed = 1;
            break;
          }
          if (timer_ns() - started >= timeout_ns) {
            fail(p, kReadyTimeout, sender, epoch, kReady);
            failed = 1;
            break;
          }
          seen = relaxed(cells + cell);
        }
        if (failed) break;
        const unsigned value = static_cast<unsigned>(seen);
        if ((seen & ~0xffffffffULL) != tag || (cell == 0 && value != bytes)) {
          fail(p, kMetadataMismatch, sender, epoch, kReady);
          failed = 1;
          break;
        }
        store(cells + cell, 0);
        if (cell > 0) {
          const uint64_t first = 4 * (cell - 1);
          for (uint64_t b = 0; b < 4 && first + b < bytes; ++b) {
            output[sender * bytes + first + b] =
                static_cast<unsigned char>(value >> (8 * b));
          }
        }
      }
    }
  }
  if (__syncthreads_or(failed) && threadIdx.x == 0) bad = 1;
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0 && !bad) bad = observe_fault(p, kComplete);
  __syncthreads();
  if (bad) {
    for (uint64_t i = threadIdx.x; i < kRanks * bytes; i += blockDim.x) {
      output[i] = 0;
    }
    return;
  }
  if (threadIdx.x == 0) write_local(p.state + kStateEpoch, epoch);
}

__global__ void pull_reduce(const __grid_constant__ Params p, Word* snapshot) {
  __shared__ Word epoch;
  __shared__ Word started;
  __shared__ Word timeout_ns;
  __shared__ int bad;
  __shared__ int last;
  if (threadIdx.x == 0) {
    const Word prior = read_state(p.state + kStateEpoch);
    timeout_ns = read_state(p.state + kStateTimeout);
    epoch = prior + 1;
    bad = observe_fault(p, kReady);
    if (!bad && prior == std::numeric_limits<Word>::max()) {
      fail(p, kEpochExhausted, p.rank, prior, kReady);
      bad = 1;
    }
    started = timer_ns();
  }
  __syncthreads();
  int admission_failed = bad;
  if (!admission_failed && threadIdx.x < kRanks) {
    admission_failed =
        !wait_ready(p, threadIdx.x, epoch, 0, kClose, started, timeout_ns);
  }
  __syncthreads_or(admission_failed);

  // No CTA waits for another CTA: the final acq_rel RMW gathers completion.
  // CLOSE has no output stores. Slot reuse is ordered across devices by the
  // next publication, not by another system fence here.
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    observe_fault(p, kConsume);
    last = complete_ticket(p.state + kStateConsumeTicket) ==
           static_cast<Word>(gridDim.x - 1);
    if (last) {
      observe_fault(p, kComplete);
      bad = read_state(p.state + kStateError) != 0;
    }
  }
  __syncthreads();
  if (last) {
    device_fence();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (!bad) write_local(p.state + kStateEpoch, epoch);
      write_local(p.state + kStateConsumeTicket, 0);
      if (snapshot != nullptr)
        write_local(snapshot, read_state(p.state + kStateError));
    }
  }
}

int64_t buffer_size(int64_t capacity, int64_t max_blocks, int64_t transport) {
  return static_cast<int64_t>(layout(capacity, max_blocks, transport).bytes);
}

Context& context(int64_t handle) {
  STD_TORCH_CHECK(handle != 0, "PCIe pull context is null");
  return *reinterpret_cast<Context*>(handle);
}

cudaStream_t checked_stream(const Context& ctx) {
  const cudaStream_t stream = get_current_cuda_stream(ctx.device);
  STD_TORCH_CHECK(stream == ctx.stream,
                  "PCIe pull must use its bound stream, including capture");
  return stream;
}

void require_outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  STD_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  STD_TORCH_CHECK(
      status == cudaStreamCaptureStatusNone,
      "PCIe pull lifecycle operations are forbidden during capture");
}

bool overlaps(uintptr_t a, uint64_t a_bytes, uintptr_t b, uint64_t b_bytes);

int64_t init(const std::vector<int64_t>& peer_ptrs,
             torch::stable::Tensor& state, int64_t rank, int64_t capacity,
             int64_t blocks, int64_t threads, int64_t timeout_ns,
             int64_t transport) {
  const Layout sizes = layout(capacity, blocks, transport);
  STD_TORCH_CHECK(peer_ptrs.size() == kRanks && rank >= 0 && rank < kRanks,
                  "PCIe pull requires exactly four peers and rank in [0,3]");
  STD_TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
                  "PCIe pull threads must be 128, 256 or 512");
  STD_TORCH_CHECK(timeout_ns > 0, "PCIe pull timeout must be positive");
  STD_TORCH_CHECK(
      state.device().is_cuda() && state.is_contiguous() &&
          state.scalar_type() == torch::headeronly::ScalarType::Long &&
          state.numel() >= kStateWords,
      "PCIe pull state must be contiguous CUDA int64 with >=8 words");
  const int device = state.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const cudaStream_t stream = get_current_cuda_stream(device);
  require_outside_capture(stream);
  cudaDeviceProp properties;
  STD_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  STD_TORCH_CHECK(properties.major == 12 && properties.minor == 0,
                  "PCIe pull currently requires SM120");
  const auto state_begin = reinterpret_cast<uintptr_t>(state.const_data_ptr());
  for (int r = 0; r < kRanks; ++r) {
    STD_TORCH_CHECK(
        peer_ptrs[r] != 0 && (static_cast<uint64_t>(peer_ptrs[r]) & 255) == 0,
        "PCIe pull slabs must be nonnull and 256-byte aligned");
    for (int other = 0; other < r; ++other) {
      STD_TORCH_CHECK(peer_ptrs[r] != peer_ptrs[other],
                      "PCIe pull slabs must be independently allocated");
    }
    const auto begin = static_cast<uintptr_t>(peer_ptrs[r]);
    STD_TORCH_CHECK(
        !overlaps(state_begin, kStateWords * sizeof(Word), begin, sizes.bytes),
        "PCIe pull state must not alias a slab");
    cudaPointerAttributes attributes;
    STD_CUDA_CHECK(
        cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(begin)));
    STD_TORCH_CHECK(attributes.type == cudaMemoryTypeDevice,
                    "PCIe pull slabs must be CUDA device allocations");
    if (r == rank) {
      STD_TORCH_CHECK(attributes.device == device,
                      "PCIe pull local slab must belong to the state device");
    }
  }
  CUdeviceptr allocation_base = 0;
  size_t allocation_bytes = 0;
  const CUresult range_status =
      cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                           static_cast<CUdeviceptr>(peer_ptrs[rank]));
  STD_TORCH_CHECK(
      range_status == CUDA_SUCCESS &&
          allocation_base == static_cast<CUdeviceptr>(peer_ptrs[rank]) &&
          allocation_bytes >= sizes.bytes,
      "PCIe pull local allocation does not cover the selected layout");
  auto ctx = std::make_unique<Context>(Context{{},
                                               state,
                                               static_cast<uint64_t>(capacity),
                                               sizes.bytes,
                                               device,
                                               static_cast<int>(threads),
                                               stream});
  for (int r = 0; r < kRanks; ++r) {
    ctx->params.slabs[r] = reinterpret_cast<char*>(peer_ptrs[r]);
  }
  ctx->params.state = reinterpret_cast<Word*>(state.mutable_data_ptr());
  ctx->params.payload_offset = sizes.payload_offset;
  ctx->params.slot_stride = sizes.slot_stride;
  ctx->params.fault_offset = sizes.fault_offset;
  ctx->params.ready_offset = sizes.ready_offset;
  ctx->params.r2_offset = sizes.r2_offset;
  ctx->params.gather_offset = sizes.gather_offset;
  ctx->params.rank = static_cast<int>(rank);
  initialize_state<<<1, 1, 0, stream>>>(ctx->params.state, timeout_ns);
  STD_CUDA_CHECK(cudaGetLastError());
  return reinterpret_cast<int64_t>(ctx.release());
}

bool overlaps(uintptr_t a, uint64_t a_bytes, uintptr_t b, uint64_t b_bytes) {
  return a <= b ? b - a < a_bytes : a - b < b_bytes;
}

void reject_workspace_alias(const Context& ctx, const void* pointer,
                            uint64_t bytes) {
  const auto begin = reinterpret_cast<uintptr_t>(pointer);
  STD_TORCH_CHECK(
      !overlaps(begin, bytes, reinterpret_cast<uintptr_t>(ctx.params.state),
                kStateWords * sizeof(Word)),
      "PCIe pull tensors must not alias device state");
  for (int r = 0; r < kRanks; ++r) {
    STD_TORCH_CHECK(!overlaps(begin, bytes,
                              reinterpret_cast<uintptr_t>(ctx.params.slabs[r]),
                              ctx.slab_bytes),
                    "PCIe pull tensors must not alias IPC slabs");
  }
}

void all_reduce(int64_t handle, const torch::stable::Tensor& input,
                torch::stable::Tensor& output) {
  const Context& ctx = context(handle);
  const torch::stable::accelerator::DeviceGuard guard(ctx.device);
  const cudaStream_t stream = checked_stream(ctx);
  STD_TORCH_CHECK(
      input.device().is_cuda() && output.device().is_cuda() &&
          input.get_device_index() == ctx.device &&
          output.get_device_index() == ctx.device &&
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.scalar_type() == input.scalar_type() &&
          input.is_contiguous() && output.is_contiguous() &&
          input.dim() == output.dim(),
      "PCIe pull input/output must be same-shape contiguous local BF16");
  for (int d = 0; d < input.dim(); ++d) {
    STD_TORCH_CHECK(input.size(d) == output.size(d),
                    "PCIe pull input/output shapes differ");
  }
  const uint64_t count = input.numel();
  STD_TORCH_CHECK(count > 0 && count <= ctx.capacity,
                  "PCIe pull element count exceeds capacity or is empty");
  const void* source = input.const_data_ptr();
  void* destination = output.mutable_data_ptr();
  STD_TORCH_CHECK(
      !overlaps(reinterpret_cast<uintptr_t>(source), count * 2,
                reinterpret_cast<uintptr_t>(destination), count * 2),
      "PCIe pull preserves input: overlapping input/output is forbidden");
  reject_workspace_alias(ctx, source, count * 2);
  reject_workspace_alias(ctx, destination, count * 2);
  STD_TORCH_CHECK(count % 8 == 0,
                  "PCIe paired single-CTA count must be divisible by 8");
  paired_single<<<1, ctx.threads, 0, stream>>>(
      ctx.params, static_cast<const unsigned short*>(source), count);
  STD_CUDA_CHECK(cudaGetLastError());
  constexpr unsigned kCombineThreads = 128;
  const unsigned combine_blocks = static_cast<unsigned>(
      (count / 8 + kCombineThreads - 1) / kCombineThreads);
  paired_combine<<<combine_blocks, kCombineThreads, 0, stream>>>(
      ctx.params, static_cast<const unsigned short*>(source),
      static_cast<unsigned short*>(destination), count);
  STD_CUDA_CHECK(cudaGetLastError());
}

void all_gather(int64_t handle, const torch::stable::Tensor& input,
                torch::stable::Tensor& output) {
  const Context& ctx = context(handle);
  const torch::stable::accelerator::DeviceGuard guard(ctx.device);
  const cudaStream_t stream = checked_stream(ctx);
  STD_TORCH_CHECK(
      input.device().is_cuda() && output.device().is_cuda() &&
          input.get_device_index() == ctx.device &&
          output.get_device_index() == ctx.device &&
          output.scalar_type() == input.scalar_type() &&
          input.is_contiguous() && output.is_contiguous() &&
          output.numel() == kRanks * input.numel(),
      "PCIe gather output must be a contiguous local stack of four inputs");
  const uint64_t bytes = input.numel() * input.element_size();
  STD_TORCH_CHECK(bytes > 0 && bytes <= kGatherBytes,
                  "PCIe gather payload is empty or exceeds the gather cells");
  const void* source = input.const_data_ptr();
  void* destination = output.mutable_data_ptr();
  STD_TORCH_CHECK(
      !overlaps(reinterpret_cast<uintptr_t>(source), bytes,
                reinterpret_cast<uintptr_t>(destination), kRanks * bytes),
      "PCIe gather input and output must not overlap");
  reject_workspace_alias(ctx, source, bytes);
  reject_workspace_alias(ctx, destination, kRanks * bytes);
  gather_small<<<1, ctx.threads, 0, stream>>>(
      ctx.params, static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination), bytes);
  STD_CUDA_CHECK(cudaGetLastError());
}

void close_step(int64_t handle, torch::stable::Tensor& snapshot) {
  const Context& ctx = context(handle);
  const torch::stable::accelerator::DeviceGuard guard(ctx.device);
  const cudaStream_t stream = checked_stream(ctx);
  STD_TORCH_CHECK(
      snapshot.device().is_cuda() &&
          snapshot.get_device_index() == ctx.device &&
          snapshot.scalar_type() == torch::headeronly::ScalarType::Long &&
          snapshot.is_contiguous() && snapshot.dim() == 1 &&
          snapshot.numel() == 1,
      "PCIe pull close snapshot must be local CUDA int64[1]");
  auto* status = reinterpret_cast<Word*>(snapshot.mutable_data_ptr());
  reject_workspace_alias(ctx, status, sizeof(Word));
  publish<<<1, 32, 0, stream>>>(ctx.params, status);
  STD_CUDA_CHECK(cudaGetLastError());
  pull_reduce<<<1, 32, 0, stream>>>(ctx.params, status);
  STD_CUDA_CHECK(cudaGetLastError());
}

void bind_stream(int64_t handle) {
  Context& ctx = context(handle);
  const torch::stable::accelerator::DeviceGuard guard(ctx.device);
  const cudaStream_t stream = get_current_cuda_stream(ctx.device);
  require_outside_capture(stream);
  require_outside_capture(ctx.stream);
  // Whole-group quiescence is the Python owner's prerequisite, not a hidden
  // device synchronization or collective performed by this host-only binding.
  ctx.stream = stream;
}

int64_t local_status(int64_t handle) {
  const Context& ctx = context(handle);
  const torch::stable::accelerator::DeviceGuard guard(ctx.device);
  const cudaStream_t stream = checked_stream(ctx);
  require_outside_capture(stream);
  Word values[kRanks + 3]{};
  STD_CUDA_CHECK(cudaMemcpyAsync(values, ctx.params.state + kStateError,
                                 sizeof(Word), cudaMemcpyDeviceToHost, stream));
  STD_CUDA_CHECK(cudaMemcpyAsync(values + kRanks + 1,
                                 ctx.params.state + kStateConsumeTicket,
                                 sizeof(Word), cudaMemcpyDeviceToHost, stream));
  STD_CUDA_CHECK(cudaMemcpyAsync(values + kRanks + 2,
                                 ctx.params.state + kStatePublishTicket,
                                 sizeof(Word), cudaMemcpyDeviceToHost, stream));
  const auto* faults = reinterpret_cast<const Fault*>(
      ctx.params.slabs[ctx.params.rank] + ctx.params.fault_offset);
  for (int peer = 0; peer < kRanks; ++peer) {
    STD_CUDA_CHECK(cudaMemcpyAsync(values + peer + 1, &faults[peer].code,
                                   sizeof(Word), cudaMemcpyDeviceToHost,
                                   stream));
  }
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (values[kRanks + 1] || values[kRanks + 2]) return kDirtyCompletion;
  for (int i = 0; i <= kRanks; ++i) {
    if (values[i] != 0) return static_cast<int64_t>(values[i]);
  }
  return 0;
}

void dispose(int64_t handle) { delete reinterpret_cast<Context*>(handle); }

void close_ipc_handle(int64_t pointer) {
  STD_TORCH_CHECK(pointer != 0, "Cannot close a null CUDA IPC mapping");
  int device;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  require_outside_capture(get_current_cuda_stream(device));
  STD_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(pointer)));
}
}  // namespace aphrodite::pcie_pull

STABLE_TORCH_LIBRARY_FRAGMENT(_C_pcie_pull, ops) {
  ops.def("buffer_size(int capacity, int max_blocks, int transport=2) -> int");
  ops.def(
      "init(int[] peer_ptrs, Tensor state, int rank, int capacity, int blocks, "
      "int threads, int timeout_ns, int transport=2) -> int");
  ops.def("all_reduce(int handle, Tensor input, Tensor! output) -> ()");
  ops.def("all_gather(int handle, Tensor input, Tensor! output) -> ()");
  ops.def("close_step(int handle, Tensor! snapshot) -> ()");
  ops.def("bind_stream(int handle) -> ()");
  ops.def("local_status(int handle) -> int");
  ops.def("dispose(int handle) -> ()");
  ops.def("close_ipc_handle(int pointer) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_pcie_pull, CUDA, ops) {
  ops.impl("init", TORCH_BOX(&aphrodite::pcie_pull::init));
  ops.impl("all_reduce", TORCH_BOX(&aphrodite::pcie_pull::all_reduce));
  ops.impl("all_gather", TORCH_BOX(&aphrodite::pcie_pull::all_gather));
  ops.impl("close_step", TORCH_BOX(&aphrodite::pcie_pull::close_step));
}

STABLE_TORCH_LIBRARY_IMPL(_C_pcie_pull, CompositeExplicitAutograd, ops) {
  ops.impl("buffer_size", TORCH_BOX(&aphrodite::pcie_pull::buffer_size));
  ops.impl("bind_stream", TORCH_BOX(&aphrodite::pcie_pull::bind_stream));
  ops.impl("local_status", TORCH_BOX(&aphrodite::pcie_pull::local_status));
  ops.impl("dispose", TORCH_BOX(&aphrodite::pcie_pull::dispose));
  ops.impl("close_ipc_handle",
           TORCH_BOX(&aphrodite::pcie_pull::close_ipc_handle));
}
