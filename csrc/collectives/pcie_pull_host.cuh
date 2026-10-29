// SPDX-License-Identifier: Apache-2.0
// Included inside aphrodite::pcie_pull after the shared checked-protocol
// helpers.

__device__ __forceinline__ HostLane* host_lane(const Params& p, int rank,
                                               unsigned lane) {
  // Mode3 uses the otherwise-unused gather area for raw-protocol controls.
  return reinterpret_cast<HostLane*>(p.slabs[rank] + p.gather_offset) + lane;
}
__device__ __forceinline__ unsigned char* host_payload(const Params& p,
                                                       int rank, Word epoch) {
  return reinterpret_cast<unsigned char*>(payload(p, rank, epoch));
}
__device__ __forceinline__ uint4
host_load_vector(const unsigned char* pointer) {
  uint4 value;
  // System acquire orders publication, but payload must also discard an L2
  // line retained from a previous replay on non-coherent PCIe GPUs.
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer)
               : "memory");
  return value;
}
__device__ __forceinline__ unsigned host_load_byte(
    const unsigned char* pointer) {
  unsigned value;
  asm volatile("ld.global.cv.u8 %0, [%1];"
               : "=r"(value)
               : "l"(pointer)
               : "memory");
  return value;
}
__device__ __forceinline__ void host_store_vector(unsigned char* pointer,
                                                  uint4 value) {
  asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(pointer), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
}
__device__ __forceinline__ void host_store_byte(unsigned char* pointer,
                                                unsigned value) {
  asm volatile("st.global.wt.u8 [%0], %1;"
               :
               : "l"(pointer), "r"(value)
               : "memory");
}
__device__ __forceinline__ uint4
host_local_vector(const unsigned char* pointer) {
  if ((reinterpret_cast<uintptr_t>(pointer) & 15) == 0)
    return *reinterpret_cast<const uint4*>(pointer);
  uint4 value;
  auto* bytes = reinterpret_cast<unsigned char*>(&value);
#pragma unroll
  for (int i = 0; i < 16; ++i) bytes[i] = pointer[i];
  return value;
}
__device__ __forceinline__ void host_local_store(unsigned char* pointer,
                                                 uint4 value) {
  if ((reinterpret_cast<uintptr_t>(pointer) & 15) == 0) {
    *reinterpret_cast<uint4*>(pointer) = value;
  } else {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
#pragma unroll
    for (int i = 0; i < 16; ++i) pointer[i] = bytes[i];
  }
}
__device__ void host_publish_payload(const unsigned char* input,
                                     unsigned char* destination,
                                     uint64_t bytes) {
  const uint64_t vectors =
      (reinterpret_cast<uintptr_t>(input) & 15) == 0 ? bytes / 16 : 0;
  for (uint64_t i = threadIdx.x; i < vectors; i += blockDim.x) {
    host_store_vector(destination + 16 * i,
                      *reinterpret_cast<const uint4*>(input + 16 * i));
  }
  for (uint64_t i = vectors * 16 + threadIdx.x; i < bytes; i += blockDim.x)
    host_store_byte(destination + i, input[i]);
  // A leader-only fence cannot drain the other threads' posted PCIe stores.
  // Every writer fences, then the CTA rendezvous precedes READY's release.
  __threadfence_system();
}
__device__ bool host_wait_ready(const Params& p, unsigned lane, int peer,
                                Word epoch, uint64_t bytes, Word kind,
                                Word started, Word timeout_ns) {
  const HostLane* control = host_lane(p, peer, lane);
  const unsigned parity = epoch & 1;
  unsigned misses = 0;
  for (;;) {
    const Word seen = acquire(&control->epoch[parity]);
    if (seen == epoch) {
      if (acquire(&control->descriptor[parity]) != ((bytes << 3) | kind)) {
        fail(p, kMetadataMismatch, peer, epoch, kReady);
        return false;
      }
      return true;
    }
    if (seen > epoch) {
      fail(p, kFutureEpoch, peer, epoch, kReady);
      return false;
    }
    if ((++misses & 31U) == 0) {
      if (observe_fault(p, kReady)) return false;
      if (timer_ns() - started >= timeout_ns) {
        fail(p, kReadyTimeout, peer, epoch, kReady);
        return false;
      }
    }
  }
}
__device__ __forceinline__ float host_from_bf16(unsigned bits) {
  return __bfloat162float(
      __ushort_as_bfloat16(static_cast<unsigned short>(bits)));
}
__device__ __forceinline__ unsigned host_to_bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}
__device__ void host_reduce_payload(const Params& p, const unsigned char* input,
                                    unsigned char* output, Word epoch,
                                    uint64_t begin, uint64_t bytes) {
  const uint64_t vectors = bytes / 16;
  for (uint64_t i = threadIdx.x; i < vectors; i += blockDim.x) {
    const uint64_t offset = begin + 16 * i;
    uint4 values[kRanks];
#pragma unroll
    for (int peer = 0; peer < kRanks; ++peer) {
      values[peer] =
          peer == p.rank
              ? host_local_vector(input + offset)
              : host_load_vector(host_payload(p, peer, epoch) + offset);
    }
    float sums[8] = {};
#pragma unroll
    for (int peer = 0; peer < kRanks; ++peer) {
      const uint4 value = values[peer];
      const unsigned words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        sums[2 * pair] = add_rn(sums[2 * pair], host_from_bf16(words[pair]));
        sums[2 * pair + 1] =
            add_rn(sums[2 * pair + 1], host_from_bf16(words[pair] >> 16));
      }
    }
    unsigned words[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      words[pair] = host_to_bf16(sums[2 * pair]) |
                    (host_to_bf16(sums[2 * pair + 1]) << 16);
    }
    host_local_store(output + offset,
                     make_uint4(words[0], words[1], words[2], words[3]));
  }
  for (uint64_t i = vectors * 16 + 2 * threadIdx.x; i < bytes;
       i += 2 * blockDim.x) {
    const uint64_t offset = begin + i;
    float sum = 0;
#pragma unroll
    for (int peer = 0; peer < kRanks; ++peer) {
      unsigned bits;
      if (peer == p.rank) {
        bits = *reinterpret_cast<const unsigned short*>(input + offset);
      } else {
        const unsigned char* source = host_payload(p, peer, epoch) + offset;
        bits = host_load_byte(source) | (host_load_byte(source + 1) << 8);
      }
      sum = add_rn(sum, host_from_bf16(bits));
    }
    *reinterpret_cast<unsigned short*>(output + offset) = host_to_bf16(sum);
  }
}
__device__ void host_gather_payload(const Params& p, const unsigned char* input,
                                    unsigned char* output, Word epoch,
                                    uint64_t total, uint64_t begin,
                                    uint64_t bytes) {
  const uint64_t vectors = bytes / 16;
  for (uint64_t item = threadIdx.x; item < vectors * kRanks;
       item += blockDim.x) {
    const int peer = item % kRanks;
    const uint64_t offset = begin + 16 * (item / kRanks);
    const uint4 value =
        peer == p.rank
            ? host_local_vector(input + offset)
            : host_load_vector(host_payload(p, peer, epoch) + offset);
    host_local_store(output + peer * total + offset, value);
  }
  const uint64_t tail = bytes - vectors * 16;
  for (uint64_t item = threadIdx.x; item < tail * kRanks; item += blockDim.x) {
    const int peer = item % kRanks;
    const uint64_t offset = begin + vectors * 16 + item / kRanks;
    output[peer * total + offset] =
        peer == p.rank ? input[offset]
                       : host_load_byte(host_payload(p, peer, epoch) + offset);
  }
}
__device__ void host_zero_output(unsigned char* output, uint64_t total,
                                 uint64_t begin, uint64_t bytes, Word kind) {
  const int copies = kind == kHostGather ? kRanks : 1;
  for (int peer = 0; peer < copies; ++peer)
    for (uint64_t i = threadIdx.x; i < bytes; i += blockDim.x)
      output[peer * total + begin + i] = 0;
}

// One CTA per <=64KiB lane; never a residency-dependent inter-CTA barrier.
// Unlike the experiment's per-lane counters, the shared owner has one epoch.
// Every active lane checks every rank's publication. Completing E+1 therefore
// proves all peers completed *all* reads of E before they published E+1, even
// when the lane count shrinks, an inactive lane returns, or E+1 is CLOSE.
// CLOSE's separate all-rank publish/consume kernels supply exactly the same
// stream-ordered completion proof, so E+2 may safely reuse E's raw parity.
__global__ void host_collective(const __grid_constant__ Params p,
                                const unsigned char* input,
                                unsigned char* output, uint64_t total,
                                Word kind) {
  const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * kHostLaneBytes;
  const uint64_t remaining = total - begin;
  const uint64_t bytes =
      remaining < kHostLaneBytes ? remaining : kHostLaneBytes;
  __shared__ Word epoch;
  __shared__ Word started;
  __shared__ Word timeout_ns;
  __shared__ int bad;
  if (threadIdx.x == 0) {
    const Word prior = read_state(p.state + kStateEpoch);
    epoch = prior + 1;
    started = timer_ns();
    timeout_ns = read_state(p.state + kStateTimeout);
    // Host fault records require PCIe reads. Keep those scans in stalled waits
    // and the mandatory all-rank CLOSE gate, not in every healthy data CTA.
    bad = read_state(p.state + kStateError) != 0;
    if (!bad && prior == std::numeric_limits<Word>::max()) {
      fail(p, kEpochExhausted, p.rank, prior, kPublish);
      bad = 1;
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ != 1200
    if (!bad) {
      fail(p, kUnsupportedDevice, p.rank, epoch, kPublish);
      bad = 1;
    }
#endif
  }
  __syncthreads();
  if (!bad) {
    host_publish_payload(input + begin, host_payload(p, p.rank, epoch) + begin,
                         bytes);
    __syncthreads();
    if (threadIdx.x == 0) {
      bad = read_state(p.state + kStateError) != 0;
      if (!bad) {
        HostLane* control = host_lane(p, p.rank, blockIdx.x);
        const unsigned parity = epoch & 1;
        store(&control->descriptor[parity], (total << 3) | kind);
        release(&control->epoch[parity], epoch);
      }
    }
    __syncthreads();
    int failed = bad;
    // Wait for peers in parallel, not a serial sum of their PCIe latencies.
    if (!failed && threadIdx.x < kRanks && threadIdx.x != p.rank)
      failed = !host_wait_ready(p, blockIdx.x, threadIdx.x, epoch, total, kind,
                                started, timeout_ns);
    if (__syncthreads_or(failed) && threadIdx.x == 0) bad = 1;
    __syncthreads();
    if (threadIdx.x == 0 && !bad) bad = read_state(p.state + kStateError) != 0;
    __syncthreads();
    if (!bad) {
      if (kind == kHostReduce)
        host_reduce_payload(p, input, output, epoch, begin, bytes);
      else
        host_gather_payload(p, input, output, epoch, total, begin, bytes);
    }
  }
  // Finish all host reads/output writes before the next stream-ordered
  // publication. Late faults also invalidate completed outputs at CLOSE.
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0) bad = read_state(p.state + kStateError) != 0;
  __syncthreads();
  if (bad) host_zero_output(output, total, begin, bytes, kind);
  device_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const bool last = complete_ticket(p.state + kStateConsumeTicket) ==
                      static_cast<Word>(gridDim.x - 1);
    if (last) {
      // The acq_rel ticket gathers every CTA's completion, including CTAs
      // that failed before publishing. All CTAs have already sampled epoch.
      if (read_state(p.state + kStateError) == 0)
        write_local(p.state + kStateEpoch, epoch);
      write_local(p.state + kStateConsumeTicket, 0);
    }
  }
}
