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
__device__ __forceinline__ void host_local_store_bf16(unsigned char* pointer,
                                                      uint4 value) {
  if ((reinterpret_cast<uintptr_t>(pointer) & 15) == 0) {
    *reinterpret_cast<uint4*>(pointer) = value;
  } else {
    auto* destination = reinterpret_cast<unsigned short*>(pointer);
    const unsigned words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      destination[2 * pair] = static_cast<unsigned short>(words[pair]);
      destination[2 * pair + 1] =
          static_cast<unsigned short>(words[pair] >> 16);
    }
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
                                Word started, Word timeout_ns,
                                bool reduced = false) {
  const HostLane* control = host_lane(p, peer, lane);
  const Word* publication = reduced ? control->reduced_ready : control->epoch;
  const Word stage = reduced ? kConsume : kReady;
  const unsigned parity = epoch & 1;
  unsigned misses = 0;
  for (;;) {
    const Word seen = acquire(&publication[parity]);
    if (seen == epoch) {
      if (acquire(&control->descriptor[parity]) != ((bytes << 3) | kind)) {
        fail(p, kMetadataMismatch, peer, epoch, stage);
        return false;
      }
      return true;
    }
    if (seen > epoch) {
      fail(p, kFutureEpoch, peer, epoch, stage);
      return false;
    }
    if ((++misses & 31U) == 0) {
      if (observe_fault(p, stage)) return false;
      if (timer_ns() - started >= timeout_ns) {
        fail(p, kReadyTimeout, peer, epoch, stage);
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
// Only a sharded lane's owner reads its inputs. Each thread may therefore
// replace its disjoint part of the owner's host input with the reduced result.
template <bool Sharded>
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
    const uint4 result = make_uint4(words[0], words[1], words[2], words[3]);
    host_local_store_bf16(output + offset, result);
    if constexpr (Sharded)
      host_store_vector(host_payload(p, p.rank, epoch) + offset, result);
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
    const unsigned bits = host_to_bf16(sum);
    *reinterpret_cast<unsigned short*>(output + offset) =
        static_cast<unsigned short>(bits);
    if constexpr (Sharded) {
      unsigned char* destination = host_payload(p, p.rank, epoch) + offset;
      host_store_byte(destination, bits);
      host_store_byte(destination + 1, bits >> 8);
    }
  }
}
__device__ void host_copy_reduced_payload(const unsigned char* reduced,
                                          unsigned char* output, uint64_t begin,
                                          uint64_t bytes) {
  const uint64_t vectors = bytes / 16;
  for (uint64_t i = threadIdx.x; i < vectors; i += blockDim.x) {
    const uint64_t offset = begin + 16 * i;
    host_local_store_bf16(output + offset, host_load_vector(reduced + offset));
  }
  for (uint64_t i = vectors * 16 + 2 * threadIdx.x; i < bytes;
       i += 2 * blockDim.x) {
    const unsigned char* source = reduced + begin + i;
    const unsigned bits =
        host_load_byte(source) | (host_load_byte(source + 1) << 8);
    *reinterpret_cast<unsigned short*>(output + begin + i) =
        static_cast<unsigned short>(bits);
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

// One CTA per lane; never a residency-dependent inter-CTA barrier. Small
// reductions and gathers use direct 64KiB lanes; large reductions use 16KiB
// lanes whose sole reader/reducer is blockIdx.x % kRanks.
//
// The checked owner has one global epoch, not per-lane epochs. Every active
// lane proves all ranks published E+1, either by checking input-ready directly
// or transitively through its owner's reduced-ready. Each publication follows
// that rank's completed kernel and thus ALL its reads of E. This proof does not
// depend on lane boundaries/counts: even shrinking, switching geometry or
// reactivating a lane permits E+2 to reuse E's raw parity and metadata. CLOSE's
// separate all-rank publish/consume kernels supply the same stream-ordered
// proof. The final completion ticket advances the global epoch exactly once.
template <bool Sharded>
__global__ void host_collective(const __grid_constant__ Params p,
                                const unsigned char* input,
                                unsigned char* output, uint64_t total,
                                Word kind) {
  constexpr uint64_t lane_bytes =
      Sharded ? kHostShardedLaneBytes : kHostLaneBytes;
  constexpr bool sharded = Sharded;
  const int owner = blockIdx.x % kRanks;
  const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * lane_bytes;
  const uint64_t remaining = total - begin;
  const uint64_t bytes = remaining < lane_bytes ? remaining : lane_bytes;
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
    // Only the owner reads sharded inputs. Its reduced-ready transitively
    // certifies every input-ready and descriptor for the nonowners.
    if (!failed) {
      if (!sharded || p.rank == owner) {
        // Check peers in parallel, not a serial sum of their PCIe latencies.
        if (threadIdx.x < kRanks && threadIdx.x != p.rank)
          failed = !host_wait_ready(p, blockIdx.x, threadIdx.x, epoch, total,
                                    kind, started, timeout_ns);
      } else if (threadIdx.x == 0) {
        failed = !host_wait_ready(p, blockIdx.x, owner, epoch, total, kind,
                                  started, timeout_ns, true);
      }
    }
    if (__syncthreads_or(failed) && threadIdx.x == 0) bad = 1;
    __syncthreads();
    if (threadIdx.x == 0 && !bad) bad = read_state(p.state + kStateError) != 0;
    __syncthreads();
    if (!bad) {
      if (kind == kHostGather) {
        host_gather_payload(p, input, output, epoch, total, begin, bytes);
      } else if (!sharded) {
        host_reduce_payload<false>(p, input, output, epoch, begin, bytes);
      } else if (p.rank == owner) {
        host_reduce_payload<true>(p, input, output, epoch, begin, bytes);
        // Drain every reduced-payload writer before releasing availability;
        // the input-ready/descriptor publication remains untouched.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          bad = read_state(p.state + kStateError) != 0;
          if (!bad) {
            HostLane* control = host_lane(p, p.rank, blockIdx.x);
            release(&control->reduced_ready[epoch & 1], epoch);
          }
        }
      } else {
        host_copy_reduced_payload(host_payload(p, owner, epoch), output, begin,
                                  bytes);
      }
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
