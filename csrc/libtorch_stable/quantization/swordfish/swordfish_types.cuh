// Swordfish packed-weight ABI v1 constants and metadata. The ABI encodes layout, quant
// metadata, alignment, and versioning only, never launch geometry.
#pragma once

#include <cstdint>

namespace swordfish {

// ---- ABI v1 layout constants ------------------------------------------------
inline constexpr int kAbiVersion = 1;
inline constexpr int kTileLayoutId = 1;  // marlin16x64 / pack_idx{0,2,4,6,1,3,5,7}
                                         // in (NB, KB, 2048B) block-linear order

inline constexpr int kBlockN = 64;   // columns per packed block
inline constexpr int kBlockK = 64;   // K rows per packed block
inline constexpr int kMarlinTileK = 16;
inline constexpr int kMarlinTileN = 64;
inline constexpr int kSubTileBytes = 512;   // one marlin 16x64 int4 tile
inline constexpr int kTilesPerBlock = kBlockK / kMarlinTileK;         // 4
inline constexpr int kBlockBytes = kTilesPerBlock * kSubTileBytes;    // 2048
inline constexpr int kBlockInt32 = kBlockBytes / 4;                   // 512

// 8-bit blocks double every byte figure; the tile permutation is the same.
inline constexpr int kSubTileBytes8 = 2 * kSubTileBytes;
inline constexpr int kBlockBytes8 = 2 * kBlockBytes;
inline constexpr int kBlockInt32_8 = 2 * kBlockInt32;

// ---- schemes (metadata enums; v1 supports exactly one of each) --------------
enum class WeightScheme : uint8_t { kU4B8 = 1, kU8B128 = 2 };
enum class ScaleScheme : uint8_t { kGroupContiguous = 1 };  // [groups][N] fp16/bf16
enum class ZpScheme : uint8_t { kNone = 0 };

// feature_bits (all zero in v1)
inline constexpr uint32_t kFeatHasZp = 1u << 0;
inline constexpr uint32_t kFeatActOrder = 1u << 1;
inline constexpr uint32_t kFeatPaddedTail = 1u << 2;

// ---- shape rules (the v1 tail policy rejects non-multiples) ------------------
__host__ __device__ inline constexpr bool shape_ok(int64_t k, int64_t n) {
  return k > 0 && n > 0 && (k % kBlockK) == 0 && (n % kBlockN) == 0;
}
__host__ __device__ inline constexpr int64_t num_blocks_n(int64_t n) {
  return n / kBlockN;
}
__host__ __device__ inline constexpr int64_t num_blocks_k(int64_t k) {
  return k / kBlockK;
}

}  // namespace swordfish
