// Swordfish ABI v1 address math, shared by the prepack kernel and both
// mainloops. Single source of truth for the (NB, KB, tile) block-linear
// layout; the in-tile permutation is Marlin's.
#pragma once

#include "swordfish_types.cuh"

namespace swordfish {

// Byte offset of packed block (nb, kb) in a (NB, KB, kBlockBytes) tensor.
__host__ __device__ inline constexpr int64_t block_byte_offset(int64_t nb,
                                                               int64_t kb,
                                                               int64_t num_kb) {
  return (nb * num_kb + kb) * kBlockBytes;
}

// Byte offset of the k16-slice sub-tile `t` (0..3) within a block.
__host__ __device__ inline constexpr int64_t subtile_byte_offset(int t) {
  return int64_t(t) * kSubTileBytes;
}

// Mapping from the Marlin flat repack layout to Swordfish blocks.
//
// gptq_marlin_repack emits int32[K/16][N*16/8] = int32[K/16][N*2]; row r is
// the k16-slice starting at k = 16*r; within a row, each Marlin 16x64 n-tile
// occupies 128 consecutive int32 (64 cols * 16 rows / 8 nibbles-per-int32).
// Swordfish block (nb, kb) gathers rows {4*kb .. 4*kb+3}, int32 columns
// [128*nb, 128*(nb+1)), i.e. int32 index
//   marlin_idx(row, col) = row * (N*2) + col
//   swordfish word w of block (nb,kb):  t = w / 128, c = w % 128
//     -> marlin row = 4*kb + t, marlin col = 128*nb + c
__host__ __device__ inline constexpr int64_t marlin_word_index(
    int64_t nb, int64_t kb, int w, int64_t n) {
  const int t = w / 128;
  const int c = w % 128;
  return (4 * kb + t) * (n * 2) + 128 * nb + c;
}

}  // namespace swordfish
