include(FetchContent)

FetchContent_Declare(
  dphn_blake3
  GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
  GIT_TAG 93a431c78a52d7ccf0f366f106467f5070e6075e # 1.8.5
  GIT_PROGRESS TRUE)
FetchContent_Declare(
  dphn_monocypher
  GIT_REPOSITORY https://github.com/LoupVaillant/Monocypher.git
  GIT_TAG 0d85f98c9d9b0227e42cf795cb527dff372b40a4 # 4.0.2
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(dphn_blake3 dphn_monocypher)

# Host ATen operations support CPU, CUDA/ROCm and MPS tensors without GPU kernels.
define_extension_target(
  _proof_C
  DESTINATION aphrodite
  LANGUAGE CXX
  SOURCES
    csrc/proof/proof.cpp
    csrc/proof/canonical_cbor.cpp
    csrc/proof/crypto.cpp
    ${dphn_blake3_SOURCE_DIR}/c/blake3.c
    ${dphn_blake3_SOURCE_DIR}/c/blake3_dispatch.c
    ${dphn_blake3_SOURCE_DIR}/c/blake3_portable.c
    ${dphn_monocypher_SOURCE_DIR}/src/monocypher.c
  USE_SABI 3.8
  WITH_SOABI)
target_include_directories(_proof_C PRIVATE
  ${dphn_blake3_SOURCE_DIR}/c
  ${dphn_monocypher_SOURCE_DIR}/src)
set_target_properties(_proof_C PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
# Receipts hash small windows; portable BLAKE3 avoids architecture-specific flags.
target_compile_definitions(_proof_C PRIVATE
  BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512
  BLAKE3_USE_NEON=0)
