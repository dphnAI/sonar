#!/bin/sh

ROCM_PATH=$(hipconfig --rocmpath)

sudo patch "$ROCM_PATH"/lib/llvm/lib/clang/*/include/__clang_hip_cmath.h ./patches/amd.patch
