<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# Vendored CuTe DSL source

`cutedsl_kernel.py` is derived from NVIDIA CUTLASS commit
`f94ec46f4f63f96003d6cfdf2014731e7672c281`:

`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py`

Sonar integrates the kernel with its packed-weight ABI and Torch custom
operator in `cutedsl.py`. The upstream command-line reference helpers remain
in the vendored file to make future CUTLASS updates easier.

`cutedsl_grouped_kernel.py` derives from the grouped block-scaled GEMM example
at the same CUTLASS commit. It ports mixed-width operand and FP6 TMA-unpack
support from the dense implementation. NVIDIA's grouped example only accepted
same-width operands at that revision.

The native online path uses `Mxfp6LinearKernel` and
`Mxfp6LinearLayerConfig` with `CutedslMxfp6LinearKernel`; it is gated to
SM110 and retains separate dense and grouped packed-weight layouts.
Checkpoint MXFP6 selection uses `MxFp6LinearKernel` and
`MxFp6LinearLayerConfig` with Humming or software emulation. These kernel
interfaces are distinct: the checkpoint selector must not substitute
emulation for the native online conversion path. Both use the canonical
`kMxfp6E2M3*` and `kMxfp6E3M2*` quantization keys.

The CuTe DSL 4.8 integration uses separate A/B dtypes in Blackwell MMA
helpers and supplies scale tensors as auxiliary `cute.gemm` operands for
both dense and grouped kernels. Scale-factor SMEM/TMEM layouts remain
provided by `cutlass.utils.blockscaled_layout`; no direct NVVM
`tcgen05_mma_block_scale` enum binding is used in these kernels.
