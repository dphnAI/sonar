# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from aphrodite.model_executor.kernels.mhc.tilelang_kernels import ENABLE_PDL
from aphrodite.tilelang_utils import T, tilelang_jit


@tilelang_jit
def mhc_warp_partials_sm120(x, fn, out, sqrsum, source_threads: int):
    """Preserve each source lane's FP32 FMA chain and warp reduction.

    ``source_threads`` is the original projection's 512 or 1024 threads.
    The unchanged delayed epilogue adds these partials in source warp order.
    """
    m = T.dynamic("num_tokens")
    warps = source_threads // 32
    x: T.Tensor[[m, 20480], T.bfloat16]  # type: ignore[no-redef, valid-type]
    fn: T.Tensor[[24, 20480], T.float32]  # type: ignore[no-redef, valid-type]
    out: T.Tensor[[warps, m, 24], T.float32]  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor[[warps, m], T.float32]  # type: ignore[no-redef, valid-type]
    with T.Kernel(m, 6, warps, threads=32) as (row, tile, warp):
        lane = T.get_thread_binding()
        acc = T.alloc_local((4,), T.float32)
        square = T.alloc_local((1,), T.float32)
        T.clear(acc)
        T.clear(square)
        if ENABLE_PDL:
            T.pdl_sync()
        for it in T.serial(20480 // source_threads):
            k = it * source_threads + warp * 32 + lane
            value = T.cast(x[row, k], T.float32)
            for o in T.unroll(4):
                acc[o] += value * fn[tile * 4 + o, k]
            if tile == 0:
                square[0] += value * value
        for o in T.unroll(4):
            acc[o] = T.warp_reduce_sum(acc[o])
        if tile == 0:
            square[0] = T.warp_reduce_sum(square[0])
        if lane == 0:
            for o in T.unroll(4):
                out[warp, row, tile * 4 + o] = acc[o]
            if tile == 0:
                sqrsum[warp, row] = square[0]
        if ENABLE_PDL:
            T.pdl_trigger()
