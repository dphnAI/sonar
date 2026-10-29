# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch

from aphrodite.config import get_current_aphrodite_config
from aphrodite.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from aphrodite.model_executor.layers.linear import LinearBase, MergedColumnParallelLinear
from aphrodite.model_executor.layers.quantization import QuantizationConfig
from aphrodite.model_executor.utils import is_weight_cache_model_construction
from aphrodite.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant
from aphrodite.platforms import current_platform


@cache
def register_sm120_engram_ops() -> None:
    """Load native schemas and install their graph implementations once."""
    if not current_platform.is_cuda():
        return
    current_platform.import_kernels()

    if hasattr(torch.ops._C, "sm120_engram_pack"):

        @torch.library.register_fake("_C::sm120_engram_pack")
        def _sm120_engram_pack_fake(
            hidden: torch.Tensor,
            local_projection: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            packet: torch.Tensor,
            rank: int,
            num_valid_tokens: int,
        ) -> None:
            return None

    if hasattr(torch.ops._C, "sm120_engram_finish"):

        @torch.library.register_fake("_C::sm120_engram_finish")
        def _sm120_engram_finish_fake(
            hidden: torch.Tensor,
            gathered_packets: torch.Tensor,
            token_mask: torch.Tensor | None,
            output: torch.Tensor,
            num_valid_tokens: int,
            eps: float,
            clamp_value: float,
        ) -> None:
            return None


def create_sm120_engram_wkv(
    input_size: int,
    hidden_size: int,
    hc_mult: int,
    eps: float,
    clamp_value: float,
    use_sequence_parallel: bool,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> MergedColumnParallelLinear | None:
    """Choose the striped checkpoint layout only with a complete native path."""
    if (
        not current_platform.is_cuda()
        or use_sequence_parallel
        or (input_size, hidden_size, hc_mult) != (6144, 5120, 4)
        or eps != 1e-20
        or clamp_value != 1e-6
        or quant_config is None
        or quant_config.online_quantization_config is not None
        or is_weight_cache_model_construction()
    ):
        return None
    config = get_current_aphrodite_config()
    parallel = config.parallel_config
    if (
        config.lora_config is not None
        or config.model_config is None
        or config.model_config.dtype != torch.bfloat16
        or torch.get_default_dtype() != torch.bfloat16
        or get_tensor_model_parallel_world_size() != 4
        or parallel.tensor_parallel_size != 4
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        return None
    if not torch.accelerator.is_available() or not current_platform.is_device_capability(
        120, torch.accelerator.current_device_index()
    ):
        return None

    register_sm120_engram_ops()
    from aphrodite.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from aphrodite.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from aphrodite.model_executor.layers.quantization.modelopt import (
        FormatScheme,
        ModelOptLinearMethod,
        ModelOptMixedPrecisionConfig,
        ModelOptMxFp8Config,
    )
    from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )
    from aphrodite.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    if type(quant_config) not in (
        ModelOptMixedPrecisionConfig,
        ModelOptMxFp8Config,
        DeepseekV4FP8Config,
    ):
        return None
    for name in ("sm120_engram_pack", "sm120_engram_finish"):
        if not hasattr(torch.ops._C, name) or not torch._C._dispatch_has_kernel_for_dispatch_key(f"_C::{name}", "CUDA"):
            return None

    # LinearBase resolves the per-prefix method without creating checkpoint
    # tensors. Online overrides are excluded above, keeping this probe read-only.
    probe = LinearBase(input_size, 25600, quant_config=quant_config, prefix=prefix)
    method = probe.quant_method
    if (
        type(method) is not ModelOptLinearMethod
        or method.spec.weight != kMxfp8Static
        or method.spec.activation != kMxfp8Dynamic
        or type(method.fmt) is not FormatScheme
        or method.ctx.scale_block_size not in (None, (1, 32), (32, 32))
        or method.input_dtype != torch.bfloat16
        or method.out_dtype != torch.bfloat16
    ):
        return None
    kernel = init_mxfp8_linear_kernel(weight_shape=(6400, input_size))
    if type(kernel) not in (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    ):
        return None

    # Each 1024-row band contributes rank*256:rank*256+256, for all five
    # key/value components. MXFP8 stores both values and E8M0 scales in
    # ModelWeightParameter(output_dim=0), so the fused loader stripes both.
    wkv = MergedColumnParallelLinear(
        input_size,
        [1024] * 25,
        bias=False,
        gather_output=False,
        quant_config=quant_config,
        return_bias=False,
        prefix=prefix,
    )
    assert type(wkv.quant_method) is ModelOptLinearMethod
    assert type(wkv.quant_method.kernel) is type(kernel) and can_fuse_query_quant([wkv])
    return wkv


def sm120_engram_post(
    hidden: torch.Tensor,
    local_projection: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: torch.Tensor | None,
    num_valid_tokens: int,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """Gather rank-major partial reductions and striped BF16 values."""
    output = torch.empty_like(hidden)
    if hidden.shape[0] == 0:
        return output
    # Per-call graph-pool allocations must not alias another captured shape's
    # output or scratch. The collective already allocates its gathered output.
    packet = torch.empty((hidden.shape[0], 2624), dtype=torch.uint8, device=hidden.device)
    torch.ops._C.sm120_engram_pack(
        hidden, local_projection, q_weight, k_weight, packet, get_tensor_model_parallel_rank(), num_valid_tokens
    )
    gathered = tensor_model_parallel_all_gather(packet, dim=0)
    torch.ops._C.sm120_engram_finish(hidden, gathered, token_mask, output, num_valid_tokens, eps, clamp_value)
    return output
