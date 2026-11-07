# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration policy for native SM120 DeepSeek-V4.1 execution."""

from typing import TYPE_CHECKING

from aphrodite.platforms import current_platform

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig, ModelConfig


def verify_and_update_model_config(model_config: "ModelConfig") -> None:
    if not current_platform.is_cuda() or not current_platform.is_device_capability_family(120):
        raise ValueError("APHRODITE_DSV41_SM120_NATIVE requires an SM120-family CUDA GPU.")

    for config in (
        model_config.hf_config,
        model_config.hf_text_config,
        model_config.model_arch_config,
    ):
        quant_config = getattr(config, "quantization_config", None)
        if isinstance(quant_config, dict) and quant_config.get("quant_method") in (
            "fp8",
            "deepseek_v4_fp8",
        ):
            quant_config["quant_method"] = "deepseek_v41_fp8"
    if getattr(model_config.hf_config, "head_dtype", None) is None:
        model_config.hf_config.head_dtype = "float32"


def verify_and_update_config(aphrodite_config: "AphroditeConfig") -> None:
    from aphrodite.config.compilation import CompilationMode, CUDAGraphMode
    from aphrodite.v1.attention.backends.registry import AttentionBackendEnum

    backend = AttentionBackendEnum.B12X_DSV41
    attention_config = aphrodite_config.attention_config
    parallel_config = aphrodite_config.parallel_config
    if (
        parallel_config.pipeline_parallel_size != 1
        or parallel_config.decode_context_parallel_size != 1
        or parallel_config.prefill_context_parallel_size != 1
        or parallel_config.enable_dbo
    ):
        raise ValueError("Native SM120 DeepSeek-V4.1 requires TP-only execution without DCP, PCP, PP, or DBO.")
    if attention_config.backend not in (None, backend):
        raise ValueError("Native SM120 DeepSeek-V4.1 requires the B12X_DSV41 attention backend.")
    attention_config.backend = backend
    if not aphrodite_config.cache_config.user_specified_block_size:
        aphrodite_config.cache_config.block_size = 256
    compilation_config = aphrodite_config.compilation_config
    if compilation_config.mode is None:
        compilation_config.mode = CompilationMode.NONE
    if compilation_config.cudagraph_mode is None and not aphrodite_config.model_config.enforce_eager:
        compilation_config.cudagraph_mode = CUDAGraphMode.FULL
