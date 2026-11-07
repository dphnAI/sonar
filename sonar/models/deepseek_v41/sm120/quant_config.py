# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM120 quantization for the original V4.1 FP8/FP4 checkpoint."""

from importlib.util import find_spec
from pathlib import Path

import torch

import aphrodite.envs as envs
from aphrodite.config import get_current_aphrodite_config
from aphrodite.model_executor.layers.fused_moe import RoutedExperts
from aphrodite.model_executor.layers.linear import LinearBase
from aphrodite.model_executor.layers.quantization.fp8 import Fp8Config
from aphrodite.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from aphrodite.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from aphrodite.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from aphrodite.platforms import current_platform


class DeepseekV41FP8Config(Fp8Config):
    is_scale_e8m0 = True

    @staticmethod
    def _native_enabled():
        return (
            envs.APHRODITE_DSV41_SM120_NATIVE
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight_block_size != [32, 32]:
            raise ValueError("V4.1 native execution requires checkpoint block32 FP8 weights")
        if self.activation_scheme != "dynamic" or self.store_dtype is not None:
            raise ValueError("V4.1 native execution requires dynamic BF16 activations and serialized FP8 weights")

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def from_config(cls, config):
        if not cls._native_enabled():
            raise ValueError("deepseek_v41_fp8 requires APHRODITE_DSV41_SM120_NATIVE=1 on an SM120-family CUDA GPU")
        if config.get("quant_method") not in ("fp8", cls.get_name()) or config.get("moe_quant_algo"):
            raise ValueError("V4.1 native execution supports only the original block32 FP8/MXFP4 checkpoint")
        try:
            import b12x
        except ImportError as error:
            raise ValueError("Native SM120 execution requires FlashInfer's bundled B12x.") from error
        flashinfer = find_spec("flashinfer")
        provider_file = getattr(b12x, "__file__", None)
        if (
            flashinfer is None
            or flashinfer.origin is None
            or provider_file is None
            or not Path(provider_file)
            .resolve()
            .is_relative_to(Path(flashinfer.origin).resolve().parent / "experimental" / "b12x")
        ):
            raise ValueError(
                "Native SM120 execution requires FlashInfer's bundled B12x. "
                "Remove the standalone b12x package and reinstall flashinfer-python==0.7.2."
            )
        return super().from_config(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if not cls._native_enabled():
            return None
        if getattr(hf_config, "model_type", None) in (
            "deepseek_v41",
            "deepseek_v41_text",
        ) and hf_quant_cfg.get("quant_method") in ("fp8", cls.get_name()):
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            from .b12x_layers import B12xFP8LinearMethod, B12xLinearMethod

            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return B12xLinearMethod()
            return B12xFP8LinearMethod(self, shared_expert="shared_experts" in prefix.split("."))
        if isinstance(layer, VocabParallelEmbedding):
            from .b12x_layers import B12xEmbeddingMethod

            return B12xEmbeddingMethod()
        if isinstance(layer, RoutedExperts):
            hf_config = get_current_aphrodite_config().model_config.hf_text_config
            if getattr(hf_config, "expert_dtype", "fp4") != "fp4":
                raise ValueError("V4.1 native execution requires the original MXFP4 experts")
            return Mxfp4MoEMethod(layer.moe_config)
        return None
