# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support Aphrodite.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "AfmoeConfig": "aphrodite.transformers_utils.configs.afmoe",
    "AXK1Config": "aphrodite.transformers_utils.configs.AXK1",
    "BagelConfig": "aphrodite.transformers_utils.configs.bagel",
    "CheersConfig": "aphrodite.transformers_utils.configs.cheers",
    "ChatGLMConfig": "aphrodite.transformers_utils.configs.chatglm",
    "ColModernVBertConfig": "aphrodite.transformers_utils.configs.colmodernvbert",
    "ColPaliConfig": "aphrodite.transformers_utils.configs.colpali",
    "ColQwen3Config": "aphrodite.transformers_utils.configs.colqwen3",
    "OpsColQwen3Config": "aphrodite.transformers_utils.configs.colqwen3",
    "Qwen3VLNemotronEmbedConfig": "aphrodite.transformers_utils.configs.colqwen3",
    "DeepseekVLV2Config": "aphrodite.transformers_utils.configs.deepseek_vl2",
    "DeepseekV4Config": "aphrodite.transformers_utils.configs.deepseek_v4",
    "DotsOCRConfig": "aphrodite.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "aphrodite.transformers_utils.configs.eagle",
    "FireRedLIDConfig": "aphrodite.transformers_utils.configs.fireredlid",
    "FlexOlmoConfig": "aphrodite.transformers_utils.configs.flex_olmo",
    "FunAudioChatConfig": "aphrodite.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "aphrodite.transformers_utils.configs.funaudiochat",
    "Granite4VisionConfig": "aphrodite.transformers_utils.configs.granite4_vision",
    "HunYuanVLConfig": "aphrodite.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLTextConfig": "aphrodite.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLVisionConfig": "aphrodite.transformers_utils.configs.hunyuan_vl",
    "HYV3Config": "aphrodite.transformers_utils.configs.hy_v3",
    "HyperCLOVAXConfig": "aphrodite.transformers_utils.configs.hyperclovax",
    "IsaacConfig": "aphrodite.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "aphrodite.transformers_utils.configs.falcon",
    "JAISConfig": "aphrodite.transformers_utils.configs.jais",
    "LagunaConfig": "aphrodite.transformers_utils.configs.laguna",
    "Lfm2MoeConfig": "aphrodite.transformers_utils.configs.lfm2_moe",
    "MedusaConfig": "aphrodite.transformers_utils.configs.medusa",
    "MiDashengLMConfig": "aphrodite.transformers_utils.configs.midashenglm",
    "MLPSpeculatorConfig": "aphrodite.transformers_utils.configs.mlp_speculator",
    "Moondream3Config": "aphrodite.transformers_utils.configs.moondream3",
    "Moondream3TextConfig": "aphrodite.transformers_utils.configs.moondream3",
    "Moondream3VisionConfig": "aphrodite.transformers_utils.configs.moondream3",
    "MoonViTConfig": "aphrodite.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "aphrodite.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "aphrodite.transformers_utils.configs.kimi_vl",
    "KimiK25Config": "aphrodite.transformers_utils.configs.kimi_k25",
    "NemotronConfig": "aphrodite.transformers_utils.configs.nemotron",
    "NemotronHConfig": "aphrodite.transformers_utils.configs.nemotron_h",
    "OlmoHybridConfig": "aphrodite.transformers_utils.configs.olmo_hybrid",
    "OvisConfig": "aphrodite.transformers_utils.configs.ovis",
    "PixelShuffleSiglip2VisionConfig": "aphrodite.transformers_utils.configs.isaac",
    "RadioConfig": "aphrodite.transformers_utils.configs.radio",
    "SpeculatorsConfig": "aphrodite.transformers_utils.configs.speculators",
    "UltravoxConfig": "aphrodite.transformers_utils.configs.ultravox",
    "Step3VLConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "Step3p5Config": "aphrodite.transformers_utils.configs.step3p5",
    "Qwen3ASRConfig": "aphrodite.transformers_utils.configs.qwen3_asr",
    "Qwen3NextConfig": "aphrodite.transformers_utils.configs.qwen3_next",
    "Qwen3_5Config": "aphrodite.transformers_utils.configs.qwen3_5",
    "Qwen3_5TextConfig": "aphrodite.transformers_utils.configs.qwen3_5",
    "Qwen3_5MoeConfig": "aphrodite.transformers_utils.configs.qwen3_5_moe",
    "Qwen3_5MoeTextConfig": "aphrodite.transformers_utils.configs.qwen3_5_moe",
    "Tarsier2Config": "aphrodite.transformers_utils.configs.tarsier2",
    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
}

__all__ = [
    "AfmoeConfig",
    "AXK1Config",
    "BagelConfig",
    "CheersConfig",
    "ChatGLMConfig",
    "ColModernVBertConfig",
    "ColPaliConfig",
    "ColQwen3Config",
    "OpsColQwen3Config",
    "Qwen3VLNemotronEmbedConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DeepseekV4Config",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FlexOlmoConfig",
    "FireRedLIDConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "Granite4VisionConfig",
    "HunYuanVLConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLVisionConfig",
    "HYV3Config",
    "HyperCLOVAXConfig",
    "IsaacConfig",
    "RWConfig",
    "JAISConfig",
    "LagunaConfig",
    "Lfm2MoeConfig",
    "MedusaConfig",
    "MiDashengLMConfig",
    "MLPSpeculatorConfig",
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "KimiK25Config",
    "NemotronConfig",
    "NemotronHConfig",
    "OlmoHybridConfig",
    "OvisConfig",
    "PixelShuffleSiglip2VisionConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Step3p5Config",
    "Qwen3ASRConfig",
    "Qwen3NextConfig",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Tarsier2Config",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
