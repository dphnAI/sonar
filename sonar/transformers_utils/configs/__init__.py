# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support Aphrodite.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "Alpamayo2SuperConfig": "aphrodite.transformers_utils.configs.alpamayo2_super",
    "BagelConfig": "aphrodite.transformers_utils.configs.bagel",
    "BailingMoeV3TextConfig": "aphrodite.transformers_utils.configs.bailing_moe_v3_vl",
    "BailingMoeV3VisionConfig": "aphrodite.transformers_utils.configs.bailing_moe_v3_vl",
    "BailingMoeV3VLConfig": "aphrodite.transformers_utils.configs.bailing_moe_v3_vl",
    "BeamConfig": "aphrodite.transformers_utils.configs.beam",
    "ChatGLMConfig": "aphrodite.transformers_utils.configs.chatglm",
    "ColUltraVBertConfig": "aphrodite.transformers_utils.configs.colultravbert",
    "ColQwen3Config": "aphrodite.transformers_utils.configs.colqwen3",
    "OpsColQwen3Config": "aphrodite.transformers_utils.configs.colqwen3",
    "Qwen3VLNemotronEmbedConfig": "aphrodite.transformers_utils.configs.colqwen3",
    "DeepseekVLV2Config": "aphrodite.transformers_utils.configs.deepseek_vl2",
    "DeepseekV41Config": "aphrodite.transformers_utils.configs.deepseek_v41",
    "DoryConfig": "aphrodite.transformers_utils.configs.dory",
    "Dots3NoteConfig": "aphrodite.transformers_utils.configs.dots3_note",
    "K3DSparkConfig": "aphrodite.transformers_utils.configs.k3_dspark",
    "DotsOCRConfig": "aphrodite.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "aphrodite.transformers_utils.configs.eagle",
    "FunAudioChatConfig": "aphrodite.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "aphrodite.transformers_utils.configs.funaudiochat",
    "HYV4Config": "aphrodite.transformers_utils.configs.hy_v4",
    "IsaacConfig": "aphrodite.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "aphrodite.transformers_utils.configs.falcon",
    "MiDashengLMConfig": "aphrodite.transformers_utils.configs.midashenglm",
    "MingV3ASRConfig": "aphrodite.transformers_utils.configs.ming_v3_asr",
    "MiniMaxM3Config": "aphrodite.transformers_utils.configs.minimax_m3",
    "MiniMaxM3MTPConfig": "aphrodite.transformers_utils.configs.minimax_m3",
    "MiniMaxM3TextConfig": "aphrodite.transformers_utils.configs.minimax_m3",
    "Moondream3Config": "aphrodite.transformers_utils.configs.moondream3",
    "Moondream3TextConfig": "aphrodite.transformers_utils.configs.moondream3",
    "Moondream3VisionConfig": "aphrodite.transformers_utils.configs.moondream3",
    "MossTranscribeDiarizeConfig": ("aphrodite.transformers_utils.configs.moss_transcribe_diarize"),
    "Nemotron3_5TranscribeConfig": ("aphrodite.transformers_utils.configs.nemotron3_5_transcribe"),
    "MoonViTConfig": "aphrodite.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "aphrodite.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "aphrodite.transformers_utils.configs.kimi_vl",
    "MuseSpark1xTextConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "MuseSparkTextConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "MuseSparkVisionConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "MuseSparkAudioConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "MuseSparkDSparkConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "MuseSparkConfig": "aphrodite.transformers_utils.configs.muse_spark",
    "KimiK3Config": "aphrodite.transformers_utils.configs.kimi_k3",
    "KimiK3VisionConfig": "aphrodite.transformers_utils.configs.kimi_k3",
    "OpenVLAConfig": "aphrodite.transformers_utils.configs.openvla",
    "OvisConfig": "aphrodite.transformers_utils.configs.ovis",
    "PixelShuffleSiglip2VisionConfig": "aphrodite.transformers_utils.configs.isaac",
    "RadioConfig": "aphrodite.transformers_utils.configs.radio",
    "SpeculatorsConfig": "aphrodite.transformers_utils.configs.speculators",
    "UltravoxConfig": "aphrodite.transformers_utils.configs.ultravox",
    "UnlimitedOCRConfig": "aphrodite.transformers_utils.configs.unlimited_ocr",
    "Step3VLConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "aphrodite.transformers_utils.configs.step3_vl",
    "PerceptionEncoderConfig": "aphrodite.transformers_utils.configs.step5",
    "Step5Config": "aphrodite.transformers_utils.configs.step5",
    "Step5VLConfig": "aphrodite.transformers_utils.configs.step5",
    "Qwen3ASRConfig": "aphrodite.transformers_utils.configs.qwen3_asr",
    "FlintConfig": "aphrodite.models.flint.configs",
    "InklingModelConfig": "aphrodite.models.inkling.configs",
    "InklingAudioConfig": "aphrodite.models.inkling.configs",
    "InklingVisionConfig": "aphrodite.models.inkling.configs",
    "InklingMMConfig": "aphrodite.models.inkling.configs",
    # Upstream Transformers classes registered in _CONFIG_REGISTRY
    "DeepseekV3Config": "transformers",
    "HyperCLOVAXConfig": "transformers",
    "Kimi_K25Config": "transformers",
    "Step3p7TextConfig": "transformers",
}

__all__ = [
    "Alpamayo2SuperConfig",
    "BagelConfig",
    "BailingMoeV3TextConfig",
    "BailingMoeV3VisionConfig",
    "BailingMoeV3VLConfig",
    "BeamConfig",
    "ChatGLMConfig",
    "ColUltraVBertConfig",
    "ColQwen3Config",
    "OpsColQwen3Config",
    "Qwen3VLNemotronEmbedConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DoryConfig",
    "Dots3NoteConfig",
    "K3DSparkConfig",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "HYV4Config",
    "HyperCLOVAXConfig",
    "IsaacConfig",
    "RWConfig",
    "MiDashengLMConfig",
    "MingV3ASRConfig",
    "MiniMaxM3Config",
    "MiniMaxM3MTPConfig",
    "MiniMaxM3TextConfig",
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "MossTranscribeDiarizeConfig",
    "Nemotron3_5TranscribeConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "Kimi_K25Config",
    "MuseSpark1xTextConfig",
    "MuseSparkTextConfig",
    "MuseSparkVisionConfig",
    "MuseSparkAudioConfig",
    "MuseSparkDSparkConfig",
    "MuseSparkConfig",
    "KimiK3Config",
    "KimiK3VisionConfig",
    "OpenVLAConfig",
    "OvisConfig",
    "PixelShuffleSiglip2VisionConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "UnlimitedOCRConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Step3p7TextConfig",
    "PerceptionEncoderConfig",
    "Step5Config",
    "Step5VLConfig",
    "Qwen3ASRConfig",
    "FlintConfig",
    "InklingModelConfig",
    "InklingAudioConfig",
    "InklingVisionConfig",
    "InklingMMConfig",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
