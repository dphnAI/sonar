# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support Aphrodite.
"""

import importlib

__all__ = [
    "BagelProcessor",
    "CheersProcessor",
    "CohereASRProcessor",
    "Cosmos3EdgeProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FireRedLIDProcessor",
    "FunASRProcessor",
    "GLM4VProcessor",
    "Granite4VisionProcessor",
    "H2OVLProcessor",
    "Moondream3Processor",
    "InternVLProcessor",
    "IsaacProcessor",
    "KimiAudioProcessor",
    "KimiK25Processor",
    "MiMoOmniProcessor",
    "MiniCPMOProcessor",
    "MiniCPMVProcessor",
    "MiniMaxM3VLImageProcessor",
    "MiniMaxM3VLVideoProcessor",
    "MiniMaxVLProcessor",
    "MistralCommonPixtralProcessor",
    "MistralCommonVoxtralProcessor",
    "NanoNemotronVLProcessor",
    "NemotronVLProcessor",
    "LlamaNemotronVLEmbedProcessor",
    "NVLMProcessor",
    "OpenVLAProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "Qwen3ASRProcessor",
    "Step3VLProcessor",
    "InklingProcessor",
    "InklingImageProcessor",
    "InklingAudioFeatureExtractor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "aphrodite.transformers_utils.processors.bagel",
    "CheersProcessor": "aphrodite.transformers_utils.processors.cheers",
    "CohereASRProcessor": "aphrodite.transformers_utils.processors.cohere_asr",
    "Cosmos3EdgeProcessor": "aphrodite.transformers_utils.processors.cosmos3_edge",
    "DeepseekVLV2Processor": "aphrodite.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "aphrodite.transformers_utils.processors.fireredasr2",
    "FireRedLIDProcessor": "aphrodite.transformers_utils.processors.fireredlid",
    "FunASRProcessor": "aphrodite.transformers_utils.processors.funasr",
    "GLM4VProcessor": "aphrodite.transformers_utils.processors.glm4v",
    "Granite4VisionProcessor": "aphrodite.transformers_utils.processors.granite4_vision",
    "H2OVLProcessor": "aphrodite.transformers_utils.processors.h2ovl",
    "InternVLProcessor": "aphrodite.transformers_utils.processors.internvl",
    "IsaacProcessor": "aphrodite.transformers_utils.processors.isaac",
    "KimiAudioProcessor": "aphrodite.transformers_utils.processors.kimi_audio",
    "KimiK25Processor": "aphrodite.transformers_utils.processors.kimi_k25",
    "MiMoOmniProcessor": "aphrodite.transformers_utils.processors.mimo_v2_omni",
    "MiniCPMOProcessor": "aphrodite.transformers_utils.processors.minicpmo",
    "MiniCPMVProcessor": "aphrodite.transformers_utils.processors.minicpmv",
    "MiniMaxM3VLImageProcessor": "aphrodite.transformers_utils.processors.minimax_m3",
    "MiniMaxM3VLVideoProcessor": "aphrodite.transformers_utils.processors.minimax_m3",
    "MiniMaxVLProcessor": "aphrodite.transformers_utils.processors.minimax_m3",
    "MistralCommonPixtralProcessor": "aphrodite.transformers_utils.processors.pixtral",
    "MistralCommonVoxtralProcessor": "aphrodite.transformers_utils.processors.voxtral",
    "Moondream3Processor": "aphrodite.transformers_utils.processors.moondream3",
    "NanoNemotronVLProcessor": "aphrodite.transformers_utils.processors.nano_nemotron_vl",
    "NemotronVLProcessor": "aphrodite.transformers_utils.processors.nemotron_vl",
    "LlamaNemotronVLEmbedProcessor": "aphrodite.transformers_utils.processors.nemotron_vl",
    "NVLMProcessor": "aphrodite.transformers_utils.processors.nvlm_d",
    "OpenVLAProcessor": "aphrodite.transformers_utils.processors.openvla",
    "OvisProcessor": "aphrodite.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "aphrodite.transformers_utils.processors.ovis2_5",
    "Qwen3ASRProcessor": "aphrodite.transformers_utils.processors.qwen3_asr",
    "Step3VLProcessor": "aphrodite.transformers_utils.processors.step3_vl",
    "InklingProcessor": "aphrodite.transformers_utils.processors.inkling",
    "InklingImageProcessor": "aphrodite.transformers_utils.processors.inkling",
    "InklingAudioFeatureExtractor": "aphrodite.transformers_utils.processors.inkling",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
