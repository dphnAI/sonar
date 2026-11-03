# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support Aphrodite.
"""

import importlib

__all__ = [
    "BagelProcessor",
    "CohereASRProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "GLM4VProcessor",
    "H2OVLProcessor",
    "Moondream3Processor",
    "IsaacProcessor",
    "KimiAudioProcessor",
    "KimiK25Processor",
    "KimiK3Processor",
    "MiMoOmniProcessor",
    "MingASRFeatureExtractor",
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
    "Step3VLProcessor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "aphrodite.transformers_utils.processors.bagel",
    "CohereASRProcessor": "aphrodite.transformers_utils.processors.cohere_asr",
    "DeepseekVLV2Processor": "aphrodite.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "aphrodite.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "aphrodite.transformers_utils.processors.funasr",
    "GLM4VProcessor": "aphrodite.transformers_utils.processors.glm4v",
    "H2OVLProcessor": "aphrodite.transformers_utils.processors.h2ovl",
    "IsaacProcessor": "aphrodite.transformers_utils.processors.isaac",
    "KimiAudioProcessor": "aphrodite.transformers_utils.processors.kimi_audio",
    "KimiK25Processor": "aphrodite.transformers_utils.processors.kimi_k25",
    "KimiK3Processor": "aphrodite.transformers_utils.processors.kimi_k3",
    "MiMoOmniProcessor": "aphrodite.transformers_utils.processors.mimo_v2_omni",
    "MingASRFeatureExtractor": "aphrodite.transformers_utils.processors.ming_audio",
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
    "Step3VLProcessor": "aphrodite.transformers_utils.processors.step3_vl",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
