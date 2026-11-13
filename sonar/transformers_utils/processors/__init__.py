# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support Sonar.
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
    "Lfm3AudioProcessor",
    "Lfm3VlProcessor",
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
    "BagelProcessor": "sonar.transformers_utils.processors.bagel",
    "CohereASRProcessor": "sonar.transformers_utils.processors.cohere_asr",
    "DeepseekVLV2Processor": "sonar.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "sonar.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "sonar.transformers_utils.processors.funasr",
    "GLM4VProcessor": "sonar.transformers_utils.processors.glm4v",
    "H2OVLProcessor": "sonar.transformers_utils.processors.h2ovl",
    "IsaacProcessor": "sonar.transformers_utils.processors.isaac",
    "KimiAudioProcessor": "sonar.transformers_utils.processors.kimi_audio",
    "KimiK25Processor": "sonar.transformers_utils.processors.kimi_k25",
    "KimiK3Processor": "sonar.transformers_utils.processors.kimi_k3",
    "Lfm3AudioProcessor": "sonar.transformers_utils.processors.lfm3_audio",
    "Lfm3VlProcessor": "sonar.transformers_utils.processors.lfm3_vl",
    "MiMoOmniProcessor": "sonar.transformers_utils.processors.mimo_v2_omni",
    "MingASRFeatureExtractor": "sonar.transformers_utils.processors.ming_audio",
    "MiniCPMOProcessor": "sonar.transformers_utils.processors.minicpmo",
    "MiniCPMVProcessor": "sonar.transformers_utils.processors.minicpmv",
    "MiniMaxM3VLImageProcessor": "sonar.transformers_utils.processors.minimax_m3",
    "MiniMaxM3VLVideoProcessor": "sonar.transformers_utils.processors.minimax_m3",
    "MiniMaxVLProcessor": "sonar.transformers_utils.processors.minimax_m3",
    "MistralCommonPixtralProcessor": "sonar.transformers_utils.processors.pixtral",
    "MistralCommonVoxtralProcessor": "sonar.transformers_utils.processors.voxtral",
    "Moondream3Processor": "sonar.transformers_utils.processors.moondream3",
    "NanoNemotronVLProcessor": "sonar.transformers_utils.processors.nano_nemotron_vl",
    "NemotronVLProcessor": "sonar.transformers_utils.processors.nemotron_vl",
    "LlamaNemotronVLEmbedProcessor": "sonar.transformers_utils.processors.nemotron_vl",
    "NVLMProcessor": "sonar.transformers_utils.processors.nvlm_d",
    "OpenVLAProcessor": "sonar.transformers_utils.processors.openvla",
    "OvisProcessor": "sonar.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "sonar.transformers_utils.processors.ovis2_5",
    "Step3VLProcessor": "sonar.transformers_utils.processors.step3_vl",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
