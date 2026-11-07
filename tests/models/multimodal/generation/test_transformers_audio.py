# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import AutoModelForSeq2SeqLM

from sonar.assets.audio import AudioAsset
from sonar.envs import disable_envs_cache
from sonar.lora.request import LoRARequest
from sonar.multimodal.audio import AudioResampler

from ....conftest import HfRunner, SonarRunner
from ...utils import check_logprobs_close

AUDIO_ASSET = AudioAsset("mary_had_lamb")

AUDIO_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "ibm-granite/granite-speech-3.3-2b": {
        "prompt": (
            "<|start_of_role|>system<|end_of_role|>"
            "You are a helpful AI assistant<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|>"
            "<|audio|>can you transcribe the speech into a written format?"
            "<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        ),
        "audio_lora_path": "ibm-granite/granite-speech-3.3-2b",
    },
    "nvidia/audio-flamingo-3-hf": {
        "prompt": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<sound>Transcribe the input speech.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "sonar_runner_kwargs": {
            "gpu_memory_utilization": 0.85,
        },
    },
    "microsoft/VibeVoice-ASR-HF": {
        "prompt": (
            "<|im_start|>system\n"
            "You are a helpful assistant that transcribes audio input "
            "into text output in JSON format.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|object_ref_start|><|box_start|><|object_ref_end|>\n"
            "This is a 1.0 seconds audio, please transcribe it with "
            "these keys: Start time, End time, Speaker ID, Content"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "sampling_rate": 24000,
        "sonar_runner_kwargs": {
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.85,
        },
    },
    "zai-org/GLM-ASR-Nano-2512": {
        "prompt": (
            "<|user|>\n"
            "<|begin_of_audio|><|pad|><|end_of_audio|><|user|>\n"
            "Please transcribe this audio into text"
            "<|assistant|>\n"
        ),
    },
}


@pytest.fixture(autouse=True)
def use_spawn_for_audio_models(monkeypatch):
    """Single-process workaround for V1 fork safety deadlock issue
    (https://github.com/vllm-project/vllm/issues/17676). Running multiple audio models together
    under pytest can cause (possibly flaky) hangs, so they are grouped under
    the same config. Using SONAR_WORKER_MULTIPROC_METHOD=spawn avoids the
    deadlock and allows worker processes to terminate cleanly, and release
    GPU memory between test runs until the issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("SONAR_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("model_id", list(AUDIO_MODEL_SETTINGS))
def test_transformers_audio_generation(
    hf_runner: type[HfRunner],
    sonar_runner: type[SonarRunner],
    model_id: str,
):
    settings = AUDIO_MODEL_SETTINGS[model_id]
    audio_lora_path = settings.get("audio_lora_path")

    audio, orig_sr = AUDIO_ASSET.audio_and_sample_rate
    target_sr = settings.get("sampling_rate", orig_sr)
    if orig_sr != target_sr:
        audio = AudioResampler(target_sr=target_sr).resample(audio, orig_sr=orig_sr)
    audio = (audio, target_sr)

    with sonar_runner(
        model_id,
        model_impl="transformers",
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
        enable_lora=audio_lora_path is not None,
        max_lora_rank=64,
        **settings.get("sonar_runner_kwargs", {}),
    ) as sonar_model:
        lora_request = LoRARequest("audio", 1, audio_lora_path) if audio_lora_path else None
        sonar_outputs = sonar_model.generate_greedy_logprobs(
            [settings["prompt"]],
            128,
            num_logprobs=10,
            audios=[audio],
            lora_request=lora_request,
        )

    with hf_runner(model_id, dtype="bfloat16", auto_cls=AutoModelForSeq2SeqLM) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit([settings["prompt"]], 128, num_logprobs=10, audios=[audio])

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=sonar_outputs,
        name_0="hf",
        name_1="sonar",
    )
