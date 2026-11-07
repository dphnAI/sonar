# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core test implementation to be shared across modalities."""

from collections.abc import Callable
from typing import Any

import torch
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from sonar.config.model import RunnerOption
from sonar.tokenizers import TokenizerLike

from .....conftest import HfRunner, SonarRunner
from ....registry import HF_EXAMPLE_MODELS
from .types import PromptWithMultiModalInput, RunnerOutput


def run_test(
    *,
    hf_runner: type[HfRunner],
    sonar_runner: type[SonarRunner],
    inputs: list[PromptWithMultiModalInput],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    enforce_eager: bool,
    max_model_len: int,
    max_num_seqs: int,
    hf_output_post_proc: Callable[[RunnerOutput, str], Any] | None,
    sonar_output_post_proc: Callable[[RunnerOutput, str], Any] | None,
    auto_cls: type[_BaseAutoModelClass],
    use_tokenizer_eos: bool,
    comparator: Callable[..., None],
    get_stop_token_ids: Callable[[TokenizerLike], list[int]] | None,
    stop_str: list[str] | None,
    limit_mm_per_prompt: dict[str, int],
    sonar_runner_kwargs: dict[str, Any] | None,
    hf_model_kwargs: dict[str, Any] | None,
    hf_processor: Callable[[str], Any] | None,
    patch_hf_runner: Callable[[HfRunner], HfRunner] | None,
    runner: RunnerOption = "auto",
    distributed_executor_backend: str | None = None,
    tensor_parallel_size: int = 1,
    sonar_embeddings: torch.Tensor | None = None,
):
    """Modality agnostic test executor for comparing HF/Sonar outputs."""
    # In the case of embeddings, Sonar takes separate input tensors
    sonar_inputs = sonar_embeddings if sonar_embeddings is not None else inputs

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    limit_mm_per_prompt = default_limits | limit_mm_per_prompt

    sonar_outputs_per_mm = []
    hf_outputs_per_mm = []

    # NOTE: take care of the order. run Sonar first, and then run HF.
    # Sonar needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    sonar_runner_kwargs_: dict[str, Any] = {"mm_processor_cache_gb": 0}
    if model_info.tokenizer:
        sonar_runner_kwargs_["tokenizer_name"] = model_info.tokenizer
    if model_info.tokenizer_mode:
        sonar_runner_kwargs_["tokenizer_mode"] = model_info.tokenizer_mode
    if model_info.hf_overrides:
        sonar_runner_kwargs_["hf_overrides"] = model_info.hf_overrides
    if model_info.require_embed_inputs:
        for k in ("skip_tokenizer_init", "enable_prompt_embeds", "enable_mm_embeds"):
            sonar_runner_kwargs_[k] = model_info.require_embed_inputs
    if not model_info.enable_prefix_caching:
        sonar_runner_kwargs_["enable_prefix_caching"] = False

    if sonar_runner_kwargs:
        sonar_runner_kwargs_.update(sonar_runner_kwargs)

    # Avoid passing limit_mm_per_prompt twice when sonar_runner_kwargs
    # already contains it (e.g. gemma4 sets it via sonar_runner_kwargs).
    if "limit_mm_per_prompt" in sonar_runner_kwargs_:
        limit_mm_per_prompt = sonar_runner_kwargs_.pop("limit_mm_per_prompt")

    with sonar_runner(
        model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        limit_mm_per_prompt=limit_mm_per_prompt,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
        runner=runner,
        **sonar_runner_kwargs_,
    ) as sonar_model:
        tokenizer = sonar_model.llm.get_tokenizer()

        sonar_kwargs: dict[str, Any] = {}
        if get_stop_token_ids is not None:
            sonar_kwargs["stop_token_ids"] = get_stop_token_ids(tokenizer)
        if stop_str:
            sonar_kwargs["stop"] = stop_str

        for prompts, image_data, video_data, audio_data in sonar_inputs:
            mm_data = dict(images=image_data, videos=video_data, audios=audio_data)
            sonar_kwargs_with_mm_data = sonar_kwargs | mm_data
            sonar_output = sonar_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                **sonar_kwargs_with_mm_data,
            )
            sonar_outputs_per_mm.append(sonar_output)

    hf_runner_kwargs: dict[str, Any] = {}
    if model_info.tokenizer:
        hf_runner_kwargs["tokenizer_name"] = model_info.tokenizer
    if hf_processor is not None:
        hf_runner_kwargs["processor"] = hf_processor(model)

    hf_model = hf_runner(
        model,
        dtype=dtype,
        auto_cls=auto_cls,
        model_kwargs=hf_model_kwargs,
        **hf_runner_kwargs,
    )

    # Some models need to patch things like the model processor, e.g., internvl
    if patch_hf_runner is not None:
        hf_model = patch_hf_runner(hf_model)

    with hf_model, torch.no_grad():
        tokenizer = hf_model.tokenizer

        # Some models need to explicitly pass the eos_token_id off the tokenizer
        # or processor for a good comparison;
        # currently assume processor/tokenizer agree on the EOS, and pull it off
        # the tokenizer if requested.
        hf_kwargs = {}
        if use_tokenizer_eos:
            hf_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if stop_str:
            hf_kwargs["stop_strings"] = stop_str

        for prompts, image_data, video_data, audio_data in inputs:
            mm_data = dict(images=image_data, videos=video_data, audios=audio_data)
            hf_kwargs_with_mm_data = hf_kwargs | mm_data
            hf_output = hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                tokenizer=tokenizer,
                **hf_kwargs_with_mm_data,
            )
            hf_outputs_per_mm.append(hf_output)

    # Apply output processing / sanitation to the Sonar and HF runner results
    hf_outputs_per_mm, sonar_outputs_per_mm = process_runner_outputs(
        model,
        first_runner_outputs=hf_outputs_per_mm,
        second_runner_outputs=sonar_outputs_per_mm,
        first_runner_processor=hf_output_post_proc,
        second_runner_processor=sonar_output_post_proc,
    )

    for hf_outputs, sonar_outputs in zip(hf_outputs_per_mm, sonar_outputs_per_mm):
        # This is usually check_logprobs_close, but it's passed through to
        # allow things like check_outputs_equal where needed
        comparator(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=sonar_outputs,
            name_0="hf",
            name_1="sonar",
        )


def process_runner_outputs(
    model,
    first_runner_outputs,
    second_runner_outputs,
    first_runner_processor=None,
    second_runner_processor=None,
):
    """Applies the runner processor(s) to the runner outputs, if any."""
    if first_runner_processor is not None:
        first_runner_outputs = process_outputs(first_runner_processor, model, first_runner_outputs)
    if second_runner_processor is not None:
        second_runner_outputs = process_outputs(second_runner_processor, model, second_runner_outputs)
    return first_runner_outputs, second_runner_outputs


def process_outputs(output_processor, model, outputs_per_image):
    """Applies a model specific post-processor function to a runner's output."""
    return [[output_processor(res, model) for res in outputs] for outputs in outputs_per_image]
