# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the short outputs of HF and Sonar when using greedy sampling.

Run `pytest tests/basic_correctness/models/test_basic_correctness.py`.
"""

import os
import weakref
from unittest.mock import Mock

import pytest
import torch

import sonar.envs as envs
from sonar import LLM
from sonar.platforms import current_platform
from sonar.v1.engine.llm_engine import LLMEngine

from ...conftest import HfRunner, SonarRunner
from ...models.utils import check_outputs_equal
from ...utils import multi_gpu_test

ATTN_BACKEND = ["ROCM_ATTN"] if current_platform.is_rocm() else ["FLASH_ATTN"]

MODELS = [
    "hmellor/tiny-random-Gemma2ForCausalLM",
    "meta-llama/Llama-3.2-1B-Instruct",
]

TARGET_TEST_SUITE_ENV = "SONAR_TARGET_TEST_SUITE"
LEGACY_TARGET_TEST_SUITE_ENV = "TARGET_TEST_SUITE"

GENERIC_DISTRIBUTED_TEST_SUITES = ("L4", "MI250", "MI300", "MI325", "MI355")
ALL_DISTRIBUTED_TEST_SUITES = (*GENERIC_DISTRIBUTED_TEST_SUITES, "A100")


def _default_target_test_suite() -> str:
    if not current_platform.is_rocm():
        return "L4"

    try:
        device_name = current_platform.get_device_name().upper()
    except Exception:
        device_name = ""

    if "MI355" in device_name:
        return "MI355"
    if "MI300" in device_name:
        return "MI300"
    if "MI325" in device_name:
        return "MI325"
    if "MI250" in device_name:
        return "MI250"

    try:
        from sonar.platforms import rocm as rocm_platform

        if rocm_platform.on_gfx950():
            return "MI355"
        if rocm_platform.on_gfx942():
            return "MI300"
    except Exception:
        pass

    return "MI250"


def _resolve_target_test_suite() -> str:
    for env_name in (TARGET_TEST_SUITE_ENV, LEGACY_TARGET_TEST_SUITE_ENV):
        value = os.environ.get(env_name, "").strip().upper()
        if value:
            return value
    return _default_target_test_suite()


TARGET_TEST_SUITE = _resolve_target_test_suite()


# ROCm can occasionally retain the object until fixture teardown. Retry only
# that assertion after cleanup; collecting cyclic garbage here would mask the
# reference cycles this test is intended to catch.
@pytest.mark.flaky(
    reruns=2,
    reruns_delay=5,
    only_rerun="AssertionError",
    condition=current_platform.is_rocm(),
)
def test_sonar_gc_ed():
    """Verify sonar instance is GC'ed when it is deleted."""
    llm = LLM("hmellor/tiny-random-LlamaForCausalLM")
    weak_llm = weakref.ref(llm)
    del llm
    # If there's any circular reference to sonar, this fails
    # because llm instance is not GC'ed.
    assert weak_llm() is None


def _fix_prompt_embed_outputs(
    sonar_outputs: list[tuple[list[int], str]],
    hf_model: HfRunner,
    example_prompts: list[str],
) -> list[tuple[list[int], str]]:
    fixed_sonar_outputs = []
    for sonar_output, hf_input, prompt in zip(sonar_outputs, hf_model.get_inputs(example_prompts), example_prompts):
        hf_input_ids = hf_input["input_ids"].tolist()[0]
        fixed_sonar_outputs.append(
            (
                hf_input_ids + sonar_output[0][len(hf_input_ids) :],
                prompt + sonar_output[1],
            )
        )
    return fixed_sonar_outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ATTN_BACKEND)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("model_executor", ["uni", "mp"])
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    model: str,
    backend: str,
    max_tokens: int,
    enforce_eager: bool,
    async_scheduling: bool,
    model_executor: str,
    enable_prompt_embeds: bool,
) -> None:
    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
        if enable_prompt_embeds:
            with torch.no_grad():
                prompt_embeds = hf_model.get_prompt_embeddings(example_prompts)

    with SonarRunner(
        model,
        max_model_len=8192,
        enforce_eager=enforce_eager,
        enable_prompt_embeds=enable_prompt_embeds,
        gpu_memory_utilization=0.7,
        async_scheduling=async_scheduling,
        distributed_executor_backend=model_executor,
        attention_config={"backend": backend},
    ) as sonar_model:
        if enable_prompt_embeds:
            sonar_outputs = sonar_model.generate_greedy(prompt_embeds, max_tokens)
            sonar_outputs = _fix_prompt_embed_outputs(sonar_outputs, hf_model, example_prompts)
        else:
            sonar_outputs = sonar_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=sonar_outputs,
        name_0="hf",
        name_1="sonar",
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    ("model, distributed_executor_backend, attention_backend, target_test_suites, extra_env"),
    [
        ("facebook/opt-125m", "ray", "", ALL_DISTRIBUTED_TEST_SUITES, {}),
        ("facebook/opt-125m", "mp", "", ALL_DISTRIBUTED_TEST_SUITES, {}),
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "ray",
            "",
            GENERIC_DISTRIBUTED_TEST_SUITES,
            {},
        ),
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "mp",
            "",
            GENERIC_DISTRIBUTED_TEST_SUITES,
            {},
        ),
    ],
)
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_models_distributed(
    monkeypatch: pytest.MonkeyPatch,
    hf_runner,
    sonar_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
    target_test_suites: tuple[str, ...],
    extra_env: dict[str, str],
    enable_prompt_embeds: bool,
) -> None:
    if TARGET_TEST_SUITE and TARGET_TEST_SUITE not in target_test_suites:
        pytest.skip(f"Skip test for {TARGET_TEST_SUITE}")

    with monkeypatch.context() as monkeypatch_context:
        for k, v in extra_env.items():
            monkeypatch_context.setenv(k, v)

        dtype = "half"
        max_tokens = 5

        # NOTE: take care of the order. run Sonar first, and then run HF.
        # Sonar needs a fresh new process without cuda initialization.
        # if we run HF first, the cuda initialization will be done and it
        # will hurt multiprocessing backend with fork method
        # (the default method).
        attention_config = {"backend": attention_backend} if attention_backend else None
        with sonar_runner(
            model,
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend=distributed_executor_backend,
            enable_prompt_embeds=enable_prompt_embeds,
            gpu_memory_utilization=0.7,
            attention_config=attention_config,
        ) as sonar_model:
            if enable_prompt_embeds:
                with hf_runner(model, dtype=dtype) as hf_model:
                    with torch.no_grad():
                        prompt_embeds = hf_model.get_prompt_embeddings(example_prompts)
                    sonar_outputs = sonar_model.generate_greedy(prompt_embeds, max_tokens)
                    sonar_outputs = _fix_prompt_embed_outputs(sonar_outputs, hf_model, example_prompts)
                    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
            else:
                sonar_outputs = sonar_model.generate_greedy(example_prompts, max_tokens)
                with hf_runner(model, dtype=dtype) as hf_model:
                    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=sonar_outputs,
        name_0="hf",
        name_1="sonar",
    )


def test_failed_model_execution(sonar_runner, monkeypatch) -> None:
    # Needed to mock an error in the same process
    monkeypatch.setenv("SONAR_ENABLE_V1_MULTIPROCESSING", "0")

    with sonar_runner("facebook/opt-125m", enforce_eager=True) as sonar_model:
        if isinstance(sonar_model.llm.llm_engine, LLMEngine):
            v1_test_failed_model_execution(sonar_model)


@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_raise_on_logit_nans(sonar_runner, monkeypatch, use_v2_model_runner: bool) -> None:
    monkeypatch.setenv("SONAR_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("SONAR_RAISE_ON_LOGIT_NANS", "1")
    monkeypatch.setenv("SONAR_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    envs.disable_envs_cache()

    try:
        with sonar_runner("facebook/opt-125m", enforce_eager=True) as sonar_model:
            engine_core = sonar_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            assert model_runner.sonar_config.use_v2_model_runner is use_v2_model_runner
            model_cls = type(model_runner.model)
            original_compute_logits = model_cls.compute_logits

            def compute_nan_logits(self, *args, **kwargs):
                logits = original_compute_logits(self, *args, **kwargs)
                # `logits[0, 0] = ...` copies the scalar H2D and blocks.
                logits[0, 0].fill_(float("nan"))
                return logits

            monkeypatch.setattr(model_cls, "compute_logits", compute_nan_logits)

            with pytest.raises(RuntimeError, match="NaNs detected in logits"):
                sonar_model.generate_greedy(["Hello, my name is"], 1, use_tqdm=False)
    finally:
        envs.disable_envs_cache()


def v1_test_failed_model_execution(sonar_model):
    engine = sonar_model.llm.llm_engine
    mocked_execute_model = Mock(side_effect=RuntimeError("Mocked Critical Error"))
    engine.engine_core.engine_core.model_executor.execute_model = mocked_execute_model

    with pytest.raises(RuntimeError) as exc_info:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sonar_model.generate_greedy(prompts, 200, use_tqdm=False)
    assert isinstance(exc_info.value, RuntimeError)
    assert "Mocked Critical Error" in str(exc_info.value)
