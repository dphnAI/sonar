# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from sonar import SamplingParams
from sonar.config import CompilationConfig
from sonar.platforms import current_platform
from tests.utils import single_gpu_only

from ..utils import (
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)


@pytest.fixture
def disable_sonar_compile_cache_on_rocm(request: pytest.FixtureRequest) -> None:
    if current_platform.is_rocm():
        request.getfixturevalue("disable_sonar_compile_cache")


@pytest.mark.parametrize(
    "speculative_config",
    [
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
        {"method": "suffix"},
    ],
)
@pytest.mark.usefixtures("disable_sonar_compile_cache_on_rocm")
@single_gpu_only
def test_ngram_and_suffix_correctness(
    speculative_config: dict,
    model_name: str,
    sonar_runner,
):
    with sonar_runner(
        model_name,
        block_size=None,
        # Keep LLM defaults; SonarRunner only provides lifecycle cleanup here.
        trust_remote_code=False,
        enable_chunked_prefill=None,
        speculative_config=speculative_config,
        max_model_len=4096,
        # Preserve LLM's default compilation/cudagraph configuration. Without
        # this, SonarRunner injects its reduced test-only capture sizes.
        compilation_config=CompilationConfig(),
    ) as runner:
        evaluate_llm_for_gsm8k(runner.llm)


@pytest.mark.parametrize("async_scheduling", [True], ids=["async"])
@single_gpu_only
def test_ngram_gpu_default_with_async_scheduling(
    async_scheduling: bool,
    sonar_runner,
):
    """Test ngram_gpu speculative decoding (k=3) correctness with and without
    async scheduling, validated via GSM8K accuracy.
    Uses Qwen/Qwen3-8B (ref GSM8K accuracy: 87%-92%).
    """
    qwen3_model = "Qwen/Qwen3-8B"
    with sonar_runner(
        qwen3_model,
        block_size=None,
        trust_remote_code=False,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 2,
        },
        max_model_len=4096,
        async_scheduling=async_scheduling,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        # Assert the resolved async_scheduling config matches what was requested.
        assert spec_runner.llm.llm_engine.sonar_config.scheduler_config.async_scheduling == async_scheduling
        evaluate_llm_for_gsm8k(spec_runner.llm, expected_accuracy_threshold=0.8)


@single_gpu_only
def test_suffix_cross_request_acceptance(
    sampling_config: SamplingParams,
    model_name: str,
    sonar_runner,
):
    """Repeating the same prompts should let suffix decoding draft the earlier
    responses from its corpus of finished requests."""
    test_prompts = get_test_prompts(mm_enabled=False)
    k = 8

    with sonar_runner(
        model_name,
        block_size=None,
        trust_remote_code=False,
        speculative_config={"method": "suffix", "num_speculative_tokens": k},
        max_model_len=1024,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        num_drafts = []
        num_accept = []
        for _ in range(2):
            spec_runner.llm.chat(test_prompts, sampling_config)
            metrics = spec_runner.llm.get_metrics()
            num_drafts.append(get_spec_decode_metric_value(metrics, "sonar:spec_decode_num_drafts"))
            num_accept.append(get_spec_decode_metric_value(metrics, "sonar:spec_decode_num_accepted_tokens"))

    # Stats are cumulative; compare per-draft acceptance of the two runs.
    first = num_accept[0] / num_drafts[0]
    second = (num_accept[1] - num_accept[0]) / (num_drafts[1] - num_drafts[0])
    summary = f"accepted per draft: first={first:.2f}, second={second:.2f}"
    # Without the corpus, the repeated pass stays close to the first.
    assert second > 2 * first, summary
