# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Sonar project

import importlib.util

import pytest

from sonar import SamplingParams, TokensPrompt
from sonar.config import KVTransferConfig
from sonar.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_full_prompt_hit_restores_kv_before_graph_replay(sonar_runner, monkeypatch):
    if importlib.util.find_spec("lmcache") is None:
        pytest.skip("LMCache is not installed")

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("LMCACHE_CONFIG_FILE", raising=False)
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "256")
    monkeypatch.setenv("LMCACHE_LOCAL_CPU", "True")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "0.5")
    monkeypatch.setenv("LMCACHE_USE_LAYERWISE", "False")

    with sonar_runner(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.2,
        kv_cache_memory_bytes=256 << 20,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2],
        },
        kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both"),
    ) as runner:
        llm = runner.get_llm()
        prompt = TokensPrompt(prompt_token_ids=[1000 + i % 64 for i in range(768)])
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        cold = llm.generate(prompt, params, use_tqdm=False)[0]
        cached = llm.generate(prompt, params, use_tqdm=False)[0]

        assert cold.num_cached_tokens == 0
        # Recomputing just the final prompt token selects the full decode graph.
        assert cached.num_cached_tokens == 767
        assert cached.outputs[0].token_ids == cold.outputs[0].token_ids
