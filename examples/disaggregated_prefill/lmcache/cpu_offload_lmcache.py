# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload KV cache to CPU with LMCache and reuse a shared prompt prefix.

Install dependencies with tools/install_kv_connectors.sh, then run with
PYTHONHASHSEED=0.
"""

import argparse
import contextlib
import os
import time

from sonar import LLM, SamplingParams
from sonar.config import KVTransferConfig


def setup_environment_variables():
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"


@contextlib.contextmanager
def build_llm_with_lmcache(model: str):
    llm = LLM(
        model=model,
        kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both"),
        max_model_len=8000,
        # Attribute reuse to LMCache rather than Sonar's local prefix cache.
        enable_prefix_caching=False,
    )
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()


def print_output(llm: LLM, prompt: list[str], sampling_params: SamplingParams, req_str: str):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    for output in outputs:
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Cached prompt tokens: {output.num_cached_tokens}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    setup_environment_variables()

    with build_llm_with_lmcache(args.model) as llm:
        shared_prompt = "Hello, how are you?" * 1000
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        print_output(llm, [shared_prompt + "Hello, my name is"], sampling_params, "first")
        print_output(llm, [shared_prompt + "Tell me a very long story"], sampling_params, "second")


if __name__ == "__main__":
    main()
