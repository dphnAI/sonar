# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of disaggregated prefilling
with LMCache.
We will launch 2 sonar instances (GPU 0 for prefill and GPU 1 for decode),
and launch an additional LMCache server.
KV cache is transferred in the following manner:
Sonar prefill node -> LMCache server -> Sonar decode node.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""

import os
import subprocess
import sys
import time
from multiprocessing import Event, Process

from sonar import LLM, SamplingParams
from sonar.config import KVTransferConfig

# LMCache-related environment variables
# The port to start LMCache server
port = 8100

prompts = [
    "Hello, how are you?" * 1000,
]


def run_prefill(prefill_done, prompts):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"lmcache.mp.port": port},
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    # llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("Prefill node is finished.")
    prefill_done.set()

    llm.llm_engine.engine_core.shutdown()


def run_decode(prefill_done, prompts, timeout=1):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={"lmcache.mp.port": port},
    )
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # of memory. Reduce the value if your GPU has less memory.
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    print("Waiting for prefill node to finish...")
    prefill_done.wait()
    time.sleep(timeout)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    llm.llm_engine.engine_core.shutdown()


def run_lmcache_server(port):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            "localhost",
            "--port",
            str(port),
            "--l1-size-gb",
            "5",
            "--l1-init-size-gb",
            "5",
            "--eviction-policy",
            "LRU",
        ]
    )


def main():
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, prompts))
    decode_process = Process(target=run_decode, args=(prefill_done, prompts))
    lmcache_server_process = run_lmcache_server(port)

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Clean up the processes
    decode_process.join()
    prefill_process.terminate()
    lmcache_server_process.terminate()
    lmcache_server_process.wait()


if __name__ == "__main__":
    main()
