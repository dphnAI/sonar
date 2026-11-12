# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of remote KV cache sharing
with LMCache.
We will launch 2 sonar instances, and launch an additional LMCache server.
KV cache is transferred in the following manner:
(1) Sonar instance 1 -> LMCache server (KV cache store).
(2) LMCache server -> Sonar instance 2 (KV cache reuse/retrieve).

Note that lmcache needs to be installed to run this example.
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


def run_store(store_done, prompts):
    # We use GPU 0 for KV cache store process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
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

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("KV cache store is finished.")
    store_done.set()

    llm.llm_engine.engine_core.shutdown()


def run_retrieve(store_done, prompts, timeout=1):
    # We use GPU 1 for KV cache retrieve process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
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

    print("Waiting for KV cache store to finish...")
    store_done.wait()
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
    store_done = Event()
    store_process = Process(target=run_store, args=(store_done, prompts))
    retrieve_process = Process(target=run_retrieve, args=(store_done, prompts))
    lmcache_server_process = run_lmcache_server(port)

    # Start KV cache store process
    store_process.start()

    # Start KV cache retrieve process
    retrieve_process.start()

    # Clean up the processes
    store_process.join()
    retrieve_process.terminate()
    lmcache_server_process.terminate()
    lmcache_server_process.wait()


if __name__ == "__main__":
    main()
