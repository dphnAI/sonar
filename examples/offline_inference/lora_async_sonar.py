# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use the multi-LoRA functionality for offline
inference. Requires HuggingFace credentials for access to Llama2.
"""

import asyncio

from sonar import AsyncEngineArgs, AsyncSonar, SamplingParams
from sonar.lora.request import LoRARequest


def create_test_prompts(lora_path: str) -> list[tuple[str, SamplingParams, LoRARequest | None]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(
                temperature=0.0,
                # logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            None,
        ),
        ("To be or not to be,", SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128), None),
        (
            """[user] Write a SQL query to answer the question based on the
            table schema.\n\n context: CREATE TABLE table_name_74
            (icao VARCHAR, airport VARCHAR)\n\n
            question: Name the ICAO for lilongwe
            international airport [/user] [assistant]""",
            SamplingParams(
                temperature=0.0,
                # logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
                stop_token_ids=[32003],
            ),
            LoRARequest(lora_name="l2-lora-test", lora_int_id=1, lora_path=lora_path),
        ),
        (
            """[user] Write a SQL query to answer the question based on the table
         schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR,
         elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector
         what is under nationality? [/user] [assistant]""",
            SamplingParams(n=3, best_of=3, temperature=0.8, max_tokens=128, stop_token_ids=[32003]),
            LoRARequest(lora_name="l2-lora-test", lora_int_id=1, lora_path=lora_path),
        ),
        (
            """[user] Write a SQL query to answer the question based on the
            table schema.\n\n context: CREATE TABLE table_name_74 (icao
            VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe
            international airport [/user] [assistant]""",
            SamplingParams(
                temperature=0.0,
                # logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
                stop_token_ids=[32003],
            ),
            LoRARequest(lora_name="l2-lora-test2", lora_int_id=2, lora_path=lora_path),
        ),
        (
            """[user] Write a SQL query to answer the question based on the table
         schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR,
         elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector
         what is under nationality? [/user] [assistant]""",
            SamplingParams(n=3, best_of=3, temperature=0.9, max_tokens=128, stop_token_ids=[32003]),
            LoRARequest(lora_name="l2-lora-test", lora_int_id=1, lora_path=lora_path),
        ),
    ]  # type: ignore


async def process_requests(engine: AsyncSonar, test_prompts: list[tuple[str, SamplingParams, LoRARequest | None]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    active_requests = []

    for prompt, sampling_params, lora_request in test_prompts:
        request_generator = engine.generate(prompt, sampling_params, str(request_id), lora_request=lora_request)
        active_requests.append(request_generator)
        # No increment needed as enumerate() provides the request_id

    # Process all requests
    for request_generator in active_requests:
        # Don't await the generator itself, just iterate over it
        async for request_output in request_generator:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> AsyncSonar:
    """Initialize the AsyncSonar."""
    # Function remains unchanged as it's just initialization
    engine_args = AsyncEngineArgs(
        model="NousResearch/Llama-2-7b-hf",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=256,
    )
    return AsyncSonar.from_engine_args(engine_args)


async def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    test_prompts = create_test_prompts("alpindale/l2-lora-test")
    await process_requests(engine, test_prompts)


if __name__ == "__main__":
    asyncio.run(main())
