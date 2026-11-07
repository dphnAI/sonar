# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from sonar import EngineArgs, SamplingParams, SonarEngine


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = SonarEngine.from_engine_args(engine_args)

    # Test the following prompts.
    test_prompts = [
        ("<|system|>Enter chat mode.<|user|>Hello!<|model|>", SamplingParams(temperature=0.0)),
        (
            "<|system|>Enter RP mode.<|model|>Hello!<|user|>What are you doing?<|model|>",  # noqa: E501
            SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2),
        ),
        (
            "<|system|>Enter chat mode.<|user|>What is the meaning of life?<|model|>",  # noqa: E501
            SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, frequency_penalty=0.1),
        ),
        (
            "<|system|>Enter QA mode.<|user|>What is a man?<|model|>A miserable",
            SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0),
        ),
    ]

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    while True:
        # To test continuous batching, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on using the SonarEngine class directly")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
