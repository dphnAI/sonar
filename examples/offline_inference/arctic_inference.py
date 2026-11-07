# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from sonar import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox jumps over the lazy dog.",
    "The meaning of life is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1.15, min_p=0.06)

# Create an LLM.
llm = LLM(
    model="snowflake/snowflake-arctic-instruct",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
    trust_remote_code=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
