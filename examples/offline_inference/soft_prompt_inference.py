# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from sonar.prompt_adapter.request import PromptAdapterRequest

from sonar import LLM, SamplingParams

# Define the model and prompt adapter paths
MODEL_PATH = "bigscience/bloomz-560m"
PA_PATH = "stevhliu/bloomz-560m_PROMPT_TUNING_CAUSAL_LM"


def do_sample(llm, pa_name: str, pa_id: int):
    # Sample prompts
    prompts = [
        "Tweet text : @nationalgridus I have no water and the bill is \
        current and paid. Can you do something about this? Label : ",
        "Tweet text : @nationalgridus Looks good thanks! Label : ",
    ]
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3, stop_token_ids=[3])

    # Generate outputs using the LLM
    outputs = llm.generate(
        prompts,
        sampling_params,
        prompt_adapter_request=PromptAdapterRequest(pa_name, pa_id, PA_PATH, 8) if pa_id else None,
    )

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    # Create an LLM with prompt adapter enabled
    llm = LLM(MODEL_PATH, enforce_eager=True, enable_prompt_adapter=True, max_prompt_adapter_token=8)

    # Run the sampling function
    do_sample(llm, "twitter_pa", pa_id=1)


if __name__ == "__main__":
    main()
