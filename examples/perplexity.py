import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from aphrodite import LLM, SamplingParams

# Load the wikitext2 dataset.
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Get the first 2000 elements from the 'train' split.
prompts = dataset['train']['text'][:2000]

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# Create a tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokenize the prompts and discard or truncate any prompts longer than 2048 tokens.
tokenized_prompts = [tokenizer.encode(prompt, truncation=True,
                                      max_length=4096) for prompt in prompts]

# Detokenize the prompts.
detokenized_prompts = [tokenizer.decode(tokens
                                        ) for tokens in tokenized_prompts]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.0,
    ignore_eos=True,
    max_tokens=10,
    skip_special_tokens=False,
    spaces_between_special_tokens=False,
    logprobs=1,
    prompt_logprobs=1,
)

# Create an LLM.
llm = LLM(model=model_id)

# Generate texts from the detokenized prompts.
outputs = llm.generate(detokenized_prompts, sampling_params)

# Calculate the perplexity.
all_logprobs = []
for output in outputs:
    all_logprobs.extend([next(iter(lp.values())) for lp in output.prompt_logprobs[1:]])

all_logprobs = np.array([lp.logprob for lp in all_logprobs])
# NOTE: we need to divide by 2 to match the perplexity results
# for the same model on llama.cpp. I'm unsure if this
# approach to ppx measurement is correct.
perplexity = (np.exp(-all_logprobs.mean())) / 2
print(f"Perplexity: {perplexity}")
