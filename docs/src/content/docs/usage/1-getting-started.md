---
title: Getting Started
---

Aphrodite can be used for several purposes, and can be run in several ways. This guide will show you how to:

- Launch an OpenAI-compatible API server,
- run batched inference on a dataset,
- and build an API server for an LLM yourself.

Be sure to read the installation instructions for your device before continuing.

## OpenAI API server

Aphrodite has implemented the OpenAI API protocol, and has almost perfect feature parity with it. For this reason, it can be used as a drop-in replacement for almost any application that uses OpenAI API. Aphrodite launches the server at `http://localhost:2242` by default, making the base URL `http://localhost:2242/v1`.

The server currently runs one model at a time, and implements the following endpoints:

- `/v1/models`: To show a list of the models available. This can include the primary LLM, and adapters (e.g. LoRA).
- `/v1/completions`: Provides a POST endpoint to send text completions requests to. The model field in the body is mandatory.
- `/v1/completions/render`: Renders text completion requests into token IDs for token-in/token-out serving.
- `/v1/completions/derender`: Derenders generated token IDs into text completion responses.
- `/v1/chat/completions`: Provides a POST endpoint to send chat completions requests to. The model field in the body is mandatory.
- `/v1/chat/completions/render`: Renders chat completion requests into token IDs for token-in/token-out serving.
- `/v1/chat/completions/derender`: Derenders generated token IDs into chat completion responses.

There are two ways to start the server; using a YAML config file, or the CLI. In this guide, we assume you want to run the [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) model on 2 GPUs.

### CLI

Start the server:

```sh
export HUGGINGFACE_HUB_TOKEN=<your hf token>  # only if using private or gated repos
aphrodite run meta-llama/Meta-Llama-3.1-8B-Instruct -tp 2
```

To see the full list of supported arguments, run `aphrodite run -h`.

By default, the server will use the chat template (for `/v1/chat/completions`) stored in the model's tokenizer. You can override this by adding the `--chat-template` argument:

```sh
aphrodite run meta-llama/Meta-Llama-3.1-8B-Instruct -tp 2 --chat-template ./examples/chat_templates/chatml.jinja
```

You may also provide direct download URLs to the argument.

You can launch the server with authentication enabled via API keys by either exporting `APHRODITE_API_KEY` environment variable, or passing your key to the `--api-keys` argument.
When API keys are configured, the render and derender endpoints require the same bearer token authentication as the `/v1/completions` and `/v1/chat/completions` endpoints.

### YAML Config

Aphrodite allows its users to define a YAML config for easier repeated launches of the engine. We provide an example [here](https://github.com/PygmalionAI/aphrodite-engine/tree/main/config.yaml). You can use this to get stated, by filling out the fields with your required parameters. Here's how launching Llama-3.1-8B-Instruct would look:

```yaml
basic_args:
  # Your model name. Can be a local path or huggingface model ID
  - model:

  # The tensor parallelism degree. Set this to the number of GPUs you have
  # Keep in mind that for **quantized** models, this will typically only work
  # with values between 1, 2, 4, and 8.
  - tensor_parallel_size: 2
```

You can save this to a `config.yaml` file, then launch Aphrodite like this:

```sh
aphrodite yaml config.yaml
```

As per the notice the sample config, Tensor Parallelism only works with odd-numbered GPUs for non-quantized models. For quantized models, it's recommended to use `pipeline_parallel_size` if you need to launch the model on 3, 5, 6, or 7 GPUs.

### Example Usage

Query the `/v1/models` endpoints like this:

```sh
curl http://localhost:2242/v1/models | jq .
```

Send a prompt and request completion tokens like this:

```sh
curl http://localhost:2242/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 128,
    "temperature": 1.1,
    "min_p": 0.1
  }' | jq .
```

:::warning
These curl commands assume you have `jq` installed to prettify the output JSON in the terminal. If you don't wish to install it, or don't have it installed already, please remove the `| jq .` at the end of each command.
:::

You can also use the endpoints via the `openai` python library:

```py
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:2242/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    prompt="Once upon a time",
    temperature=1.1,
    extra_body={"min_p": 0.1}
)

print("Completion result:", completion)
```

## Offline Batched Inference

You can use Aphrodite to process large datasets, or generate large amounts of data using a list of inputs.

To get started, first import the `LLM` and `SamplingParams` modules from Aphrodite. `LLM` is used to create a model object, and `SamplingParams` is used to define the sampling parameters to use for the requests.

```py
from aphrodite import LLM, SamplingParams
```

Then you can define your prompt list and sampling params. For the sake of simplicity, we won't load a dataset but rather define a few hardcoded prompts here:

```py
prompts = [
    "Once upon a time",
    "A robot may hurt a human if",
    "To get started with HF transformers,"
]
sampling_params = SamplingParams(temperature=1.1, min_p=0.1)
```

Now, initialize the engine using the `LLM` class with your model of choice. We will use Meta-Llama-3.1-8B-Instruct on 2 GPUs for this example.

```py
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=2)
```

The `LLM` class has a `generate()` method that we can use now. It adds the input prompts to the engine's waiting queue and executes them to generate outputs with high throughput, in parallel. The outputs are returned as a list of `RequestOutput` objects, which include all the output tokens.

```py
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r},\nGenerated text: {generated_text!r}")
```

Please view the [examples/](https://github.com/PygmalionAI/aphrodite-engine/tree/main/examples) directory for a full list of examples, for various different use-cases.
