---
title: Using Vision Language Models
---

Aphrodite provides experimental support for Vision Language Models (VLMs). See the [list of supported VLMs here](/usage/models#multimodal-language-models). This document shows you how to run and serve these models using Aphrodite.

:::warning
We are actively working on improving the VLM support in Aphrodite. Expect breaking changes in the future without any deprecation warnings.

Currently, the support for VLMs has the following limitation:

- Only single image input is supported per text prompt.

We are continuously improving user & developer experience. If you have any feedback or feature requests, please [open an issue](https://github.com/PygmalionAI/aphrodite-engine/issues/new/choose).
:::

## Offline Batched Inference

To initialize a VLM, the aforementioned arguments must be passed to the `LLM` class for instantiating the engine.

```py
llm = LLM(model="llava-hf/llava-1.5-7b-hf")
```

To pass an image to the model, note the following in `aphrodite.inputs.PromptInputs`:

- `prompt`: The prompt should follow the format that is documented on Hugging Face.
- `multi_modal_data`: This is a dictionary that follows the schema defined in `aphrodite.multimodal.MultiModalDataDict`

```py
# Refer to the HuggingFace repo for the correct format to use
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# Load the image using PIL.Image
image = PIL.Image.open(...)

# Single prompt inference
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

# Batch inference
image_1 = PIL.Image.open(...)
image_2 = PIL.Image.open(...)
outputs = llm.generate(
    [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        }
    ]
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

## Online OpenAI Vision API Inference

You can serve vision language models with Aphrodite's OpenAI server.

:::tip
Currently, Aphrodite supports only *single* `image_url` input per `messages`. Support for multi-image inputs will be added in the future.
:::

Below is an example on how to launch the same llava-hf/llava-1.5-7b-hf with Aphrodite API server.

:::danger
Since OpenAI Vision API is based on [Chat](https://platform.openai.com/docs/api-reference/chat) API, a chat template is required to launch the API server if the model's tokenizer does not come with one. In this example, we use the HuggingFace Llava chat template that you can find in the example folder [here](https://github.com/PygmalionAI/aphrodite-engine/tree/main/examples/chat_templates/llava.jinja).
:::

```bash
aphrodite run llava-hf/llava-1.5-7b-hf --chat-template llava.jinja
```

To send a request to the server, you can use the following code:

```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:2242/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
chat_response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            # NOTE: The prompt formatting with the image token `<image>` is not needed
            # since the prompt will be processed automatically by the API server.
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            },
        ],
    }],
)
print("Chat response:", chat_response)
```

:::tip
By default, the timeout for fetching images through http url is `10` seconds. You can override this by setting this env variable:

```bash
export APHRODITE_IMAGE_FETCH_TIMEOUT=<timeout>
```

:::

### Video Decoding Backend

Aphrodite decodes video bytes into frames using a selectable decoding backend.
Three FFmpeg-backed software backends are supported:

- `opencv` (default): OpenCV-based decoder.
- `pyav`: PyAV decoder.
- `torchcodec`: TorchCodec (PyTorch-native) decoder.

`torchcodec` lets you choose which FFmpeg version is used, while `opencv` and
`pyav` rely on whichever FFmpeg build they were linked against.

Select the codec backend by passing `backend` via `--media-io-kwargs`:

```bash
aphrodite run Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec"}}'
```

TorchCodec-specific parameters:

- `num_ffmpeg_threads`: Number of FFmpeg decoding threads. `0` (default) uses
  the FFmpeg default.
- `seek_mode`: Seek mode for the decoder. `"exact"` (default) guarantees
  frame-accurate sampling. `"approximate"` skips the initial scan and relies on
  the file's metadata.

```bash
aphrodite run Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec", "seek_mode": "approximate", "num_ffmpeg_threads": 4}}'
```

:::tip
There is no need to format the prompt in the API request since it'll be handled by the server.
:::

Here's a curl example:

```bash
curl -X POST "http://localhost:2242/v1/chat/completions" -H "Content-Type: application/json" \
    -H "Authorization : Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "llava-hf/llava-1.5-7b-hf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ]
    }'
```
