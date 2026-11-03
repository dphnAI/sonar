# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An example showing how to use aphrodite to serve VLMs.
Launch the aphrodite server with the following command:
aphrodite run fixie-ai/ultravox-v0_3
"""

import os

import pybase64 as base64
from openai import OpenAI

# Get path to the audio file in ../audio directory
audio_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "audio",
    "mary_had_lamb.ogg",
)

# Modify OpenAI's API key and API base to use aphrodite's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:2242/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id


def encode_audio_base64_from_file(file_path: str) -> str:
    """Encode an audio file to base64 format."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Use base64 encoded audio in the payload
audio_base64 = encode_audio_base64_from_file(audio_path)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this audio?"},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/ogg;base64,{audio_base64}"},
                },
            ],
        }  # type: ignore
    ],
    model=model,
    max_tokens=128,
)
result = chat_completion.choices[0].message.content
print(f"Chat completion output: {result}")
