# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai import OpenAI

# Modify OpenAI's API key and API base to use Sonar's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:2242/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "The weather is nice today",
    ],
    model=model,
)

for data in responses.data:
    print(data.embedding)  # list of float of len 4096
