---
title: Tool calling
description: Parse model output into OpenAI-compatible tool calls.
---

The model and parser must use the same tool-call format.

## Configure the server

Start the server with automatic tool selection and a parser.

```bash
sonar serve MODEL \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

`--enable-auto-tool-choice` lets the model decide whether to call a tool.
`--tool-call-parser` converts the model's text format into structured tool-call
fields. Select the parser documented for the model.

Test explicit tool choice and automatic tool choice. Some models require a
specific chat template in addition to the parser.

## Send tools

Send tools through the OpenAI client.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="MODEL",
    messages=[{"role": "user", "content": "What is the weather in Kabul?"}],
    tools=tools,
    tool_choice="auto",
)
```

Inspect the returned message before you execute anything:

```python
message = response.choices[0].message

if message.tool_calls:
    for call in message.tool_calls:
        print(call.function.name)
        print(call.function.arguments)
else:
    print(message.content)
```

The arguments field is model output. Parse it as JSON and validate it against
the tool schema. Apply authorization checks after validation. Do not execute
unknown functions or pass untrusted arguments to a shell.

## Return a tool result

Append the assistant message and one tool message for each call. Then send the
conversation back to the model:

```python
import json

call = message.tool_calls[0]
messages = [
    {"role": "user", "content": "What is the weather in Kabul?"},
    message,
    {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps({"temperature_c": 24, "condition": "clear"}),
    },
]

final = client.chat.completions.create(
    model="MODEL",
    messages=messages,
    tools=tools,
)
print(final.choices[0].message.content)
```

Preserve the tool-call ID. It links the result to the original request.

## Streaming

Streaming responses can split a tool name and its JSON arguments across
several events. Accumulate deltas by tool-call index. Execute the tool only
after the stream reports a completed tool call and the arguments pass
validation.

Test streaming separately. A parser can support non-streaming output before it
supports all streaming edge cases.

## Troubleshoot parser output

- If the response contains raw tool syntax in `content`, check the parser and
  chat template.
- If arguments are incomplete, increase the output-token limit and inspect the
  finish reason.
- If the model always calls a tool, test `tool_choice="none"` and confirm that
  the model supports tool choice.
- If a streaming call has invalid JSON, confirm that the client joined all
  argument deltas in order.

Read `sonar serve --help=tool-call-parser` for the current parser names.
Use the parser recommended by the model publisher when one is available.
See [Reasoning and tool parsers](/features/reasoning-and-tools/) for parser
selection and reasoning-output behavior.
