---
title: Reasoning and tool parsers
description: Configure model-specific reasoning and tool-call output parsing.
---

Reasoning and tool parsers convert model text into structured API response
fields. The parser must match the format produced by the model checkpoint.

## Configure reasoning

```bash
sonar serve MODEL --reasoning-parser glm45
```

The parser separates reasoning content from the final answer. Parser names are
model-format names, not general quality settings.

List the registered values:

```bash
sonar serve --help=reasoning-parser
```

Test a known prompt and inspect both response fields before production use.
Model template changes can alter delimiters that the parser expects.

## Configure automatic tool choice

```bash
sonar serve MODEL \
  --enable-auto-tool-choice \
  --tool-call-parser glm47
```

Clients can now send `tools` and use `tool_choice="auto"`. The model decides
whether to call a tool, and the parser converts its output to the OpenAI tool
call structure.

List current parser names:

```bash
sonar serve --help=tool-call-parser
```

## Validate a model pair

Use the parser recommended by the checkpoint publisher. Check the chat template
and parser together because the template controls the tool instructions.

Run these cases:

- A request that should call one tool
- A request that should answer without a tool
- More than one tool in the request
- Invalid tool arguments from the model
- Streaming output

Applications must validate tool names and arguments. A parsed call is model
output and remains untrusted.

## Custom parsers

`--reasoning-parser-plugin` can load a custom reasoning parser. Plugin code runs
inside the server process. Pin and review it as you would remote model code.

## Troubleshoot parsing

Disable sampling with temperature zero. Capture the raw model output and compare
its delimiters with the parser format.

If reasoning text appears in the final answer, check the reasoning parser. If a
tool call remains plain text, check the tool parser and automatic tool-choice
flag.
