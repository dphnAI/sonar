# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

from aphrodite.parser.qwen3 import qwen3_config
from aphrodite.tokenizers import TokenizerLike
from aphrodite.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from aphrodite.tool_parsers.utils import Tool


class Step5ToolParser(Qwen3EngineToolParser):
    """Qwen Coder tool grammar with Step-5's content whitespace preserved."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None, **kwargs) -> None:
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=replace(
                qwen3_config(thinking=False, name="step5"),
                drop_whitespace_only_content_before_tools=False,
                strip_content_whitespace_with_tools=False,
            ),
            **kwargs,
        )
