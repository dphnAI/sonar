# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openai_harmony import HarmonyError

from aphrodite.entrypoints.chat_utils import make_tool_call_id
from aphrodite.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from aphrodite.entrypoints.openai.parser.harmony_utils import (
    extract_function_from_recipient,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)
from aphrodite.tool_parsers.abstract_tool_parser import Tool, ToolParser

if TYPE_CHECKING:
    from aphrodite.tokenizers import TokenizerLike


def _normalize_recipient(recipient: str | None) -> str | None:
    """Remove constrained formats misparsed into recipients by older Harmony."""
    if recipient is None:
        return None
    constrain_index = recipient.find("<|constrain|>")
    if constrain_index == -1:
        return recipient
    return recipient[:constrain_index].rstrip() or None


class GptOssToolParser(ToolParser):
    """
    Tool parser for gpt-oss/harmony models.

    Streaming is handled by HarmonyParser (registered as its
    tool_parser_cls capability declaration), which keeps a persistent
    StreamableParser across deltas. Non-streaming extraction is
    implemented here directly from the raw output token IDs: Harmony's
    channel/recipient framing lives in special tokens, so it can't be
    recovered by re-parsing the decoded text.
    """

    def __init__(self, tokenizer: "TokenizerLike", tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(
        self,
        model_output: str,
        token_ids: Sequence[int] | None,
        request,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise ValueError(
                "GptOssToolParser.extract_tool_calls requires token_ids: "
                "Harmony's channel/recipient framing lives in special "
                "tokens and cannot be recovered from decoded text alone."
            )

        parser = get_streamable_parser_for_assistant()
        try:
            for token_id in token_ids:
                parser.process(token_id)
            parser.process_eos()
        except HarmonyError:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for msg in parser.messages:
            if msg.author.role != "assistant" or not msg.content:
                continue
            text = msg.content[0].text
            recipient = _normalize_recipient(msg.recipient)
            if recipient and is_function_recipient(recipient):
                content_type = msg.content_type
                if content_type is not None and "json" not in content_type:
                    arguments = text
                else:
                    try:
                        arguments = json.dumps(json.loads(text))
                    except json.JSONDecodeError:
                        arguments = text
                tool_calls.append(
                    ToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=extract_function_from_recipient(recipient),
                            arguments=arguments,
                        ),
                    )
                )
            elif text and (msg.channel == "final" or (msg.channel == "commentary" and recipient is None)):
                content_parts.append(text)

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content="\n".join(content_parts) or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssToolParser does not support standalone streaming extraction. "
            "Use HarmonyParser, which keeps the persistent parser state that "
            "incremental Harmony decoding requires."
        )
