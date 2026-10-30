# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
from typing import Any

from aphrodite.renderers.chat_utils import ChatCompletionMessageParam, ConversationMessage
from aphrodite.tokenizers.deepseek_v41_encoding import merge_tool_messages, sort_tool_results_by_call_order

from .deepseek_v4 import DeepseekV4Renderer
from .inputs import DictPrompt
from .params import ChatParams


def _order_like_v41_prompt(messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """Match image parsing to the encoder's parallel tool-result order."""
    if not all(isinstance(message, dict) for message in messages):
        return messages
    markers: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        marker: dict[str, Any] = {"role": role, "content": index}
        if role == "tool":
            marker["tool_call_id"] = message.get("tool_call_id", "")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            marker["tool_calls"] = [
                {"id": call.get("id") or call.get("function", {}).get("id", "")} for call in tool_calls
            ]
        if message.get("task") is not None:
            marker["task"] = message["task"]
        markers.append(marker)
    order: list[int] = []
    for merged in sort_tool_results_by_call_order(merge_tool_messages(markers)):
        blocks = merged.get("content_blocks")
        if blocks is None:
            order.append(merged["content"])
        else:
            order.extend(block["content"] if block["type"] == "tool_result" else block["text"] for block in blocks)
    return [messages[index] for index in order]


class DeepseekV41Renderer(DeepseekV4Renderer):
    """Keep V4.1 tool-result ordering out of the shared V4 renderer."""

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return super().render_messages(_order_like_v41_prompt(messages), params)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return await super().render_messages_async(_order_like_v41_prompt(messages), params)
