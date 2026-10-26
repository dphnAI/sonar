# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from aphrodite.config import AphroditeConfig
from aphrodite.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from aphrodite.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from aphrodite.tokenizers.deepseek_v41_encoding import merge_tool_messages, sort_tool_results_by_call_order
from aphrodite.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

_FIM_BEGIN = "<｜fim▁begin｜>"
_FIM_HOLE = "<｜fim▁hole｜>"
_FIM_END = "<｜fim▁end｜>"


def _order_like_v41_prompt(messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """Put messages in the order the V4.1 encoder places them in the prompt.

    The encoder sorts parallel tool results by call order; images are matched to
    placeholders in prompt order, so they must be parsed in that order too.
    """
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


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    def __init__(
        self,
        config: AphroditeConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(self._apply_chat_template, executor=self._executor)

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_completion_suffix(self, prompt: str, suffix: str) -> str | None:
        return f"{_FIM_BEGIN}{prompt}{_FIM_HOLE}{suffix}{_FIM_END}"

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        if self.model_config.tokenizer_mode == "deepseek_v41":
            messages = _order_like_v41_prompt(messages)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        if self.model_config.tokenizer_mode == "deepseek_v41":
            messages = _order_like_v41_prompt(messages)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
