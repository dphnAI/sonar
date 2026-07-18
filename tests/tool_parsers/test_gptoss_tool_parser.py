# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from unittest.mock import Mock

import pytest
from openai_harmony import (
    Conversation,
    Message,
    RenderConversationConfig,
    Role,
)

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from aphrodite.entrypoints.openai.parser.harmony_utils import get_encoding
from aphrodite.tool_parsers.gptoss_tool_parser import GptOssToolParser


@pytest.fixture
def parser() -> GptOssToolParser:
    # extract_tool_calls never touches the tokenizer directly: Harmony's own
    # encoding (via get_encoding()) is what turns token_ids into messages.
    return GptOssToolParser(Mock())


@pytest.fixture
def chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
    )


def encode_output(harmony_str: str) -> list[int]:
    return get_encoding().encode(harmony_str, allowed_special="all")


def assistant(content: str, channel: str) -> Message:
    return Message.from_role_and_content(Role.ASSISTANT, content).with_channel(channel)


def tool_call(
    recipient: str,
    content: str,
    channel: str = "commentary",
    content_type: str | None = "json",
) -> Message:
    message = assistant(content, channel).with_recipient(recipient)
    return message if content_type is None else message.with_content_type(content_type)


def get_model_output_tokens(
    prompt_messages: Sequence[Message],
    response_messages: Sequence[Message],
) -> list[int]:
    enc = get_encoding()
    config = RenderConversationConfig(auto_drop_analysis=False)
    prompt_ids = enc.render_conversation_for_completion(
        Conversation.from_messages(list(prompt_messages)),
        Role.ASSISTANT,
        config=config,
    )
    full_ids = enc.render_conversation(
        Conversation.from_messages([*prompt_messages, *response_messages]),
        config=config,
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids
    return full_ids[len(prompt_ids) :]


def tool_call_tuples(tool_calls) -> list[tuple[str, str]]:
    return [(tc.function.name, tc.function.arguments) for tc in tool_calls]


class TestExtractToolCalls:
    def test_no_tool_call(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Hello")]
        response = [assistant("This is a test", "final")]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "This is a test"

    def test_single_tool_call(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?")]
        response = [tool_call("functions.get_current_weather", '{"location": "Tokyo"}')]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.tools_called is True
        assert result.content is None
        assert tool_call_tuples(result.tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
        ]
        assert result.tool_calls[0].id is not None
        assert result.tool_calls[0].type == "function"

    def test_multiple_tool_calls_varied_formats(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Use several tools.")]
        response = [
            tool_call("functions.get_current_weather", '{"location": "Tokyo"}'),
            tool_call("functions.no_content_type", '{"location": "Tokyo"}', content_type=None),
            tool_call("functions.not_json_no_content_type", "foo", content_type=None),
            tool_call("functions.empty_args", "{}"),
            tool_call("functions.no_args", ""),
        ]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.content is None
        assert tool_call_tuples(result.tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
            ("no_content_type", json.dumps({"location": "Tokyo"})),
            ("not_json_no_content_type", "foo"),
            ("empty_args", json.dumps({})),
            ("no_args", ""),
        ]

    def test_tool_call_bare_recipient(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Weather?")]
        response = [tool_call("get_current_weather", '{"location": "Tokyo"}')]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert tool_call_tuples(result.tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
        ]

    def test_tool_call_dotted_name(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Compute 2+3")]
        response = [tool_call("math.sum", '{"a": 2, "b": 3}')]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert tool_call_tuples(result.tool_calls) == [("math.sum", json.dumps({"a": 2, "b": 3}))]

    def test_assistant_recipient_not_tool(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "Hello")]
        response = [
            tool_call("assistant", "Some tool response", content_type=None),
            assistant("Here is the answer", "final"),
        ]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Here is the answer"

    def test_reasoning_excluded_from_content(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "What is 2+2?")]
        response = [
            assistant("I should think first.", "analysis"),
            assistant("The answer is 4.", "final"),
        ]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.tools_called is False
        assert result.content == "The answer is 4."

    def test_tool_calls_with_commentary_preamble(self, parser, chat_request):
        prompt = [Message.from_role_and_content(Role.USER, "What is the weather?")]
        response = [
            assistant("User asked about the weather.", "analysis"),
            tool_call("functions.get_current_weather", '{"location": "Tokyo"}'),
            assistant("This tool call will get the weather.", "final"),
        ]

        result = parser.extract_tool_calls(
            "",
            token_ids=get_model_output_tokens(prompt, response),
            request=chat_request,
        )

        assert result.content == "This tool call will get the weather."
        assert tool_call_tuples(result.tool_calls) == [
            ("get_current_weather", json.dumps({"location": "Tokyo"})),
        ]

    @pytest.mark.parametrize(
        ("harmony_str", "expected_content"),
        [
            (
                "<|channel|>commentary<|message|>I'll search for that",
                "I'll search for that",
            ),
            (
                "<|channel|>commentary<|message|>Let me look that up.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>",
                "Let me look that up.\nThe answer is 42.",
            ),
        ],
    )
    def test_commentary_preambles(self, parser, chat_request, harmony_str, expected_content):
        result = parser.extract_tool_calls(
            "",
            token_ids=encode_output(harmony_str),
            request=chat_request,
        )

        assert result.tools_called is False
        assert result.content == expected_content

    def test_missing_token_ids_raises(self, parser, chat_request):
        with pytest.raises(ValueError, match="token_ids"):
            parser.extract_tool_calls(
                "some raw text",
                token_ids=None,
                request=chat_request,
            )


class TestExtractToolCallsStreaming:
    def test_not_implemented(self, parser, chat_request):
        with pytest.raises(NotImplementedError):
            parser.extract_tool_calls_streaming(
                previous_text="",
                current_text="",
                delta_text="",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=chat_request,
            )
