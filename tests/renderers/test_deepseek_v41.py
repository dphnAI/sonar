# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import json

import pytest

from aphrodite.renderers.deepseek_v4 import _order_like_v41_prompt
from aphrodite.tokenizers.deepseek_v41 import _normalize_messages
from aphrodite.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages


def _call(call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": call_id})}}


def _tool_loop(*, follow_up: bool) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "reasoning_content": "think-1", "tool_calls": [_call("c0")]},
        {"role": "tool", "tool_call_id": "c0", "content": "r0"},
        {"role": "assistant", "content": "", "reasoning_content": "think-2", "tool_calls": [_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
    ]
    if follow_up:
        messages.append({"role": "user", "content": "follow-up"})
    return messages


def test_tool_results_do_not_start_a_new_turn():
    # Without tools defined, reasoning before the last user turn is dropped;
    # tool results inside the loop are not a user turn.
    prompt = encode_messages(_tool_loop(follow_up=False), thinking_mode="thinking")
    assert "question<｜Assistant｜><think>think-1</think>" in prompt
    assert "<tool_result>r0</tool_result><｜Assistant｜><think>think-2</think>" in prompt


def test_user_text_after_tool_results_starts_a_new_turn():
    prompt = encode_messages(_tool_loop(follow_up=True), thinking_mode="thinking")
    assert "think-1" not in prompt and "think-2" not in prompt
    assert prompt.endswith("follow-up<｜Assistant｜><think>")


def _image_tool(call_id: str, url: str) -> dict:
    content = [{"type": "text", "text": f"result {call_id}"}, {"type": "image_url", "image_url": {"url": url}}]
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _parallel_images(results: list[dict]) -> list[dict]:
    calls = {"role": "assistant", "content": "", "tool_calls": [_call("c0"), _call("c1"), _call("c2")]}
    return [{"role": "user", "content": "look"}, calls, *results, {"role": "user", "content": "compare"}]


def _encode(messages: list[dict]) -> str:
    return encode_messages(_normalize_messages(messages), thinking_mode="chat")


def test_parallel_tool_images_are_parsed_in_prompt_order():
    a, b, c = _image_tool("c0", "a.png"), _image_tool("c1", "b.png"), _image_tool("c2", "c.png")
    messages = _parallel_images([c, a, b])
    ordered = _order_like_v41_prompt(messages)
    assert ordered == _parallel_images([a, b, c])
    # The prompt itself is unchanged, and its placeholders now follow the parse order.
    prompt = _encode(messages)
    assert _encode(ordered) == prompt
    positions = [prompt.index(f"result {call_id}") for call_id in ("c0", "c1", "c2")]
    assert positions == sorted(positions)
    assert prompt.count(IMAGE_PLACEHOLDER) == 3


@pytest.mark.parametrize(
    "messages",
    [
        _tool_loop(follow_up=True),
        _parallel_images([_image_tool("c0", "a.png"), _image_tool("c1", "b.png"), _image_tool("c2", "c.png")]),
        [
            {"role": "system", "content": "sys"},
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": "q"},
        ],
    ],
)
def test_prompt_order_is_identity_when_nothing_moves(messages):
    assert _order_like_v41_prompt(messages) == messages
