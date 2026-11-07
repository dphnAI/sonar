# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
import json

from sonar.tokenizers.deepseek_v41_encoding import encode_messages


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
