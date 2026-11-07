# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.datastructures import Headers

from sonar.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
)
from sonar.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from sonar.entrypoints.openai.completion.protocol import CompletionResponseChoice
from sonar.entrypoints.serve.engine.protocol import ErrorResponse
from sonar.outputs import CompletionOutput, RequestOutput
from sonar.proof.http import (
    DphnHTTPMiddleware,
    DphnNonceError,
    DphnProofContext,
    DphnProofError,
    get_proof_nonce,
    validate_proof_nonce,
)


def _client(receipts=None):
    if receipts is None:
        receipts = [{"rank": 0, "dp_rank": 0, "worker_id": "worker/dp/0/rank/0", "proof": "opaque"}]
    return SimpleNamespace(dphn_prove=AsyncMock(return_value=receipts))


def _result(tokens, *, index=0, finished=False):
    return RequestOutput(
        request_id="engine-request",
        prompt="prompt that must never be attested",
        prompt_token_ids=[900, 901],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text="text deliberately unrelated to the IDs",
                token_ids=tokens,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
    )


async def _results(*items):
    for item in items:
        yield item


async def _frames(*payloads):
    for payload in payloads:
        yield "data: [DONE]\n\n" if payload == "[DONE]" else f"data: {json.dumps(payload)}\n\n"


@pytest.mark.parametrize(
    "value",
    [None, "", " \t", "bad\x00nonce", "bad\tnonce", "bad\nnonce", "\x1f", "\x7f", "\xe9", "a" * 1025],
)
def test_nonce_rejected(value):
    headers = Headers({} if value is None else {"X-DPHN-Proof-Nonce": value})
    with pytest.raises(DphnNonceError):
        validate_proof_nonce(headers)


def test_nonce_is_per_request_and_mode_off_is_inert():
    disabled = SimpleNamespace(sonar_config=SimpleNamespace(dphn_network_config=None))
    assert get_proof_nonce(disabled, None) is None
    enabled = SimpleNamespace(sonar_config=SimpleNamespace(dphn_network_config=object()))
    for nonce in ("first", "second", " punctuation !~ ", "a" * 1024):
        request = SimpleNamespace(headers=Headers({"X-DPHN-Proof-Nonce": nonce}))
        assert get_proof_nonce(enabled, request) == nonce
    with pytest.raises(DphnNonceError):
        get_proof_nonce(enabled, None)
    with pytest.raises(DphnNonceError):
        validate_proof_nonce(Headers(raw=[(b"x-dphn-proof-nonce", b"one"), (b"x-dphn-proof-nonce", b"two")]))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "enabled, include_reasoning, parser_kind, rejected",
    [
        (True, False, "reasoning", True),
        (True, False, "tool_only", False),
        (True, False, "none", False),
        (True, True, "reasoning", False),
        (False, False, "reasoning", False),
    ],
)
async def test_hidden_reasoning_rejected_before_generation_only_in_receipt_mode(
    monkeypatch, stream, enabled, include_reasoning, parser_kind, rejected
):
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    generate = AsyncMock()
    monkeypatch.setattr(
        serving,
        "engine_client",
        SimpleNamespace(
            sonar_config=SimpleNamespace(dphn_network_config=object() if enabled else None),
            generate=generate,
        ),
        raising=False,
    )
    monkeypatch.setattr(serving, "renderer", SimpleNamespace(tokenizer=object()), raising=False)
    monkeypatch.setattr(serving, "_effective_chat_template_kwargs", lambda request: {})
    monkeypatch.setattr(serving, "parser_cls", None if parser_kind == "none" else object(), raising=False)
    monkeypatch.setattr(
        serving,
        "_make_parser",
        lambda *args: SimpleNamespace(reasoning_parser=object() if parser_kind == "reasoning" else None),
    )
    render_error = serving.create_error_response("render sentinel")
    render_chat_request = AsyncMock(return_value=render_error)
    monkeypatch.setattr(serving, "render_chat_request", render_chat_request)
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "question"}],
        stream=stream,
        include_reasoning=include_reasoning,
    )
    raw_request = SimpleNamespace(headers=Headers({"X-DPHN-Proof-Nonce": "nonce"}))
    response = await serving._create_chat_completion(request, raw_request)
    if rejected:
        assert isinstance(response, ErrorResponse)
        assert response.error.code == 400
        assert "include_reasoning=false" in response.error.message
        render_chat_request.assert_not_awaited()
    else:
        assert response is render_error
        render_chat_request.assert_awaited_once_with(request)
    generate.assert_not_called()


@pytest.mark.asyncio
async def test_delta_capture_binds_exact_tokens_for_each_prompt_and_choice():
    client = _client()
    context = DphnProofContext(client, "cmpl-base", "nonce", 4)
    # n=2 with two prompts: output.index is local to each prompt. Completion
    # choices use index + prompt_index*n, even when choices finish out of order.
    for prompt_index in range(2):
        source = _results(
            _result([10 + prompt_index], index=1),
            _result([20 + prompt_index], index=0, finished=True),
            _result([30 + prompt_index], index=1, finished=True),
        )
        async for _ in context.capture(source, index_offset=prompt_index * 2, delta=True):
            pass
    assert context.proofs[0]["output_token_ids"] == [20]
    assert context.proofs[1]["output_token_ids"] == [10, 30]
    assert context.proofs[2]["output_token_ids"] == [21]
    assert context.proofs[3]["output_token_ids"] == [11, 31]
    assert {call.args[0] for call in client.dphn_prove.await_args_list} == {
        f"cmpl-base/choice/{index}" for index in range(4)
    }
    assert all(call.args[1] == "nonce" for call in client.dphn_prove.await_args_list)
    assert all(900 not in call.args[2] for call in client.dphn_prove.await_args_list)


@pytest.mark.asyncio
async def test_final_only_capture_does_not_duplicate_cumulative_tokens():
    client = _client()
    context = DphnProofContext(client, "chatcmpl-base", "nonce", 1)
    async for _ in context.capture(_results(_result([1]), _result([1, 2], finished=True)), delta=False):
        pass
    client.dphn_prove.assert_awaited_once_with("chatcmpl-base/choice/0", "nonce", [1, 2])
    choice = CompletionResponseChoice(index=0, text="echoed prompt plus answer", finish_reason="stop")
    context.attach_response(SimpleNamespace(choices=[choice]))
    assert choice.x_dphn_proof is not None
    assert choice.x_dphn_proof["output_token_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_stream_attaches_receipt_only_to_terminal_choice_and_retains_usage():
    context = DphnProofContext(_client(), "chatcmpl-base", "nonce", 1)
    async for _ in context.capture(_results(_result([1]), _result([2], finished=True)), delta=True):
        pass
    frames = [
        frame
        async for frame in context.stream(
            _frames(
                {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 2}},
                "[DONE]",
            )
        )
    ]
    assert "x_dphn_proof" not in json.loads(frames[0][6:])["choices"][0]
    terminal = json.loads(frames[1][6:])["choices"][0]
    assert terminal["x_dphn_proof"] == context.proofs[0]
    assert json.loads(frames[2][6:])["usage"]["completion_tokens"] == 2
    assert frames[3] == "data: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipts",
    [
        [],
        [{}],
        [
            {"rank": 0, "dp_rank": 0, "worker_id": "w", "proof": "p"},
            {"rank": 0, "dp_rank": 0, "worker_id": "w", "proof": "p"},
        ],
    ],
)
async def test_missing_malformed_or_duplicate_receipts_fail_before_terminal_output(receipts):
    context = DphnProofContext(_client(receipts), "base", "nonce", 1)
    with pytest.raises(DphnProofError):
        async for _ in context.capture(_results(_result([1], finished=True)), delta=True):
            pytest.fail("A terminal engine output was released without a valid receipt")


@pytest.mark.asyncio
async def test_receipt_rpc_error_closes_source_without_terminal_output():
    client = _client()
    client.dphn_prove.side_effect = RuntimeError("worker failure")
    context = DphnProofContext(client, "base", "nonce", 1)
    closed = []

    async def source():
        try:
            yield _result([1], finished=True)
        finally:
            closed.append(True)

    with pytest.raises(DphnProofError):
        async for _ in context.capture(source(), delta=True):
            pytest.fail("A terminal engine output was released after a proof failure")
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payloads",
    [
        ({"choices": [{"index": 0, "finish_reason": "stop"}]}, "[DONE]"),
        ({"error": {"message": "worker failure"}}, "[DONE]"),
        ({"choices": []}, "[DONE]"),
    ],
)
async def test_stream_failures_never_emit_success_or_done(payloads):
    context = DphnProofContext(_client(), "base", "nonce", 1)
    frames = [frame async for frame in context.stream(_frames(*payloads))]
    assert "error" in json.loads(frames[-1][6:])
    assert all(
        choice.get("finish_reason") is None for frame in frames for choice in json.loads(frame[6:]).get("choices", [])
    )
    assert not any("[DONE]" in frame for frame in frames)


@pytest.mark.parametrize(
    "choice",
    [
        CompletionResponseChoice(index=0, text="answer"),
        ChatCompletionResponseStreamChoice(index=0, delta={"content": "answer"}),
    ],
)
def test_disabled_choice_serialization_omits_proof_field(choice):
    assert "x_dphn_proof" not in choice.model_dump()
    assert "x_dphn_proof" not in json.loads(choice.model_dump_json())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/v1/responses",
        "/v1/chat/completions/batch",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/audio/transcriptions",
        "/cohere/v2/chat",
        "/inference/v1/generate",
        "/invocations",
        "/generative_scoring",
        "/api/extra/generate/stream",
        "/v1/decisions",
        "/pooling",
    ],
)
async def test_unsupported_inference_routes_rejected_before_execution(path):
    app = AsyncMock()
    messages = []

    async def send(message):
        messages.append(message)

    await DphnHTTPMiddleware(app)({"type": "http", "path": path, "method": "POST"}, AsyncMock(), send)
    app.assert_not_awaited()
    assert messages[0]["status"] == 400


@pytest.mark.asyncio
async def test_realtime_websocket_cannot_bypass_proof_mode():
    app = AsyncMock()
    send = AsyncMock()
    await DphnHTTPMiddleware(app)({"type": "websocket", "path": "/v1/realtime"}, AsyncMock(), send)
    app.assert_not_awaited()
    assert send.await_args is not None
    assert send.await_args.args[0]["code"] == 1008


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path",
    [("GET", "/v1/capacity"), ("POST", "/v1/fault_tolerance/apply"), ("GET", "/v1/fault_tolerance/status")],
)
async def test_operational_routes_pass_through_in_proof_mode(method, path):
    app = AsyncMock()
    scope = {"type": "http", "path": path, "method": method}
    await DphnHTTPMiddleware(app)(scope, AsyncMock(), AsyncMock())
    app.assert_awaited_once()
