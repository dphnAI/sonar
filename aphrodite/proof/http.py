# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in HTTP transport for sampled runtime weight receipts.

These receipts bind output token IDs to sampled weights, not to inference
execution. No prompt tokens or retokenized response text enter the binding.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from aphrodite.proof import NONCE_HEADER, PROOF_VERSION

if TYPE_CHECKING:
    from fastapi import Request

    from aphrodite.engine.protocol import EngineClient
    from aphrodite.outputs import RequestOutput

MAX_NONCE_BYTES = 1024


class DphnProofError(RuntimeError):
    """An enabled request cannot be completed with a weight receipt."""


class DphnNonceError(DphnProofError):
    """The caller did not supply a usable per-request nonce."""


def validate_proof_nonce(headers: Headers) -> str:
    values = headers.getlist(NONCE_HEADER)
    if len(values) != 1:
        raise DphnNonceError(f"Exactly one {NONCE_HEADER} header is required")
    nonce = values[0]
    if not nonce.strip() or len(nonce) > MAX_NONCE_BYTES or any(not 0x20 <= ord(char) <= 0x7E for char in nonce):
        raise DphnNonceError(f"{NONCE_HEADER} must be nonblank printable ASCII and at most {MAX_NONCE_BYTES} bytes")
    return nonce


def get_proof_nonce(engine_client: "EngineClient", raw_request: "Request | None") -> str | None:
    if getattr(getattr(engine_client, "aphrodite_config", None), "dphn_network_config", None) is None:
        return None
    if raw_request is None:
        raise DphnNonceError(f"DPHN network requests require {NONCE_HEADER}")
    return validate_proof_nonce(raw_request.headers)


class DphnProofContext:
    """Request-local capture below text/echo/parser transformations.

    Capture runs before the existing response generators see terminal output,
    so a receipt failure cannot release a successful terminal choice. The SSE
    adapter only runs in opt-in mode and leaves upstream parsers untouched.
    """

    def __init__(self, engine_client: "EngineClient", request_id: str, nonce: str, num_choices: int):
        self.engine_client = engine_client
        self.request_id = request_id
        self.nonce = nonce
        self.num_choices = num_choices
        self.token_ids: dict[int, list[int]] = {}
        self.proofs: dict[int, dict[str, Any]] = {}
        self.sent_choices: set[int] = set()

    async def capture(
        self,
        source: AsyncGenerator["RequestOutput", None],
        *,
        index_offset: int = 0,
        delta: bool,
    ) -> AsyncGenerator["RequestOutput", None]:
        async with aclosing(source):
            async for result in source:
                for output in result.outputs:
                    index = index_offset + output.index
                    if index in self.proofs:
                        raise DphnProofError("Received output after a completed DPHN choice")
                    if delta:
                        tokens = self.token_ids.setdefault(index, [])
                        tokens.extend(output.token_ids)
                    else:
                        tokens = self.token_ids[index] = list(output.token_ids)
                    if output.finish_reason is not None and output.finish_reason not in ("error", "abort"):
                        choice_request_id = f"{self.request_id}/choice/{index}"
                        try:
                            receipts = await self.engine_client.dphn_prove(choice_request_id, self.nonce, list(tokens))
                        except Exception as exc:
                            raise DphnProofError("Failed to generate DPHN weight receipts") from exc
                        if not isinstance(receipts, list) or not receipts:
                            raise DphnProofError("No DPHN worker receipts returned")
                        ranks: set[tuple[int, int]] = set()
                        for receipt in receipts:
                            if (
                                not isinstance(receipt, dict)
                                or type(receipt.get("rank")) is not int
                                or type(receipt.get("dp_rank")) is not int
                                or not isinstance(receipt.get("worker_id"), str)
                                or not receipt["worker_id"]
                                or not isinstance(receipt.get("proof"), str)
                                or not receipt["proof"]
                            ):
                                raise DphnProofError("Malformed DPHN worker receipt")
                            rank = (receipt["dp_rank"], receipt["rank"])
                            if rank in ranks:
                                raise DphnProofError("Duplicate DPHN worker receipt")
                            ranks.add(rank)
                        self.proofs[index] = {
                            "version": PROOF_VERSION,
                            "request_id": choice_request_id,
                            "nonce": self.nonce,
                            "output_token_ids": list(tokens),
                            "proofs": receipts,
                        }
                yield result

    def _choice_proof(self, index: int) -> dict[str, Any]:
        if index not in self.proofs:
            raise DphnProofError("Missing DPHN receipt for completed choice")
        return self.proofs[index]

    def attach_response(self, response: Any) -> Any:
        # ErrorResponse must retain the existing error/cleanup path.
        if not hasattr(response, "choices"):
            return response
        if len(response.choices) != self.num_choices:
            raise DphnProofError("Incomplete DPHN completion response")
        for choice in response.choices:
            choice.x_dphn_proof = self._choice_proof(choice.index)
        return response

    async def stream(self, source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        try:
            async with aclosing(source):
                async for frame in source:
                    # OpenAI serving yields exactly one data event per frame.
                    if frame == "data: [DONE]\n\n":
                        if len(self.sent_choices) != self.num_choices:
                            raise DphnProofError("DPHN stream ended without all completed choices")
                        yield frame
                        return
                    if not frame.startswith("data: "):
                        raise DphnProofError("Unexpected DPHN completion stream frame")
                    payload = json.loads(frame[6:])
                    if "error" in payload:
                        yield frame
                        return  # Never send [DONE] following an error.
                    changed = False
                    for choice in payload.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            index = choice["index"]
                            if index in self.sent_choices:
                                raise DphnProofError("Duplicate DPHN terminal choice")
                            choice["x_dphn_proof"] = self._choice_proof(index)
                            self.sent_choices.add(index)
                            changed = True
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n" if changed else frame
            raise DphnProofError("DPHN completion stream ended without [DONE]")
        except Exception:
            # Do not leak RPC internals or emit a successful terminal marker.
            yield (
                'data: {"error":{"message":"DPHN weight receipt generation failed",'
                '"type":"InternalServerError","code":500}}\n\n'
            )


class DphnHTTPMiddleware:
    """Reject inference surfaces without the per-choice receipt contract.

    Installed only when --dphn-network is enabled. Operational, model-listing,
    tokenization, and rendering endpoints do not execute model inference.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        path = path.rstrip("/")
        supported = path in ("/v1/chat/completions", "/v1/completions")
        non_inference = (
            path
            in (
                "/v1/models",
                "/v1/capacity",
                "/v1/fault_tolerance/apply",
                "/v1/fault_tolerance/status",
                "/v1/tokenize",
                "/v1/detokenize",
                "/v1/load_lora_adapter",
                "/v1/unload_lora_adapter",
                "/v1/messages/count_tokens",
                "/inference/v1/abort_requests",
            )
            or path.endswith("/render")
            or path.endswith("/derender")
        )
        unsupported = (
            not supported
            and not non_inference
            and (
                path.startswith(("/v1/", "/v2/", "/cohere/", "/inference/"))
                or path
                in (
                    "/generate",
                    "/invocations",
                    "/pooling",
                    "/classify",
                    "/score",
                    "/rerank",
                    "/generative_scoring",
                    "/api/latest/generate",
                    "/api/v1/generate",
                    "/api/extra/generate/stream",
                )
            )
        )
        if unsupported:
            message = "DPHN weight receipts support only /v1/chat/completions and /v1/completions"
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": message})
            else:
                response = JSONResponse(
                    {"error": {"message": message, "type": "BadRequestError", "code": 400}},
                    status_code=400,
                )
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
