# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that http_requests_total metric records correct status codes."""

from argparse import Namespace
from http import HTTPStatus

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from aphrodite.entrypoints.serve.utils.server_utils import exception_handler
from aphrodite.exceptions import APHRODITENotFoundError, APHRODITEValidationError


@pytest.fixture
def registry():
    """Create a fresh Prometheus registry for each test."""
    return CollectorRegistry()


@pytest.fixture
def app(registry):
    """Create a minimal FastAPI app mirroring Aphrodite's handlers."""

    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)

    app.exception_handler(HTTPException)(_http_exception_handler)
    app.exception_handler(RequestValidationError)(_validation_exception_handler)
    app.exception_handler(ValueError)(exception_handler)
    app.exception_handler(TypeError)(exception_handler)
    app.exception_handler(OverflowError)(exception_handler)
    app.exception_handler(NotImplementedError)(exception_handler)
    app.exception_handler(APHRODITEValidationError)(exception_handler)
    app.exception_handler(APHRODITENotFoundError)(exception_handler)
    app.exception_handler(Exception)(exception_handler)

    Instrumentator(
        excluded_handlers=["/metrics"],
        registry=registry,
    ).add().instrument(app)

    @app.get("/raise_value_error")
    async def raise_value_error():
        raise ValueError("invalid input value")

    @app.get("/raise_type_error")
    async def raise_type_error():
        raise TypeError("wrong type")

    @app.get("/raise_overflow_error")
    async def raise_overflow_error():
        raise OverflowError("number too large")

    @app.get("/raise_not_implemented_error")
    async def raise_not_implemented_error():
        raise NotImplementedError("feature not supported")

    @app.get("/raise_aphrodite_validation_error")
    async def raise_aphrodite_validation_error():
        raise APHRODITEValidationError("bad parameter", parameter="temperature")

    @app.get("/raise_aphrodite_not_found_error")
    async def raise_aphrodite_not_found_error():
        raise APHRODITENotFoundError("model not found")

    @app.get("/raise_http_exception_400")
    async def raise_http_exception_400():
        raise HTTPException(status_code=400, detail="bad request")

    @app.get("/raise_http_exception_404")
    async def raise_http_exception_404():
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/raise_runtime_error")
    async def raise_runtime_error():
        raise RuntimeError("unexpected server error")

    @app.get("/success")
    async def success():
        return {"status": "ok"}

    return app


async def _http_exception_handler(req: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def _validation_exception_handler(req: Request, exc: RequestValidationError):
    return JSONResponse({"error": str(exc)}, status_code=HTTPStatus.BAD_REQUEST)


def _get_http_requests_total(registry, method: str, handler: str):
    """Extract the http_requests_total metric values grouped by status."""
    results: dict[str, float] = {}
    for metric in registry.collect():
        if metric.name == "http_requests":
            for sample in metric.samples:
                if (
                    sample.name == "http_requests_total"
                    and sample.labels.get("method") == method
                    and sample.labels.get("handler") == handler
                ):
                    status = sample.labels.get("status")
                    results[status] = results.get(status, 0) + sample.value
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,expected_status_group,expected_http_code",
    [
        ("/raise_value_error", "4xx", 400),
        ("/raise_type_error", "4xx", 400),
        ("/raise_overflow_error", "4xx", 400),
        ("/raise_aphrodite_validation_error", "4xx", 400),
        ("/raise_aphrodite_not_found_error", "4xx", 404),
        ("/raise_http_exception_400", "4xx", 400),
        ("/raise_http_exception_404", "4xx", 404),
        ("/raise_not_implemented_error", "5xx", 501),
        ("/raise_runtime_error", "5xx", 500),
        ("/success", "2xx", 200),
    ],
    ids=[
        "ValueError->4xx",
        "TypeError->4xx",
        "OverflowError->4xx",
        "APHRODITEValidationError->4xx",
        "APHRODITENotFoundError->4xx",
        "HTTPException(400)->4xx",
        "HTTPException(404)->4xx",
        "NotImplementedError->5xx",
        "RuntimeError->5xx",
        "success->2xx",
    ],
)
async def test_http_requests_total_records_correct_status(
    app,
    registry,
    endpoint,
    expected_status_group,
    expected_http_code,
):
    """Verify that http_requests_total records the returned status group."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(endpoint)

    assert response.status_code == expected_http_code

    metrics = _get_http_requests_total(registry, "GET", endpoint)
    assert metrics[expected_status_group] == 1.0

    if expected_status_group == "4xx":
        assert "5xx" not in metrics
