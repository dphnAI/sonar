# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `aphrodite.endpoint_plugins` framework."""

from argparse import Namespace
from typing import Any

import httpx
import pytest
from aphrodite_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin
from fastapi import FastAPI

from aphrodite.entrypoints.openai.api_server import (
    _attach_endpoint_plugins,
    _init_endpoint_plugins_state,
    build_app,
)
from aphrodite.entrypoints.openai.cli_args import make_arg_parser
from aphrodite.plugins import load_endpoint_plugins
from aphrodite.plugins.endpoint_plugins.interface import EndpointPlugin
from aphrodite.utils.argparse_utils import FlexibleArgumentParser


class _RaisingEndpointPlugin:
    """Factory that raises to exercise the instantiation-fails path."""

    name = "raising_endpoint_plugin"
    required_tasks = None

    def __init__(self):
        raise RuntimeError("boom")


class _FakeEngineClient:
    """Minimal stand-in exercising `collective_rpc`. Not a real engine."""

    def __init__(self, rpc_result: Any = None):
        self.rpc_result = rpc_result
        self.rpc_calls: list[tuple[str, tuple, dict]] = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args, kwargs or {}))
        return self.rpc_result


def _build_args() -> Namespace:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers()
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return serve_parser.parse_args([])


def _fake_loader(factories: dict[str, Any]):
    def _load_plugins_by_group(group: str) -> dict[str, Any]:
        assert group == "aphrodite.endpoint_plugins"
        return factories

    return _load_plugins_by_group


def test_dummy_plugin_satisfies_protocol():
    assert isinstance(DummyAdminEndpointPlugin(), EndpointPlugin)


def test_no_plugins_loaded_when_allowlist_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("APHRODITE_PLUGINS", raising=False)

    assert load_endpoint_plugins(("generate",)) == []


def test_no_plugins_loaded_when_allowlist_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("APHRODITE_PLUGINS", "")

    assert load_endpoint_plugins(("generate",)) == []


def test_plugin_loaded_when_allowlisted_and_task_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_plugin_skipped_when_required_tasks_miss(monkeypatch: pytest.MonkeyPatch):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "aphrodite.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(("embed",)) == []
    assert len(load_endpoint_plugins(("generate",))) == 1


def test_plugin_loaded_when_required_tasks_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")

    assert len(load_endpoint_plugins(supported_tasks=None)) == 1


def test_plugin_skipped_when_required_tasks_set_but_supported_tasks_none(
    monkeypatch: pytest.MonkeyPatch,
):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "aphrodite.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(supported_tasks=None) == []


def test_factory_raising_is_logged_and_skipped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APHRODITE_PLUGINS", "raising_endpoint_plugin,dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "aphrodite.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "raising_endpoint_plugin": _RaisingEndpointPlugin,
                "dummy_admin_endpoint_plugin": DummyAdminEndpointPlugin,
            }
        ),
    )

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_attach_is_noop_when_nothing_discovered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("APHRODITE_PLUGINS", raising=False)

    app = FastAPI()
    _attach_endpoint_plugins(app, ("generate",))

    assert app.state.endpoint_plugins == []


@pytest.mark.asyncio
async def test_init_state_is_noop_without_phase_a(monkeypatch: pytest.MonkeyPatch):
    from starlette.datastructures import State

    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")

    state = State()
    await _init_endpoint_plugins_state(_FakeEngineClient(), state, _build_args())

    assert not hasattr(state, "dummy_engine_client")


@pytest.mark.asyncio
async def test_render_server_attaches_endpoint_plugins_with_no_engine_client(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, ("render",))

    assert len(app.state.endpoint_plugins) == 1
    assert any(getattr(route, "path", None) == "/v1/admin/scheduler_config" for route in app.routes)

    await _init_endpoint_plugins_state(None, app.state, args)

    assert app.state.dummy_engine_client is None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_endpoint_plugin_end_to_end(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APHRODITE_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, supported_tasks=())

    assert len(app.state.endpoint_plugins) == 1
    assert any(getattr(route, "path", None) == "/v1/admin/scheduler_config" for route in app.routes)

    fake_engine_client = _FakeEngineClient(rpc_result=["cfg-a", "cfg-b"])
    await _init_endpoint_plugins_state(fake_engine_client, app.state, args)

    assert app.state.dummy_engine_client is fake_engine_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 200
    assert response.json() == {"scheduler_config": ["cfg-a", "cfg-b"]}
    assert fake_engine_client.rpc_calls == [("get_scheduler_config", (), {})]
