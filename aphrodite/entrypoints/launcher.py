# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
import socket
import sys
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from typing import Any

import uvicorn
from fastapi import FastAPI

from aphrodite import envs
from aphrodite.engine.protocol import EngineClient
from aphrodite.entrypoints.serve.utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from aphrodite.entrypoints.serve.utils.ssl import SSLCertRefresher
from aphrodite.logger import init_logger
from aphrodite.utils.network_utils import find_process_using_port

logger = init_logger(__name__)

_ROUTE_LOG_PATHS_PER_LINE = 3
_KOBOLD_COMPACT_HIDDEN_ROUTES = {
    "/api/{v1,latest}/config/soft_prompt",
    "/api/{v1,latest}/config/soft_prompts_list",
    "/api/extra/preloadstory",
}
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_BLUE = "\033[34m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_YELLOW = "\033[33m"
_ROUTE_GROUP_COLORS = {
    "Core": _ANSI_GREEN,
    "OpenAI": _ANSI_CYAN,
    "Kobold": _ANSI_MAGENTA,
    "Docs": _ANSI_DIM,
    "Other": _ANSI_YELLOW,
}
_ROUTE_METHOD_COLORS = {
    "DELETE": _ANSI_YELLOW,
    "ENDPOINT": _ANSI_MAGENTA,
    "GET": _ANSI_BLUE,
    "PATCH": _ANSI_YELLOW,
    "POST": _ANSI_GREEN,
    "PUT": _ANSI_YELLOW,
}


def _use_route_log_color() -> bool:
    if envs.NO_COLOR or envs.APHRODITE_LOGGING_COLOR == "0":
        return False
    if envs.APHRODITE_LOGGING_COLOR == "1":
        return True
    if envs.APHRODITE_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if envs.APHRODITE_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


def _route_log_color(text: str, color: str) -> str:
    if not _use_route_log_color():
        return text
    return f"{color}{text}{_ANSI_RESET}"


def _route_method_color(label: str) -> str:
    method = label.split(",", maxsplit=1)[0]
    return _ROUTE_METHOD_COLORS.get(method, _ANSI_CYAN)


def _normalize_methods(methods: Iterable[str]) -> str:
    methods_set = set(methods)
    if "GET" in methods_set:
        methods_set.discard("HEAD")
    return ", ".join(sorted(methods_set))


def _route_group(path: str) -> str:
    if path in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}:
        return "Docs"
    if path.startswith("/v1/") or path.startswith("/inference/v1/"):
        return "OpenAI"
    if path.startswith("/api/"):
        return "Kobold"
    if path in {
        "/health",
        "/version",
        "/metrics",
        "/load",
        "/ping",
        "/tokenize",
        "/detokenize",
        "/v1/tokenize",
        "/v1/detokenize",
    }:
        return "Core"
    return "Other"


def _compact_route_path(path: str) -> str:
    if path.startswith("/api/v1/"):
        return path.replace("/api/v1/", "/api/{v1,latest}/", 1)
    if path.startswith("/api/latest/"):
        return path.replace("/api/latest/", "/api/{v1,latest}/", 1)
    return path


def _log_full_routes(app: FastAPI) -> None:
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, _normalize_methods(methods))

    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if endpoint is None or path is None or methods is not None:
            continue

        logger.info("Route: %s, Endpoint: %s", path, endpoint.__name__)


def _log_compact_routes(app: FastAPI) -> None:
    grouped_routes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    endpoint_routes: dict[str, set[str]] = defaultdict(set)
    route_count = 0

    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue

        route_count += 1
        group = _route_group(path)
        compact_path = _compact_route_path(path)
        if compact_path in _KOBOLD_COMPACT_HIDDEN_ROUTES:
            continue

        methods = getattr(route, "methods", None)
        if methods is None:
            endpoint = getattr(route, "endpoint", None)
            endpoint_name = getattr(endpoint, "__name__", "unknown")
            endpoint_routes[group].add(f"{compact_path} -> {endpoint_name}")
            continue

        grouped_routes[group][_normalize_methods(methods)].add(compact_path)

    displayed_route_count = sum(
        len(paths) for method_groups in grouped_routes.values() for paths in method_groups.values()
    ) + sum(len(endpoints) for endpoints in endpoint_routes.values())
    alias_note = ""
    if displayed_route_count != route_count:
        alias_note = f", {displayed_route_count} shown"
    logger.info(
        "%s %d total%s %s",
        _route_log_color("Available routes:", _ANSI_BOLD),
        route_count,
        alias_note,
        _route_log_color("(set APHRODITE_LOG_ROUTES=full for details)", _ANSI_DIM),
    )

    for group in ("Core", "OpenAI", "Kobold", "Docs", "Other"):
        method_groups = grouped_routes[group]
        endpoints = endpoint_routes[group]
        if not method_groups and not endpoints:
            continue

        group_route_count = sum(len(paths) for paths in method_groups.values()) + len(endpoints)
        group_label = _route_log_color(
            f"Routes [{group}]:",
            f"{_ANSI_BOLD}{_ROUTE_GROUP_COLORS[group]}",
        )
        logger.info("%s %d", group_label, group_route_count)
        for methods, paths in sorted(method_groups.items()):
            _log_route_paths(methods, sorted(paths))
        if endpoints:
            _log_route_paths("ENDPOINT", sorted(endpoints))


def _log_route_paths(label: str, paths: list[str]) -> None:
    for idx in range(0, len(paths), _ROUTE_LOG_PATHS_PER_LINE):
        chunk = paths[idx : idx + _ROUTE_LOG_PATHS_PER_LINE]
        prefix = label if idx == 0 else ""
        prefix = f"{prefix:<10}"
        if prefix.strip():
            prefix = _route_log_color(prefix, _route_method_color(label))
        logger.info("  %s %s", prefix, ", ".join(chunk))


def _log_routes(app: FastAPI) -> None:
    route_log_format = envs.APHRODITE_LOG_ROUTES
    if route_log_format == "off":
        return
    if route_log_format == "full":
        _log_full_routes(app)
        return
    _log_compact_routes(app)


async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
    options.  Supports http header limits via h11_max_incomplete_event_size and
    h11_max_header_count.
    """
    _log_routes(app)

    # Extract header limit options if present
    h11_max_incomplete_event_size = uvicorn_kwargs.pop("h11_max_incomplete_event_size", None)
    h11_max_header_count = uvicorn_kwargs.pop("h11_max_header_count", None)

    # Set safe defaults if not provided
    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs)
    # Set header limits
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()
    server = uvicorn.Server(config)
    app.state.server = server

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs,
        )
    )

    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        if shutdown_event.is_set():
            return
        logger.info_once("[shutdown] API server: shutdown triggered")
        shutdown_event.set()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    async def handle_shutdown() -> None:
        await shutdown_event.wait()

        engine_client = app.state.engine_client
        timeout = engine_client.aphrodite_config.shutdown_timeout
        mode = "abort" if timeout == 0 else "drain"

        logger.info(
            "[shutdown] API server: stopping engine client mode=%s timeout=%ss",
            mode,
            timeout,
        )

        await loop.run_in_executor(None, partial(engine_client.shutdown, timeout=timeout))
        logger.info_once("[shutdown] API server: engine client stopped")

        server.should_exit = True
        logger.info_once("[shutdown] API server: signalling HTTP server shutdown")
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    shutdown_task = loop.create_task(handle_shutdown())

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.warning(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info_once("[shutdown] API server: shutting down FastAPI HTTP server")
        return server.shutdown()
    finally:
        shutdown_task.cancel()
        watchdog_task.cancel()


async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    APHRODITE_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(APHRODITE_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    """
    See discussions here on shutting down a uvicorn server
    https://github.com/encode/uvicorn/discussions/1103
    In this case we cannot await the server shutdown here
    because handler must first return to close the connection
    for this request.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.APHRODITE_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True
