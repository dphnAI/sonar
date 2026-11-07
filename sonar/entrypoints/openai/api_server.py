# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

from aphrodite.entrypoints.launchers.api_server.app_state import init_app_state
from aphrodite.entrypoints.launchers.api_server.entry import (
    build_and_serve,
    build_async_engine_client,
    build_async_engine_client_from_engine_args,
    run_server,
    run_server_worker,
)
from aphrodite.entrypoints.launchers.api_server.routers import register_api_routers
from aphrodite.entrypoints.launchers.app import build_app
from aphrodite.entrypoints.launchers.launcher import (
    create_server_socket,
    create_server_unix_socket,
    setup_server,
    validate_api_server_args,
)
from aphrodite.entrypoints.launchers.render.app_state import init_render_app_state
from aphrodite.entrypoints.launchers.render.entry import build_and_serve_renderer

warnings.warn(
    "`aphrodite.entrypoints.openai.api_server` is deprecated and will likely be"
    "unsupported in a future version. Use the corresponding function from "
    "`aphrodite.entrypoints.launchers` instead.",
    DeprecationWarning,
    stacklevel=1,
)


__all__ = [
    "build_async_engine_client",
    "build_async_engine_client_from_engine_args",
    "build_app",
    "init_app_state",
    "init_render_app_state",
    "create_server_socket",
    "create_server_unix_socket",
    "validate_api_server_args",
    "setup_server",
    "build_and_serve",
    "build_and_serve_renderer",
    "run_server",
    "run_server_worker",
    "register_api_routers",
]

if __name__ == "__main__":
    warnings.warn(
        "The `python -m aphrodite.entrypoints.openai.api_server` command is deprecated "
        "and may be removed in a future release. Please use `aphrodite server` instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    from aphrodite.entrypoints.launchers.api_server.entry import main

    main()
