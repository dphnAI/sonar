# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

from sonar.entrypoints.launchers.api_server.app_state import init_app_state
from sonar.entrypoints.launchers.api_server.entry import (
    build_and_serve,
    build_async_engine_client,
    build_async_engine_client_from_engine_args,
    run_server,
    run_server_worker,
)
from sonar.entrypoints.launchers.api_server.routers import register_api_routers
from sonar.entrypoints.launchers.app import build_app
from sonar.entrypoints.launchers.launcher import (
    create_server_socket,
    create_server_unix_socket,
    setup_server,
    validate_api_server_args,
)
from sonar.entrypoints.launchers.render.app_state import init_render_app_state
from sonar.entrypoints.launchers.render.entry import build_and_serve_renderer

warnings.warn(
    "`sonar.entrypoints.openai.api_server` is deprecated and we will remove"
    "this file in v0.37.0. Use the corresponding function from"
    "`sonar.entrypoints.launchers` instead.",
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
        "The `python -m sonar.entrypoints.openai.api_server` command is deprecated "
        "and we will remove this file in v0.37.0. Please use `sonar serve` instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    from sonar.entrypoints.launchers.api_server.entry import main

    main()
