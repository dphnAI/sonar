# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for `aphrodite.endpoint_plugins` entry points.

An endpoint plugin adds HTTP routes to the OpenAI-compatible API server. Its
scope is the HTTP surface only: registering routes and optional per-app state
used by those routes. It reaches the engine the same way an in-tree serving
handler does, through the `EngineClient` it is handed at startup.

If a plugin also needs engine-side behavior, pair this entry point with one
registered under `aphrodite.general_plugins`. The general plugin installs the
engine-side method and the endpoint plugin exposes it over HTTP. The two are
registered and loaded independently.
"""

from argparse import Namespace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import FastAPI
from starlette.datastructures import State

if TYPE_CHECKING:
    from aphrodite.engine.protocol import EngineClient
    from aphrodite.tasks import SupportedTask


@runtime_checkable
class EndpointPlugin(Protocol):
    """Protocol implemented by `aphrodite.endpoint_plugins` factories.

    An entry point registered under the `aphrodite.endpoint_plugins` group must
    resolve to a zero-argument callable that returns an object satisfying this
    protocol.
    """

    name: str
    """Unique plugin name used in logs."""

    required_tasks: "tuple[SupportedTask, ...] | None"
    """Tasks the server must support for this plugin to be loaded.

    The plugin is loaded only if this set intersects the server's
    `supported_tasks`. `None` means the plugin has no task requirement and is
    always eligible, subject to the `APHRODITE_PLUGINS` allowlist.
    """

    def attach_router(self, app: FastAPI) -> None:
        """Register this plugin's routes on `app`.

        Called once during `build_app()` after all core routers have been
        attached. Routes attached here can shadow core routes with the same path.
        """
        ...

    async def init_state(self, engine_client: "EngineClient | None", state: State, args: Namespace) -> None:
        """Initialize per-app state consumed by this plugin's routes.

        Called once during `init_app_state()` after core state has been
        initialized. Use `engine_client` to reach the engine.

        `engine_client` is `None` on the CPU-only render server, which has no
        engine. This only happens for plugins eligible for the `render` task.
        Handle `None` explicitly if the plugin is loadable for `render` but
        cannot function without an engine.
        """
        ...
