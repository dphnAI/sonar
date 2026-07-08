---
title: Endpoint plugins
---

Endpoint plugins let out-of-tree packages add HTTP routes to the
OpenAI-compatible API server without editing Aphrodite's API server. Their
scope is the HTTP surface only: registering routes and optional app state used
by those routes. A plugin reaches the engine through the `EngineClient` it is
handed at startup.

:::warning
Endpoint plugins are not loaded by default and must be explicitly allowlisted
with `APHRODITE_PLUGINS`. Treat them as trusted server code.
:::

## Interface

Endpoint plugins implement the `EndpointPlugin` protocol:

```python
class EndpointPlugin(Protocol):
    name: str
    required_tasks: tuple[SupportedTask, ...] | None

    def attach_router(self, app: FastAPI) -> None: ...

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None: ...
```

- `name`: a unique identifier used in logs
- `required_tasks`: tasks the server must support for this plugin to load;
  `None` means no task requirement
- `attach_router`: registers routes on `app`
- `init_state`: initializes app state the routes read at request time

## Lifecycle

Routes are registered before the engine exists, so the interface has two hooks:

| Phase | Called from | Engine client | Work |
| --- | --- | --- | --- |
| Route registration | `build_app()` | No | Add routes with `attach_router(app)`. |
| State init | `init_app_state()` | Usually | Store engine-backed state with `init_state(...)`. |

The CPU-only render server has no `EngineClient`. It still runs both phases for
plugins eligible for the `render` task, but calls `init_state` with
`engine_client=None`. Plugins that need an engine should either exclude
`render` from `required_tasks` or return an appropriate error from their route.

## Registering

Register a zero-argument factory under the `aphrodite.endpoint_plugins` entry
point group. The factory must return an object satisfying `EndpointPlugin`:

```toml
[project.entry-points."aphrodite.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"
```

```python
setup(
    name="my_pkg",
    entry_points={
        "aphrodite.endpoint_plugins": [
            "my_admin_api = my_pkg.endpoints:MyAdminEndpointPlugin"
        ]
    },
)
```

Enable the plugin by setting `APHRODITE_PLUGINS` to the entry point name:

```bash
APHRODITE_PLUGINS=my_admin_api aphrodite run ...
```

## Security

Endpoint plugins can register arbitrary FastAPI routes, including routes that
reach the engine via `EngineClient.collective_rpc`. Only allowlist plugins you
trust, audit their routes before deployment, and prefer route prefixes such as
`/plugins/<plugin-name>/...` to avoid shadowing core endpoints.

If a plugin also needs engine-side behavior, ship that separately through
`aphrodite.general_plugins`. The endpoint and general plugin entry points are
loaded independently.
