# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worked example `aphrodite.endpoint_plugins` entry point."""

from fastapi import FastAPI, HTTPException, Request


class DummyAdminEndpointPlugin:
    name = "dummy_admin_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/v1/admin/scheduler_config")
        async def scheduler_config(raw_request: Request):
            engine_client = raw_request.app.state.dummy_engine_client
            if engine_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="scheduler_config requires an engine, which this server does not have",
                )
            results = await engine_client.collective_rpc("get_scheduler_config")
            return {"scheduler_config": results}

    async def init_state(self, engine_client, state, args) -> None:
        state.dummy_engine_client = engine_client
