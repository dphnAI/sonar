# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sonar.entrypoints.serve.instrumentator.capacity import router
from sonar.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware


def test_capacity_auth_replica_scope_and_unavailable():
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthenticationMiddleware, tokens=["secret"])
    core = SimpleNamespace(
        core_engines=[rank.to_bytes(2, "little") for rank in [3, 1]],
        capacity_reports={
            rank: SimpleNamespace(
                max_model_len=1000,
                max_num_seqs=32,
                kv_cache_size_tokens=tokens,
                kv_cache_max_concurrency=concurrency,
            )
            for rank, tokens, concurrency in [(1, 7850, 7.85), (3, None, None), (4, 9000, 9.0)]
        },
    )
    app.state.engine_client = SimpleNamespace(engine_core=core)
    with TestClient(app) as client:
        assert client.get("/v1/capacity").status_code == 401
        response = client.get("/v1/capacity", headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200
        assert response.json() == {
            "schema_version": 1,
            "scope": "connected_engines",
            "engines": [
                {
                    "dp_rank": rank,
                    "max_model_len": 1000,
                    "max_num_seqs": 32,
                    "kv_cache_capacity_tokens": tokens,
                    "estimated_concurrency_at_max_model_len": concurrency,
                }
                for rank, tokens, concurrency in [(1, 7850, 7.85), (3, None, None)]
            ],
        }
        core.capacity_reports = {}
        assert client.get("/v1/capacity", headers={"Authorization": "Bearer secret"}).status_code == 503
