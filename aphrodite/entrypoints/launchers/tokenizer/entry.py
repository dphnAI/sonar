# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer-only server: a model's tokenizer behind the main server's tokenization API.

Loads only the model config and tokenizer (no engine, no weights, no GPU) and serves
`/tokenize`, `/detokenize` (and `/tokenizer_info` with `--enable-tokenizer-info-endpoint`)
with the same handlers as `aphrodite serve`, plus `/v1/models`, `/health`, `/version`
and `/metrics`.
"""

import argparse
from argparse import Namespace

from fastapi import FastAPI

from aphrodite.entrypoints.openai.models.api_router import attach_router as attach_models_router
from aphrodite.entrypoints.serve.exception_handling.register import init_exception_handler
from aphrodite.entrypoints.serve.instrumentator import register_instrumentator_api_routers
from aphrodite.entrypoints.serve.middleware.register import init_entrypoints_middleware
from aphrodite.entrypoints.serve.tokenize.api_router import attach_router as attach_tokenize_router

from ..render.entry import run_launch_fastapi
from ..utils.server_utils import lifespan


def build_tokenizer_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.root_path = args.root_path

    register_instrumentator_api_routers(app)
    attach_tokenize_router(app)
    attach_models_router(app)

    init_exception_handler(app)
    init_entrypoints_middleware(args, app, ("render",))
    return app


async def run_tokenizer_server(args: argparse.Namespace) -> None:
    """Serve the tokenizer of `args.model` without starting an engine."""
    await run_launch_fastapi(args, app_builder=build_tokenizer_app)
