# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import functools
from typing import Annotated, Literal

import pydantic
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

import aphrodite.envs as envs
from aphrodite.collect_env import get_env_info
from aphrodite.config import AphroditeConfig
from aphrodite.logger import init_logger

logger = init_logger(__name__)


router = APIRouter()
PydanticAphroditeConfig = pydantic.TypeAdapter(AphroditeConfig)


def _get_aphrodite_env_vars():
    from aphrodite.config.utils import normalize_value

    aphrodite_envs = {}
    for key in dir(envs):
        if key.startswith("APHRODITE_") and "KEY" not in key:
            value = getattr(envs, key, None)
            if value is not None:
                value = normalize_value(value)
                aphrodite_envs[key] = value
    return aphrodite_envs


@functools.lru_cache(maxsize=1)
def _get_system_env_info_cached():
    return get_env_info()._asdict()


@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    aphrodite_config: AphroditeConfig = raw_request.app.state.aphrodite_config
    server_info = {
        "aphrodite_config": (
            str(aphrodite_config)
            if config_format == "text"
            else PydanticAphroditeConfig.dump_python(aphrodite_config, mode="json", fallback=str)
        ),
        # fallback=str is needed to handle e.g. torch.dtype
        "aphrodite_env": _get_aphrodite_env_vars(),
        "system_env": await asyncio.to_thread(_get_system_env_info_cached),
    }
    return JSONResponse(content=server_info)


def attach_router(app: FastAPI):
    app.include_router(router)
