# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import functools
from typing import Annotated, Literal

import pydantic
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

import sonar.envs as envs
from sonar.collect_env import get_env_info
from sonar.config import SonarConfig
from sonar.logger import init_logger

logger = init_logger(__name__)


router = APIRouter()
PydanticSonarConfig = pydantic.TypeAdapter(SonarConfig)


def _get_sonar_env_vars():
    from sonar.config.utils import normalize_value

    sonar_envs = {}
    for key in dir(envs):
        if key.startswith("SONAR_") and "KEY" not in key:
            value = getattr(envs, key, None)
            if value is not None:
                value = normalize_value(value)
                sonar_envs[key] = value
    return sonar_envs


@functools.lru_cache(maxsize=1)
def _get_system_env_info_cached():
    return get_env_info()._asdict()


@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    sonar_config: SonarConfig = raw_request.app.state.sonar_config
    server_info = {
        "sonar_config": (
            str(sonar_config)
            if config_format == "text"
            else PydanticSonarConfig.dump_python(sonar_config, mode="json", fallback=str)
        ),
        # fallback=str is needed to handle e.g. torch.dtype
        "sonar_env": _get_sonar_env_vars(),
        "system_env": await asyncio.to_thread(_get_system_env_info_cached),
    }
    return JSONResponse(content=server_info)


def attach_router(app: FastAPI):
    app.include_router(router)
