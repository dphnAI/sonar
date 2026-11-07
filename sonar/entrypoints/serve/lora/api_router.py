# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from aphrodite import envs
from aphrodite.entrypoints.openai.models.api_router import models
from aphrodite.entrypoints.openai.models.serving import OpenAIServingModels
from aphrodite.entrypoints.serve.engine.protocol import ErrorResponse
from aphrodite.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from aphrodite.entrypoints.serve.utils.api_utils import validate_json_request
from aphrodite.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


def attach_router(app: FastAPI):
    """Attach the LoRA adapter load/unload endpoints to the API server.

    Does nothing when dynamic LoRA updating is disabled. Handler levels are
    snapshotted and restored because importing
    model_hosting_container_standards may reconfigure root logging.
    """
    if not envs.APHRODITE_ALLOW_RUNTIME_LORA_UPDATING:
        """If LoRA dynamic loading & unloading is not enabled, do nothing."""
        return
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. This should ONLY be used for local development!"
    )

    snapshot = [(h, h.level) for lg in (logging.getLogger(), logging.getLogger("aphrodite")) for h in lg.handlers]

    try:
        try:
            import model_hosting_container_standards.sagemaker as sagemaker_standards
        except ImportError:
            logger.warning("Skipping runtime LoRA router because model_hosting_container_standards is unavailable.")
            return
        _attach_router(app, sagemaker_standards)
    finally:
        for handler, level in snapshot:
            handler.setLevel(level)


def _attach_router(app: FastAPI, sagemaker_standards):
    """Register the LoRA adapter load/unload routes on the app."""

    @sagemaker_standards.register_load_adapter_handler(
        request_shape={
            "lora_name": "body.name",
            "lora_path": "body.src",
            "load_inplace": "body.load_inplace || `false`",
            "is_3d_lora_weight": "body.is_3d_lora_weight || `false`",
        },
    )
    @router.post("/v1/load_lora_adapter", dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        """Handle POST /v1/load_lora_adapter: load a LoRA adapter into the
        serving engine."""
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.error.code)

        return Response(status_code=200, content=response)

    @sagemaker_standards.register_unload_adapter_handler(
        request_shape={
            "lora_name": "path_params.adapter_name",
        }
    )
    @router.post("/v1/unload_lora_adapter", dependencies=[Depends(validate_json_request)])
    async def unload_lora_adapter(request: UnloadLoRAAdapterRequest, raw_request: Request):
        """Handle POST /v1/unload_lora_adapter: unload a LoRA adapter from
        the serving engine."""
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.error.code)

        return Response(status_code=200, content=response)

    # register the router
    app.include_router(router)
