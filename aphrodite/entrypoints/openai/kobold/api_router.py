# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from aphrodite.entrypoints.openai.engine.protocol import ErrorResponse
from aphrodite.entrypoints.openai.kobold.protocol import (
    KAIGenerationInputSchema,
    KAITokenizeRequest,
)
from aphrodite.entrypoints.openai.kobold.serving import OpenAIServingKobold
from aphrodite.entrypoints.serve.utils.api_utils import validate_json_request
from aphrodite.entrypoints.utils import create_error_response

router = APIRouter()


def kobold(request: Request) -> OpenAIServingKobold | None:
    return request.app.state.openai_serving_kobold


@router.post(
    "/api/latest/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@router.post(
    "/api/v1/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def generate(kai_payload: KAIGenerationInputSchema, raw_request: Request):
    handler = kobold(raw_request)
    if handler is None:
        err = create_error_response(
            message="The model does not support KoboldAI API",
            status_code=HTTPStatus.BAD_REQUEST,
        )
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)

    result = await handler.create_kobold_response(kai_payload, raw_request)
    return JSONResponse(result)


@router.post(
    "/api/extra/generate/stream",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def generate_stream(kai_payload: KAIGenerationInputSchema, raw_request: Request):
    handler = kobold(raw_request)
    if handler is None:
        err = create_error_response(
            message="The model does not support KoboldAI streaming API",
            status_code=HTTPStatus.BAD_REQUEST,
        )
        return JSONResponse(content=err.model_dump(), status_code=err.error.code)

    generator = handler.create_kobold_stream(kai_payload, raw_request)
    return StreamingResponse(
        content=generator,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
        media_type="text/event-stream",
    )


@router.post("/api/extra/generate/check")
@router.get("/api/extra/generate/check")
async def check_generation(raw_request: Request):
    handler = kobold(raw_request)
    if handler is None:
        return JSONResponse({"results": [{"text": ""}]})

    genkey = raw_request.query_params.get("genkey")
    if genkey is None:
        try:
            request_dict = await raw_request.json()
            if isinstance(request_dict, dict):
                genkey = request_dict.get("genkey")
        except Exception:
            pass

    text = await handler.check_generation(genkey) if genkey else ""
    return JSONResponse({"results": [{"text": text}]})


@router.post("/api/extra/abort")
async def abort_generation(raw_request: Request):
    handler = kobold(raw_request)
    if handler is None:
        return JSONResponse({})

    try:
        request_dict = await raw_request.json()
        if isinstance(request_dict, dict) and "genkey" in request_dict:
            await handler.abort_generation(request_dict["genkey"])
    except Exception:
        pass

    return JSONResponse({})


@router.post(
    "/api/extra/tokencount",
    dependencies=[Depends(validate_json_request)],
)
async def count_tokens(request: KAITokenizeRequest, raw_request: Request):
    handler = kobold(raw_request)
    if handler is None:
        return JSONResponse({"value": 0})
    tokenizer = handler.renderer.get_tokenizer()
    token_ids = tokenizer.encode(request.prompt, add_special_tokens=False)
    return JSONResponse({"value": len(token_ids)})


@router.get("/api/latest/info/version")
@router.get("/api/v1/info/version")
async def get_version():
    return JSONResponse({"result": "1.2.4"})


@router.get("/api/latest/model")
@router.get("/api/v1/model")
async def get_model(raw_request: Request):
    handler = kobold(raw_request)
    model_name = handler.models.base_model_paths[0].name if handler is not None else raw_request.app.state.args.model
    return JSONResponse({"result": f"aphrodite/{model_name}"})


@router.get("/api/latest/config/soft_prompts_list")
@router.get("/api/v1/config/soft_prompts_list")
async def get_available_softprompts():
    return JSONResponse({"values": []})


@router.get("/api/latest/config/soft_prompt")
@router.get("/api/v1/config/soft_prompt")
async def get_current_softprompt():
    return JSONResponse({"value": ""})


@router.put("/api/latest/config/soft_prompt")
@router.put("/api/v1/config/soft_prompt")
async def set_current_softprompt():
    return JSONResponse({})


@router.get("/api/latest/config/max_length")
@router.get("/api/v1/config/max_length")
async def get_max_length(raw_request: Request):
    max_length = raw_request.app.state.aphrodite_config.model_config.max_model_len
    return JSONResponse({"value": max_length})


@router.get("/api/latest/config/max_context_length")
@router.get("/api/v1/config/max_context_length")
@router.get("/api/extra/true_max_context_length")
async def get_max_context_length(raw_request: Request):
    max_context_length = raw_request.app.state.aphrodite_config.model_config.max_model_len
    return JSONResponse({"value": max_context_length})


@router.get("/api/extra/preloadstory")
async def get_preloaded_story():
    return JSONResponse({})


@router.get("/api/extra/version")
async def get_extra_version():
    return JSONResponse({"result": "KoboldCpp", "version": "1.63"})


def attach_router(app: FastAPI):
    app.include_router(router)
