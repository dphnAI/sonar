# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from aphrodite.engine.protocol import EngineClient
    from aphrodite.entrypoints.logger import RequestLogger
    from aphrodite.tasks import SupportedTask
else:
    RequestLogger = object


def register_generate_api_routers(app: FastAPI):
    from aphrodite.entrypoints.openai.chat_completion.api_router import (
        attach_router as register_chat_api_router,
    )

    register_chat_api_router(app)

    from aphrodite.entrypoints.openai.responses.api_router import (
        attach_router as register_responses_api_router,
    )

    register_responses_api_router(app)

    from aphrodite.entrypoints.openai.completion.api_router import (
        attach_router as register_completion_api_router,
    )

    register_completion_api_router(app)

    from aphrodite.entrypoints.openai.kobold.api_router import (
        attach_router as register_kobold_api_router,
    )

    register_kobold_api_router(app)

    from aphrodite.entrypoints.anthropic.api_router import (
        attach_router as register_anthropic_api_router,
    )

    register_anthropic_api_router(app)


async def init_generate_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    from aphrodite.entrypoints.anthropic.serving import AnthropicServingMessages
    from aphrodite.entrypoints.chat_utils import load_chat_template
    from aphrodite.entrypoints.mcp.tool_server import (
        DemoToolServer,
        MCPToolServer,
        ToolServer,
    )
    from aphrodite.entrypoints.openai.chat_completion.batch_serving import (
        OpenAIServingChatBatch,
    )
    from aphrodite.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from aphrodite.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from aphrodite.entrypoints.openai.fingerprint import set_default_fingerprint_mode
    from aphrodite.entrypoints.openai.kobold.serving import OpenAIServingKobold
    from aphrodite.entrypoints.openai.responses.serving import OpenAIServingResponses
    from aphrodite.entrypoints.serve.disagg.serving import ServingTokens

    # Applied before any serving class is constructed so that each one picks
    # up the chosen mode on its first cache miss.
    set_default_fingerprint_mode(
        getattr(args, "fingerprint_mode", "full"),
        getattr(args, "fingerprint_value", None),
    )

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None
    resolved_chat_template = load_chat_template(args.chat_template)

    # Render endpoints are always backed by OpenAIServingRender so that
    # /v1/chat/completions/render and /v1/completions/render work on both
    # generate-mode and render-only servers. Created in init_app_state.

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
        )
        if "generate" in supported_tasks
        else None
    )
    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        openai_serving_render=state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
    )
    state.openai_serving_chat = OpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    state.openai_serving_chat_batch = OpenAIServingChatBatch(**_chat_kwargs) if "generate" in supported_tasks else None
    if state.openai_serving_chat is not None:
        state.openai_serving_chat.warmup()
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_kobold = (
        OpenAIServingKobold(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if "generate" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )
