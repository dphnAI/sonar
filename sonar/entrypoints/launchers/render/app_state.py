# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from starlette.datastructures import State

from sonar.config import SonarConfig
from sonar.entrypoints.common.cli_args import resolve_default_chat_template_kwargs
from sonar.entrypoints.mcp.tool_server import init_tool_server
from sonar.entrypoints.openai.models.protocol import BaseModelPath
from sonar.entrypoints.openai.models.serving import OpenAIModelRegistry
from sonar.entrypoints.scale_out.factories import init_render_state
from sonar.entrypoints.serve.tokenize.serving import ServingTokenization
from sonar.entrypoints.serve.utils.request_logger import RequestLogger
from sonar.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from sonar.renderers import renderer_from_config
from sonar.renderers.chat_utils import load_chat_template
from sonar.renderers.online_derenderer import OnlineDerenderer
from sonar.renderers.online_renderer import OnlineRenderer


async def init_render_app_state(
    sonar_config: SonarConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialise FastAPI app state for a CPU-only render server.

    Unlike `init_app_state` this function does not require an
    `EngineClient`; it bootstraps the
    preprocessing pipeline (renderer, input_processor)
    directly from the `SonarConfig`.
    """
    served_model_names = args.served_model_name or [args.model]
    model_registry = OpenAIModelRegistry(
        model_config=sonar_config.model_config,
        base_model_paths=[BaseModelPath(name=name, model_path=args.model) for name in served_model_names],
    )

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    renderer = renderer_from_config(sonar_config)
    resolved_chat_template = load_chat_template(args.chat_template)
    default_chat_template_kwargs = resolve_default_chat_template_kwargs(args)
    state.tool_server = await init_tool_server(args)

    # The config-resolved reasoning parser carries both the CLI flag (merged
    # in the entrypoint via `create_structured_outputs_config`) and any
    # model-specific default applied by `verify_and_update_config`
    # (e.g. "openai_gptoss" for gpt_oss), matching the main API server.
    # Keep `args.reasoning_parser` as the first source so callers that build
    # a SonarConfig without that merge still honor an explicit flag.
    reasoning_parser = args.reasoning_parser or sonar_config.structured_outputs_config.reasoning_parser

    state.online_renderer = OnlineRenderer(
        model_config=sonar_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        trust_request_mm_kwargs=args.trust_request_mm_kwargs,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        reasoning_effort_map=args.reasoning_effort_map,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    state.online_derenderer = OnlineDerenderer(
        model_config=sonar_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        response_role=args.response_role,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        reasoning_effort_map=args.reasoning_effort_map,
        log_error_stack=args.log_error_stack,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )

    state.openai_serving_models = model_registry
    state.serving_tokenization = ServingTokenization(
        model_registry,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    init_render_state(
        state,
        request_logger,
        disabled_thinking_effort=args.anthropic_disabled_thinking_effort,
    )

    state.sonar_config = sonar_config
    # Disable stats logging — there is no engine to poll.
    state.log_stats = False
    state.engine_client = None
    state.args = args
    state.enable_server_load_tracking = False
    state.server_load_metrics = 0

    # No `EngineClient` exists for the render server, so plugins get `None` and
    # must handle it themselves (see `EndpointPlugin.init_state`).
    await init_endpoint_plugins_state(None, state, args)
