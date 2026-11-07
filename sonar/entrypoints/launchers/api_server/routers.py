# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from fastapi import FastAPI

from aphrodite import envs
from aphrodite.config import ModelConfig
from aphrodite.tasks import POOLING_TASKS, SupportedTask


def register_api_routers(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    from aphrodite.entrypoints.serve import register_aphrodite_serve_api_routers

    register_aphrodite_serve_api_routers(app)

    from aphrodite.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    from aphrodite.entrypoints.generate.structured_decisions.api_router import (
        register_structured_decisions_api_router,
    )

    register_structured_decisions_api_router(app)

    from aphrodite.entrypoints.openai.decisions.api_router import (
        register_decisions_api_router,
    )

    register_decisions_api_router(app)

    from aphrodite.entrypoints.serve.sagemaker.api_router import (
        attach_router as register_sagemaker_api_router,
    )

    register_sagemaker_api_router(app, supported_tasks, model_config)

    if envs.APHRODITE_SERVER_DEV_MODE:
        from aphrodite.entrypoints.serve import register_aphrodite_dev_api_routers

        register_aphrodite_dev_api_routers(app)

        from aphrodite.entrypoints.rl.factories import register_rl_api_routers

        register_rl_api_routers(app)

    if "generate" in supported_tasks:
        from aphrodite.entrypoints.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

        from aphrodite.entrypoints.serve.elastic_ep.api_router import (
            attach_router as elastic_ep_attach_router,
        )

        elastic_ep_attach_router(app)

        if model_config is not None and model_config.enable_sleep_mode:
            from aphrodite.entrypoints.serve.checkpoint.api_router import (
                attach_router as checkpoint_attach_router,
            )

            checkpoint_attach_router(app)

    if "generate" in supported_tasks or "render" in supported_tasks:
        from aphrodite.entrypoints.scale_out.factories import register_scale_out_api_routers

        register_scale_out_api_routers(app, supported_tasks)

    if "transcription" in supported_tasks or "realtime" in supported_tasks:
        from aphrodite.entrypoints.speech_to_text.factories import (
            register_speech_to_text_api_routers,
        )

        register_speech_to_text_api_routers(app, supported_tasks)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from aphrodite.entrypoints.pooling.factories import register_pooling_api_routers

        register_pooling_api_routers(app, supported_tasks, model_config)

    if getattr(args, "enable_fault_tolerance", False):
        from aphrodite.entrypoints.serve.fault_tolerance.api_router import (
            register_fault_tolerance_api_router,
        )

        register_fault_tolerance_api_router(app)
