# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from fastapi import FastAPI

from sonar import envs
from sonar.config import ModelConfig
from sonar.tasks import POOLING_TASKS, SupportedTask


def register_api_routers(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    from sonar.entrypoints.serve import register_sonar_serve_api_routers

    register_sonar_serve_api_routers(app)

    from sonar.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    from sonar.entrypoints.generate.structured_decisions.api_router import (
        register_structured_decisions_api_router,
    )

    register_structured_decisions_api_router(app)

    from sonar.entrypoints.openai.decisions.api_router import (
        register_decisions_api_router,
    )

    register_decisions_api_router(app)

    from sonar.entrypoints.serve.sagemaker.api_router import (
        attach_router as register_sagemaker_api_router,
    )

    register_sagemaker_api_router(app, supported_tasks, model_config)

    if envs.SONAR_SERVER_DEV_MODE:
        from sonar.entrypoints.serve import register_sonar_dev_api_routers

        register_sonar_dev_api_routers(app)

        from sonar.entrypoints.rl.factories import register_rl_api_routers

        register_rl_api_routers(app)

    if "generate" in supported_tasks:
        from sonar.entrypoints.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

        from sonar.entrypoints.serve.elastic_ep.api_router import (
            attach_router as elastic_ep_attach_router,
        )

        elastic_ep_attach_router(app)

        if model_config is not None and model_config.enable_sleep_mode:
            from sonar.entrypoints.serve.checkpoint.api_router import (
                attach_router as checkpoint_attach_router,
            )

            checkpoint_attach_router(app)

    if "generate" in supported_tasks or "render" in supported_tasks:
        from sonar.entrypoints.scale_out.factories import register_scale_out_api_routers

        register_scale_out_api_routers(app, supported_tasks)

    if "transcription" in supported_tasks or "realtime" in supported_tasks:
        from sonar.entrypoints.speech_to_text.factories import (
            register_speech_to_text_api_routers,
        )

        register_speech_to_text_api_routers(app, supported_tasks)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from sonar.entrypoints.pooling.factories import register_pooling_api_routers

        register_pooling_api_routers(app, supported_tasks, model_config)

    if getattr(args, "enable_fault_tolerance", False):
        from sonar.entrypoints.serve.fault_tolerance.api_router import (
            register_fault_tolerance_api_router,
        )

        register_fault_tolerance_api_router(app)
