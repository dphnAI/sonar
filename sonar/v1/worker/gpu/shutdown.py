# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from sonar.config import SonarConfig
from sonar.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(sonar_config: SonarConfig) -> None:
    from sonar.model_executor.layers.rotary_embedding import _ROPE_DICT
    from sonar.v1.worker.workspace import reset_workspace_manager

    cache_config = sonar_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = sonar_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
