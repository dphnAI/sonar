# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from aphrodite.config import AphroditeConfig
from aphrodite.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(aphrodite_config: AphroditeConfig) -> None:
    from aphrodite.model_executor.layers.rotary_embedding import _ROPE_DICT
    from aphrodite.v1.worker.workspace import reset_workspace_manager

    cache_config = aphrodite_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = aphrodite_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
