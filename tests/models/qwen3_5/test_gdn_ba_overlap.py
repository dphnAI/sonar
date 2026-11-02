# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import Qwen3NextConfig

from aphrodite.config import set_current_aphrodite_config
from aphrodite.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from aphrodite.platforms import current_platform
from aphrodite.utils.network_utils import get_open_port
from tests.utils import init_test_distributed_environment
from tests.v1.attention.utils import create_aphrodite_config


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only overlap")
def test_ba_proj_overlaps_on_aux_stream_with_default_linear_backend() -> None:
    """A GEMM dispatch change must not silently disable the ba overlap."""
    aphrodite_config = create_aphrodite_config(dtype="bfloat16")
    with set_current_aphrodite_config(aphrodite_config):
        init_test_distributed_environment(1, 1, 0, str(get_open_port()))
        layer = QwenGatedDeltaNetAttention(
            Qwen3NextConfig(
                hidden_size=64,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
            ),
            aphrodite_config,
            prefix="model.layers.0.linear_attn",
            overlap_ba_proj=True,
        )
    assert layer._ba_stream is not None
