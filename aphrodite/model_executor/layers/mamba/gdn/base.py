# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PretrainedConfig

from aphrodite.config import (
    AphroditeConfig,
)
from aphrodite.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aphrodite.model_executor.custom_op import PluggableLayer
from aphrodite.model_executor.layers.mamba.abstract import MambaBase
from aphrodite.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from aphrodite.model_executor.models.utils import extract_layer_index
from aphrodite.v1.attention.backends.registry import MambaAttentionBackendEnum


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = aphrodite_config.model_config
        self.cache_config = aphrodite_config.cache_config
        self.quant_config = aphrodite_config.quant_config
        self.speculative_config = aphrodite_config.speculative_config
        self.num_spec = self.speculative_config.num_speculative_tokens if self.speculative_config else 0

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
