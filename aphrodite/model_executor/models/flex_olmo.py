# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only FlexOlmo model compatible with HuggingFace weights."""

import torch
from torch import nn

from aphrodite.config import AphroditeConfig
from aphrodite.distributed import get_tensor_model_parallel_world_size
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.fused_moe import (
    FusedMoE,
)
from aphrodite.model_executor.layers.layernorm import RMSNorm
from aphrodite.model_executor.layers.linear import ReplicatedLinear
from aphrodite.model_executor.models.olmoe import OlmoeAttention, OlmoeForCausalLM
from aphrodite.transformers_utils.configs.flex_olmo import FlexOlmoConfig

logger = init_logger(__name__)


class FlexOlmoAttention(OlmoeAttention):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix)

        hf_config = aphrodite_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, eps=hf_config.rms_norm_eps)
        self.q_norm = RMSNorm(self.total_num_heads * self.head_dim, eps=hf_config.rms_norm_eps)


class FlexOlmoMoE(nn.Module):
    """A tensor-parallel MoE implementation for FlexOlmo that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = ""):
        super().__init__()

        hf_config = aphrodite_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        tp_size = get_tensor_model_parallel_world_size()

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hf_config.hidden_size,
            hf_config.num_experts,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            num_experts=hf_config.num_experts,
            top_k=hf_config.num_experts_per_tok,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            renormalize=False,
            quant_config=None,
            tp_size=tp_size,
            prefix=f"{prefix}.experts",
            router_logits_dtype=torch.float32,
        )

        self.top_k = hf_config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        # Warning: The experts mutate the hidden state input! This messes up
        # basic things like the residual stream.
        final_hidden_states = self.experts(
            hidden_states=hidden_states.detach().clone(),
            router_logits=router_logits.float(),
        )

        return final_hidden_states.view(orig_shape)


class FlexOlmoDecoderLayer(nn.Module):
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = aphrodite_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.self_attn = FlexOlmoAttention(aphrodite_config=aphrodite_config, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

        self.mlp = FlexOlmoMoE(aphrodite_config=aphrodite_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Attention block.
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, None


class FlexOlmoForCausalLM(OlmoeForCausalLM):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        *,
        aphrodite_config: AphroditeConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = FlexOlmoDecoderLayer,
    ):
        super().__init__(aphrodite_config=aphrodite_config, prefix=prefix, layer_type=layer_type)
