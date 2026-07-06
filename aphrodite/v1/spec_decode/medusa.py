# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from aphrodite.config import AphroditeConfig
from aphrodite.forward_context import set_forward_context
from aphrodite.logger import init_logger
from aphrodite.model_executor.model_loader import get_model
from aphrodite.model_executor.models.interfaces import is_mixture_of_experts
from aphrodite.v1.sample.metadata import SamplingMetadata

# Initialize logger
logger = init_logger(__name__)


class MedusaProposer:
    """
    Medusa proposer class for generating token sequences
    """

    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        device: torch.device,
    ):
        # Save config parameters
        self.aphrodite_config = aphrodite_config
        assert aphrodite_config.speculative_config is not None, "Speculative config must be set"
        self.spec_config = aphrodite_config.speculative_config
        self.device = device
        self.max_num_tokens = aphrodite_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = self.spec_config.draft_model_config.get_hidden_size()
        self.dtype = aphrodite_config.model_config.dtype
        self.num_speculative_tokens = self.spec_config.num_speculative_tokens

    def propose(
        self,
        num_speculative_tokens: int,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,  # unused
    ) -> torch.Tensor:
        assert num_speculative_tokens == self.num_speculative_tokens
        # Generate blocks and compute logits
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)

        # Compute argmax for each Medusa head and stack into a single tensor
        # Shape: [batch_size, num_heads]
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)

        return draft_tokens

    def load_model(self, target_model: nn.Module) -> None:
        from aphrodite.compilation.backends import set_model_tag

        with set_model_tag("medusa_head"):
            self.model = get_model(
                aphrodite_config=self.aphrodite_config,
                model_config=self.spec_config.draft_model_config,
            )
        assert not (is_mixture_of_experts(self.model) and self.aphrodite_config.parallel_config.enable_eplb), (
            "EPLB for Medusa is not supported"
        )

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_forward_context(None, self.aphrodite_config, num_tokens=num_tokens):
            self.model(hidden_states)
