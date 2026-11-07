# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.config import AphroditeConfig
from aphrodite.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        aphrodite_config: AphroditeConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            aphrodite_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
