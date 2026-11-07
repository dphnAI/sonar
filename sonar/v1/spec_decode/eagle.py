# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from sonar.config import SonarConfig
from sonar.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        sonar_config: SonarConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            sonar_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
