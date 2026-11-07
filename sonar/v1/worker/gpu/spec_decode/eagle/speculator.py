# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from aphrodite.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from aphrodite.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class EagleSpeculator(TargetDependentARSpeculator):
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.aphrodite_config)
