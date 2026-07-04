# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from aphrodite.config import AphroditeConfig


def init_speculator(aphrodite_config: AphroditeConfig, device: torch.device):
    speculative_config = aphrodite_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        from aphrodite.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(aphrodite_config, device)
    elif speculative_config.method == "dspark":
        from aphrodite.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(aphrodite_config, device)
    elif speculative_config.use_gemma4_mtp():
        from aphrodite.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(aphrodite_config, device)
    elif speculative_config.method == "mtp":
        from aphrodite.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(aphrodite_config, device)
    elif speculative_config.use_eagle():
        from aphrodite.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(aphrodite_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
