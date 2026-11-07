# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DeepSeek-V4 MTP (speculative decoding) — deferred, not implemented.

``DeepSeekV4MTP`` is only instantiated when
``sonar/model_executor/models/registry.py``'s speculative-decoding registry
resolves it by name for a speculative-decoding config; a plain (non-spec)
DeepSeek-V4 CPU model never constructs this class. This stub exists solely
so ``sonar.models.deepseek_v4.cpu`` is import-clean.
"""

from sonar.config import SonarConfig


class DeepSeekV4MTP:
    def __init__(self, *, sonar_config: SonarConfig, prefix: str = "") -> None:
        raise NotImplementedError("DeepSeek-V4 speculative decoding (MTP) is not implemented on CPU.")
