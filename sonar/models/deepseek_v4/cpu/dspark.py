# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DeepSeek-V4 DSpark (speculative decoding) — deferred, not implemented.

``DSparkDeepseekV4ForCausalLM`` is only instantiated when the speculative-
decoding registry resolves it by name for a DSpark speculative config; a
plain (non-spec) DeepSeek-V4 CPU model never constructs this class. This
stub exists solely so ``aphrodite.models.deepseek_v4.cpu`` is import-clean.
"""

from aphrodite.config import AphroditeConfig


class DSparkDeepseekV4ForCausalLM:
    def __init__(self, *, aphrodite_config: AphroditeConfig, prefix: str = "") -> None:
        raise NotImplementedError("DeepSeek-V4 speculative decoding (DSpark) is not implemented on CPU.")
