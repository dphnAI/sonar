# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from aphrodite.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    ServeContext,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
    resolve_token_id_placeholder,
)
from aphrodite.entrypoints.openai.engine.protocol import GenerationError

__all__ = [
    "GenerateBaseServing",
    "GenerationError",
    "OpenAIServing",
    "ServeContext",
    "clamp_prompt_logprobs",
    "format_token_id_placeholder",
    "resolve_token_id_placeholder",
]


class OpenAIServing(GenerateBaseServing):
    pass
