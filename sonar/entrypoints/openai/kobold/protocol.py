# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import Field, model_validator

from aphrodite.entrypoints.serve.engine.protocol import OpenAIBaseModel


class KAIGenerationInputSchema(OpenAIBaseModel):
    genkey: str | None = None
    prompt: str
    n: int | None = 1
    max_context_length: int
    max_length: int
    rep_pen: float | None = 1.0
    top_k: int | None = 0
    top_a: float | None = 0.0
    top_p: float | None = 1.0
    min_p: float | None = 0.0
    tfs: float | None = 1.0
    eps_cutoff: float | None = 0.0
    eta_cutoff: float | None = 0.0
    typical: float | None = 1.0
    temperature: float | None = 1.0
    dynatemp_range: float | None = 0.0
    dynatemp_exponent: float | None = 1.0
    smoothing_factor: float | None = 0.0
    smoothing_curve: float | None = 1.0
    xtc_threshold: float | None = 0.1
    xtc_probability: float | None = 0.0
    use_default_badwordsids: bool | None = None
    quiet: bool | None = None
    sampler_seed: int | None = None
    stop_sequence: list[str] | None = None
    include_stop_str_in_output: bool | None = False

    @model_validator(mode="before")
    @classmethod
    def check_context(cls, values: Any) -> Any:
        if isinstance(values, dict):
            max_length = values.get("max_length")
            max_context_length = values.get("max_context_length")
            if isinstance(max_length, int) and isinstance(max_context_length, int) and max_length > max_context_length:
                raise ValueError("max_length must not be larger than max_context_length")
        return values


class KAITokenizeRequest(OpenAIBaseModel):
    prompt: str = Field(min_length=0)
