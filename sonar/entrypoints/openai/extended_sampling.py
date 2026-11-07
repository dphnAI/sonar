# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Aphrodite's extra sampler fields, shared by Completions and Chat Completions.

Keeping them in one base model means an upstream rewrite of either request
class cannot silently drop them from just one endpoint.
"""

import json
from typing import Any

from pydantic import Field, model_validator

from aphrodite.entrypoints.serve.engine.protocol import OpenAIBaseModel


def coerce_stringified_str_list(data: Any, field_name: str) -> Any:
    """Accept a JSON-encoded string list (as sent by some UIs) for a list field."""
    if not isinstance(data, dict):
        return data
    value = data.get(field_name)
    if not isinstance(value, str):
        return data
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return data
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        data[field_name] = parsed
    return data


class ExtendedSamplingRequest(OpenAIBaseModel):
    # --8<-- [start:extended-sampling-params]
    top_a: float | None = 0.0
    tfs: float | None = 1.0
    eta_cutoff: float | None = 0.0
    epsilon_cutoff: float | None = 0.0
    typical_p: float | None = 1.0
    smoothing_factor: float | None = 0.0
    smoothing_curve: float | None = 1.0
    no_repeat_ngram_size: int | None = 0
    temperature_last: bool | None = False
    xtc_threshold: float | None = 0.1
    xtc_probability: float | None = 0.0
    dry_multiplier: float | None = 0.0
    dry_base: float | None = 1.75
    dry_allowed_length: int | None = 2
    dry_sequence_breakers: list[str] | None = Field(default=None)
    dry_range: int | None = 0
    dry_max_ngram: int | None = 12
    dry_max_occurrences: int | None = 8
    dry_early_exit_match_len: int | None = 8
    dynatemp_min: float | None = 0.0
    dynatemp_max: float | None = 0.0
    dynatemp_exponent: float | None = 1.0
    nsigma: float | None = 0.0
    skew: float | None = 0.0
    mirostat_mode: int | None = 0
    mirostat_tau: float | None = 0.0
    mirostat_eta: float | None = 0.0
    # --8<-- [end:extended-sampling-params]

    @model_validator(mode="before")
    @classmethod
    def coerce_stringified_dry_sequence_breakers(cls, data: Any) -> Any:
        return coerce_stringified_str_list(data, "dry_sequence_breakers")

    def extended_sampling_kwargs(self) -> dict[str, Any]:
        """`SamplingParams.from_optional` keyword arguments for the extra samplers."""
        return {name: getattr(self, name) for name in ExtendedSamplingRequest.model_fields}
