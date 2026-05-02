# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from aphrodite.v1.sample.logits_processor import LogitsProcessors
from aphrodite.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


@dataclass
class SamplingMetadata:
    # Temperature
    temperature: torch.Tensor | None
    dynatemp_min: torch.Tensor | None
    dynatemp_max: torch.Tensor | None
    dynatemp_exp: torch.Tensor | None

    all_greedy: bool
    all_random: bool

    # Alphabet sampling
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    # min_p is done in the logits processor
    # min_p: Optional[torch.Tensor]
    top_a: torch.Tensor | None

    # DRY
    dry_multiplier: torch.Tensor | None
    dry_base: torch.Tensor | None
    dry_allowed_length: torch.Tensor | None
    dry_sequence_breaker_ids: torch.Tensor | None
    dry_ranges: torch.Tensor | None
    dry_max_ngram: torch.Tensor | None
    dry_max_occurrences: torch.Tensor | None
    dry_early_exit_match_len: torch.Tensor | None

    # No repeat ngram
    no_repeat_ngram_size: torch.Tensor | None

    # Tail-Free Sampling
    tfs: torch.Tensor | None

    # Eta Cutoff
    eta_cutoff: torch.Tensor | None

    # Epsilon Cutoff
    epsilon_cutoff: torch.Tensor | None

    # Typical Sampling
    typical_p: torch.Tensor | None

    # Quadratic Sampling
    quadratic_smoothing_factor: torch.Tensor | None
    quadratic_smoothing_curve: torch.Tensor | None

    # XTC Sampling
    xtc_threshold: torch.Tensor | None
    xtc_probability: torch.Tensor | None

    # Top-nsigma Sampling
    top_nsigma: torch.Tensor | None

    # Mirostat Sampling
    mirostat_mode: torch.Tensor | None
    mirostat_tau: torch.Tensor | None
    mirostat_eta: torch.Tensor | None

    # Skew
    skew: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    logit_bias: dict[int, dict[int, float]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    logprob_token_ids: dict[int, list[int]] | None = None

    # Request-level temperature-last execution flag.
    temperature_last: list[bool] | None = None

    # Persistent metadata for mirostat
    persistent_data: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies
    # thinking-token-budget logits (holder may exist with an empty tracking set).
    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None

    # Cached padded token-history tensor for GPU-side sampler ops.
    output_token_ids_tensor: torch.Tensor | None = None
    token_history_ids: torch.Tensor | None = None
    token_history_lens: torch.Tensor | None = None
    token_history_ids_cpu: torch.Tensor | None = None
    token_history_lens_cpu: torch.Tensor | None = None
    dry_multiplier_cpu: torch.Tensor | None = None
    dry_allowed_length_cpu: torch.Tensor | None = None
    dry_sequence_breaker_ids_cpu: torch.Tensor | None = None
    dry_ranges_cpu: torch.Tensor | None = None
    dry_max_ngram_cpu: torch.Tensor | None = None
    dry_max_occurrences_cpu: torch.Tensor | None = None
    dry_early_exit_match_len_cpu: torch.Tensor | None = None
