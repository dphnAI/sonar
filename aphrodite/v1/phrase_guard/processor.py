# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.sampling_params import SamplingParams
from aphrodite.v1.phrase_guard.matcher import validate_phrases
from aphrodite.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from aphrodite.v1.sample.logits_processor.builtin import process_dict_updates

RETRY_KEY = "_sonar_phrase_retry"


class PhraseRetryProcessor(LogitsProcessor):
    """Mask failed alternatives at their checkpoint, never at another position."""

    def __init__(self, aphrodite_config, device, is_pin_memory):
        self.device = device
        self.states: dict[int, tuple[list[int], int, torch.Tensor]] = {}

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        args = sampling_params.extra_args or {}
        if "banned_strings" in args:
            validate_phrases(args["banned_strings"])
            case_sensitive = args.get("banned_strings_case_sensitive", False)
            if not isinstance(case_sensitive, (bool, int)) or case_sensitive not in (0, 1):
                raise ValueError("banned_strings_case_sensitive must be a boolean or 0/1")
            if sampling_params.stop:
                raise ValueError("Experimental banned_strings does not support text stop strings; use stop_token_ids")
            if (
                sampling_params.structured_outputs is not None
                or sampling_params.trace_decode_token_ids is not None
                or sampling_params.mirostat_mode
            ):
                raise ValueError("Experimental banned_strings does not support grammar, replay, or Mirostat")
        if RETRY_KEY in args:
            raise ValueError(f"{RETRY_KEY} is reserved for the phrase scheduler")

    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def validate_runtime(cls, config, tokenizer):
        from transformers import TokenizersBackend

        from aphrodite.v1.engine import detokenizer

        if not detokenizer.USE_FAST_DETOKENIZER or not isinstance(tokenizer, TokenizersBackend):
            raise ValueError("Experimental banned_strings requires a fast Hugging Face tokenizer")
        if config.model_config.return_sampling_mask or config.model_config.enable_return_routed_experts:
            raise ValueError("Experimental banned_strings does not support sampling masks or routed experts")
        if not config.use_v2_model_runner and not any(
            p is cls or p == f"{cls.__module__}:{cls.__name__}" for p in config.model_config.logits_processors or []
        ):
            raise ValueError("banned_strings requires the PhraseRetryProcessor logits processor")

    @staticmethod
    def validate_input(resumable, has_encoder_inputs, prompt_embeds):
        if resumable or has_encoder_inputs or prompt_embeds is not None:
            raise ValueError("Experimental banned_strings requires non-resumable plain text generation")

    def update_state(self, batch_update: BatchUpdate | None):
        def add(params, prompt, output):
            retry = (params.extra_args or {}).get(RETRY_KEY)
            if retry is None:
                return None
            position, tokens = retry
            return output, position, torch.tensor(tokens, dtype=torch.long, device=self.device)

        process_dict_updates(self.states, batch_update, add)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row, (output, position, tokens) in self.states.items():
            if len(output) == position:
                logits[row].index_fill_(0, tokens, float("-inf"))
                # Return a blocked token as an error sentinel if constraints
                # exhaust the row. The scheduler discards it before decoding.
                first = tokens[:1]
                logits[row].scatter_(0, first, torch.where(logits[row].amax() == -torch.inf, 0.0, logits[row][first]))
        return logits
