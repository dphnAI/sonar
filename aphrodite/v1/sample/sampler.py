# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

from dataclasses import replace

import torch
import torch.nn as nn

from aphrodite.config.model import LogprobsMode
from aphrodite.logger import init_logger
from aphrodite.sampling_params import SamplerID
from aphrodite.utils.platform_utils import is_pin_memory_available
from aphrodite.v1.outputs import LogprobsTensors, SamplerOutput
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.sample.ops import SamplingOps
from aphrodite.v1.sample.ops.bad_words import apply_bad_words
from aphrodite.v1.sample.ops.logprobs import batched_count_greater_than
from aphrodite.v1.sample.ops.temperatures import apply_all_temperatures
from aphrodite.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

logger = init_logger(__name__)
_SAMPLING_EPS = 1e-5

# Default sampler execution order (same as V0)
DEFAULT_SAMPLER_ORDER = [
    SamplerID.DRY,
    SamplerID.PENALTIES,
    SamplerID.NO_REPEAT_NGRAM,
    SamplerID.TEMPERATURE,
    SamplerID.TOP_NSIGMA,
    SamplerID.TOP_P_TOP_K,
    SamplerID.TOP_A,
    SamplerID.MIN_P,
    SamplerID.TFS,
    SamplerID.ETA_CUTOFF,
    SamplerID.EPSILON_CUTOFF,
    SamplerID.TYPICAL_P,
    SamplerID.QUADRATIC,
    SamplerID.XTC,
]


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:
    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.sampling_ops = SamplingOps()
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE: Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.sampling_ops.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)

        # Apply samplers in priority order
        logits = self._execute_samplers_in_order(logits, sampling_metadata)

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(torch.empty(0), raw_logprobs, torch.empty(0))
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return apply_all_temperatures(logits, sampling_metadata)

    def _execute_samplers_in_order(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Execute samplers in the specified priority order.

        Args:
            logits: Input logits tensor
            sampling_metadata: Sampling metadata containing priority information

        Returns:
            Modified logits tensor after applying samplers in priority order
        """
        # Check if mirostat is active - if so, disable other samplers
        has_mirostat = False
        if (
            sampling_metadata.mirostat_mode is not None
            and sampling_metadata.mirostat_tau is not None
            and sampling_metadata.mirostat_eta is not None
        ):
            batch_size = len(sampling_metadata.output_token_ids)
            has_mirostat = any(sampling_metadata.mirostat_mode[i].item() == 2 for i in range(batch_size))

        if has_mirostat:
            # Mirostat is active - only apply mirostat and skip other samplers
            logger.debug("Mirostat active - applying mirostat only")
            logits = self.sampling_ops.apply_mirostat(logits, sampling_metadata)
            return logits

        temperature_last_flags = sampling_metadata.temperature_last
        if not temperature_last_flags:
            return self._apply_sampler_order(logits, sampling_metadata, do_temperature_last=False)

        if all(temperature_last_flags):
            return self._apply_sampler_order(logits, sampling_metadata, do_temperature_last=True)

        if not any(temperature_last_flags):
            return self._apply_sampler_order(logits, sampling_metadata, do_temperature_last=False)

        temp_last_indices = [i for i, enabled in enumerate(temperature_last_flags) if enabled]
        default_indices = [i for i, enabled in enumerate(temperature_last_flags) if not enabled]

        if default_indices:
            default_metadata = self._subset_sampling_metadata(sampling_metadata, default_indices)
            default_logits = self._apply_sampler_order(logits[default_indices], default_metadata, False)
            logits[default_indices] = default_logits

        if temp_last_indices:
            temp_last_metadata = self._subset_sampling_metadata(sampling_metadata, temp_last_indices)
            temp_last_logits = self._apply_sampler_order(logits[temp_last_indices], temp_last_metadata, True)
            logits[temp_last_indices] = temp_last_logits

        return logits

    def _apply_sampler_order(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        do_temperature_last: bool,
    ) -> torch.Tensor:
        sampler_order = []
        for sampler_id in DEFAULT_SAMPLER_ORDER:
            if sampler_id == SamplerID.TEMPERATURE and do_temperature_last:
                continue
            sampler_order.append(sampler_id)
            if sampler_id == SamplerID.XTC and do_temperature_last:
                sampler_order.append(SamplerID.TEMPERATURE)

        # Log the execution order for debugging
        logger.debug("Sampler execution order: ")
        for i, sampler_id in enumerate(sampler_order, 1):
            logger.debug("%d. %s", i, sampler_id.name)

        # Execute samplers in the specified order
        for sampler_id in sampler_order:
            if sampler_id == SamplerID.DRY and sampling_metadata.dry_multiplier is not None:
                logger.debug("Applying DRY with dry_multiplier: %s", sampling_metadata.dry_multiplier)
                logits = self.sampling_ops.apply_dry(logits, sampling_metadata)

            elif sampler_id == SamplerID.PENALTIES and not sampling_metadata.no_penalties:
                logger.debug("Applying penalties")
                logits = self.sampling_ops.apply_penalties(
                    logits, sampling_metadata, sampling_metadata.output_token_ids
                )

            elif sampler_id == SamplerID.NO_REPEAT_NGRAM and sampling_metadata.no_repeat_ngram_size is not None:
                logger.debug("Applying no_repeat_ngram with size: %s", sampling_metadata.no_repeat_ngram_size)
                logits = self.sampling_ops.apply_no_repeat_ngram(logits, sampling_metadata)

            elif sampler_id == SamplerID.TEMPERATURE and sampling_metadata.temperature is not None:
                logger.debug("Applying temperature: %s", sampling_metadata.temperature)
                logits = self.apply_temperature(logits, sampling_metadata)

            elif sampler_id == SamplerID.TOP_NSIGMA and sampling_metadata.top_nsigma is not None:
                logger.debug("Applying Top-Nsigma with nsigma: %s", sampling_metadata.top_nsigma)
                logits = self.sampling_ops.apply_top_nsigma(logits, sampling_metadata)

            elif sampler_id == SamplerID.TOP_P_TOP_K:
                # Defer top-k/top-p to TopKTopPSampler so CUDA/FlashInfer/native
                # fast paths can sample without materializing full-vocab masks.
                pass

            elif sampler_id == SamplerID.TOP_A and sampling_metadata.top_a is not None:
                logger.debug("Applying Top-a with top_a: %s", sampling_metadata.top_a)
                logits = self.sampling_ops.apply_top_a(logits, sampling_metadata)

            elif sampler_id == SamplerID.TFS and sampling_metadata.tfs is not None:
                logger.debug("Applying TFS with tfs: %s", sampling_metadata.tfs)
                logits = self.sampling_ops.apply_tfs(logits, sampling_metadata)

            elif sampler_id == SamplerID.ETA_CUTOFF and sampling_metadata.eta_cutoff is not None:
                logger.debug("Applying ETA Cutoff with eta_cutoff: %s", sampling_metadata.eta_cutoff)
                logits = self.sampling_ops.apply_eta_cutoff(logits, sampling_metadata)

            elif sampler_id == SamplerID.EPSILON_CUTOFF and sampling_metadata.epsilon_cutoff is not None:
                logger.debug("Applying Epsilon Cutoff with epsilon_cutoff: %s", sampling_metadata.epsilon_cutoff)
                logits = self.sampling_ops.apply_epsilon_cutoff(logits, sampling_metadata)

            elif sampler_id == SamplerID.TYPICAL_P and sampling_metadata.typical_p is not None:
                logger.debug("Applying Typical P with typical_p: %s", sampling_metadata.typical_p)
                logits = self.sampling_ops.apply_typical_p(logits, sampling_metadata)

            elif sampler_id == SamplerID.QUADRATIC and sampling_metadata.quadratic_smoothing_factor is not None:
                logger.debug(
                    "Applying Quadratic with smoothing_factor: %s",
                    sampling_metadata.quadratic_smoothing_factor,
                )
                logits = self.sampling_ops.apply_quadratic(logits, sampling_metadata)

            elif sampler_id == SamplerID.XTC and sampling_metadata.xtc_threshold is not None:
                logger.debug("Applying XTC with threshold: %s", sampling_metadata.xtc_threshold)
                logits = self.sampling_ops.apply_xtc(logits, sampling_metadata)

        return logits

    @staticmethod
    def _subset_sampling_metadata(
        sampling_metadata: SamplingMetadata,
        indices: list[int],
    ) -> SamplingMetadata:
        index_tensor = torch.tensor(
            indices, device=sampling_metadata.temperature.device if sampling_metadata.temperature is not None else None
        )

        def maybe_index_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.index_select(0, index_tensor.to(tensor.device))

        def reindex_dict_list(
            mapping: dict[int, list[int]] | None,
        ) -> dict[int, list[int]] | None:
            if mapping is None:
                return None
            return {new_i: mapping[old_i] for new_i, old_i in enumerate(indices) if old_i in mapping}

        def reindex_dict_nested_list(
            mapping: dict[int, list[list[int]]],
        ) -> dict[int, list[list[int]]]:
            return {new_i: mapping[old_i] for new_i, old_i in enumerate(indices) if old_i in mapping}

        def reindex_dict_float_map(
            mapping: dict[int, dict[int, float]],
        ) -> dict[int, dict[int, float]]:
            return {new_i: mapping[old_i] for new_i, old_i in enumerate(indices) if old_i in mapping}

        return replace(
            sampling_metadata,
            temperature=maybe_index_tensor(sampling_metadata.temperature),
            dynatemp_min=maybe_index_tensor(sampling_metadata.dynatemp_min),
            dynatemp_max=maybe_index_tensor(sampling_metadata.dynatemp_max),
            dynatemp_exp=maybe_index_tensor(sampling_metadata.dynatemp_exp),
            top_p=maybe_index_tensor(sampling_metadata.top_p),
            top_k=maybe_index_tensor(sampling_metadata.top_k),
            top_a=maybe_index_tensor(sampling_metadata.top_a),
            dry_multiplier=maybe_index_tensor(sampling_metadata.dry_multiplier),
            dry_base=maybe_index_tensor(sampling_metadata.dry_base),
            dry_allowed_length=maybe_index_tensor(sampling_metadata.dry_allowed_length),
            dry_sequence_breaker_ids=maybe_index_tensor(sampling_metadata.dry_sequence_breaker_ids),
            dry_ranges=maybe_index_tensor(sampling_metadata.dry_ranges),
            dry_max_ngram=maybe_index_tensor(sampling_metadata.dry_max_ngram),
            dry_max_occurrences=maybe_index_tensor(sampling_metadata.dry_max_occurrences),
            dry_early_exit_match_len=maybe_index_tensor(sampling_metadata.dry_early_exit_match_len),
            no_repeat_ngram_size=maybe_index_tensor(sampling_metadata.no_repeat_ngram_size),
            tfs=maybe_index_tensor(sampling_metadata.tfs),
            eta_cutoff=maybe_index_tensor(sampling_metadata.eta_cutoff),
            epsilon_cutoff=maybe_index_tensor(sampling_metadata.epsilon_cutoff),
            typical_p=maybe_index_tensor(sampling_metadata.typical_p),
            quadratic_smoothing_factor=maybe_index_tensor(sampling_metadata.quadratic_smoothing_factor),
            quadratic_smoothing_curve=maybe_index_tensor(sampling_metadata.quadratic_smoothing_curve),
            xtc_threshold=maybe_index_tensor(sampling_metadata.xtc_threshold),
            xtc_probability=maybe_index_tensor(sampling_metadata.xtc_probability),
            top_nsigma=maybe_index_tensor(sampling_metadata.top_nsigma),
            mirostat_mode=maybe_index_tensor(sampling_metadata.mirostat_mode),
            mirostat_tau=maybe_index_tensor(sampling_metadata.mirostat_tau),
            mirostat_eta=maybe_index_tensor(sampling_metadata.mirostat_eta),
            skew=maybe_index_tensor(sampling_metadata.skew),
            prompt_token_ids=maybe_index_tensor(sampling_metadata.prompt_token_ids),
            frequency_penalties=maybe_index_tensor(sampling_metadata.frequency_penalties),
            presence_penalties=maybe_index_tensor(sampling_metadata.presence_penalties),
            repetition_penalties=maybe_index_tensor(sampling_metadata.repetition_penalties),
            output_token_ids=[sampling_metadata.output_token_ids[i] for i in indices],
            allowed_token_ids_mask=maybe_index_tensor(sampling_metadata.allowed_token_ids_mask),
            bad_words_token_ids=reindex_dict_nested_list(sampling_metadata.bad_words_token_ids),
            logit_bias=reindex_dict_float_map(sampling_metadata.logit_bias),
            logprob_token_ids=reindex_dict_list(sampling_metadata.logprob_token_ids),
            temperature_last=[sampling_metadata.temperature_last[i] for i in indices]
            if sampling_metadata.temperature_last is not None
            else None,
            persistent_data={
                new_i: sampling_metadata.persistent_data.get(old_i, {}).copy() for new_i, old_i in enumerate(indices)
            },
            spec_token_ids=(
                [
                    sampling_metadata.spec_token_ids[i] if i < len(sampling_metadata.spec_token_ids) else []
                    for i in indices
                ]
                if sampling_metadata.spec_token_ids is not None
                else None
            ),
            output_token_ids_tensor=maybe_index_tensor(sampling_metadata.output_token_ids_tensor),
            token_history_ids=maybe_index_tensor(sampling_metadata.token_history_ids),
            token_history_lens=maybe_index_tensor(sampling_metadata.token_history_lens),
            token_history_ids_cpu=maybe_index_tensor(sampling_metadata.token_history_ids_cpu),
            token_history_lens_cpu=maybe_index_tensor(sampling_metadata.token_history_lens_cpu),
            dry_multiplier_cpu=maybe_index_tensor(sampling_metadata.dry_multiplier_cpu),
            dry_allowed_length_cpu=maybe_index_tensor(sampling_metadata.dry_allowed_length_cpu),
            dry_sequence_breaker_ids_cpu=maybe_index_tensor(sampling_metadata.dry_sequence_breaker_ids_cpu),
            dry_ranges_cpu=maybe_index_tensor(sampling_metadata.dry_ranges_cpu),
            dry_max_ngram_cpu=maybe_index_tensor(sampling_metadata.dry_max_ngram_cpu),
            dry_max_occurrences_cpu=maybe_index_tensor(sampling_metadata.dry_max_occurrences_cpu),
            dry_early_exit_match_len_cpu=maybe_index_tensor(sampling_metadata.dry_early_exit_match_len_cpu),
            generators={
                new_i: sampling_metadata.generators[old_i]
                for new_i, old_i in enumerate(indices)
                if old_i in sampling_metadata.generators
            },
        )

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply skew
        if sampling_metadata.skew is not None:
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            probs = self.sampling_ops.apply_skew(probs, sampling_metadata)
            logits = torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))

        # Apply sampling (multinomial sampling from the processed logits)
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        return [[*out, *spec] if spec else out for out, spec in zip(output_token_ids, spec_token_ids)]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking_combine = holder is not None and holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and (any_penalties_or_bad_words or needs_thinking_combine):
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        if holder is not None and holder.has_tracked_requests():
            holder.update_state(
                output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        return logits
