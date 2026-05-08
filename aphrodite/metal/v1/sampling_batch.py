# SPDX-License-Identifier: Apache-2.0
"""Internal sampling batch ownership and token sampling for Metal v1.

Pure functions: logits in, token IDs out.  No model runner state accessed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import torch

from aphrodite.metal.pytorch_backend.tensor_bridge import mlx_to_torch
from aphrodite.sampling_params import SamplingParams
from aphrodite.utils.torch_utils import make_tensor_with_pad
from aphrodite.v1.outputs import LogprobsTensors
from aphrodite.v1.sample.logits_processor import LogitsProcessors
from aphrodite.v1.sample.logits_processor.interface import BatchUpdate
from aphrodite.v1.sample.metadata import SamplingMetadata
from aphrodite.v1.sample.sampler import Sampler

GREEDY_TEMPERATURE_EPS = 1e-5


@dataclass
class MetalSamplerResult:
    token_ids: list[int]
    logprobs_tensors: LogprobsTensors | None


class SamplingBatch:
    """Sampling-side batch owner for ``MetalModelRunner``.

    This is an interim extraction that keeps sampling policy and
    ``SamplingMetadata`` construction out of ``model_runner.py`` while the
    runner is being slimmed down.

    Today it owns only the sampling-side state for one step. As more per-step
    batch state moves out of ``model_runner.py``, this should evolve into a
    fuller ``MetalInputBatch``-style owner that can absorb request indexing,
    token views, generators, logits processor ownership, and metadata refresh.
    """

    def __init__(
        self,
        sampling_params_list: Sequence[SamplingParams],
        prompt_token_id_lists: Sequence[list[int]],
        output_token_id_lists: Sequence[list[int]],
        *,
        vocab_size: int,
        device: torch.device,
        logitsprocs: LogitsProcessors | None = None,
        generators: dict[int, torch.Generator] | None = None,
    ) -> None:
        batch_size = len(sampling_params_list)
        if len(prompt_token_id_lists) != batch_size:
            raise ValueError(
                "Expected prompt token ids for each request in the batch "
                f"(len(prompt_token_id_lists)={len(prompt_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )
        if len(output_token_id_lists) != batch_size:
            raise ValueError(
                "Expected output token ids for each request in the batch "
                f"(len(output_token_id_lists)={len(output_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )

        self.sampling_params_list = list(sampling_params_list)
        self.prompt_token_id_lists = list(prompt_token_id_lists)
        self.output_token_id_lists = list(output_token_id_lists)
        self.vocab_size = vocab_size
        self.device = device
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.generators = {} if generators is None else generators
        self.all_greedy = all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS for sampling_params in self.sampling_params_list
        )
        self.all_random = not self.all_greedy and all(
            sampling_params.temperature >= GREEDY_TEMPERATURE_EPS for sampling_params in self.sampling_params_list
        )
        self.no_top_p = all(sampling_params.top_p == 1.0 for sampling_params in self.sampling_params_list)
        self.no_top_k = all(
            not (0 < sampling_params.top_k < self.vocab_size) for sampling_params in self.sampling_params_list
        )
        self.no_dynatemp = all(
            sampling_params.dynatemp_min <= 0.0 and sampling_params.dynatemp_max <= 0.0
            for sampling_params in self.sampling_params_list
        )
        self.no_top_a = all(sampling_params.top_a <= 0.0 for sampling_params in self.sampling_params_list)
        self.no_dry = all(sampling_params.dry_multiplier <= 0.0 for sampling_params in self.sampling_params_list)
        self.no_no_repeat_ngram = all(
            sampling_params.no_repeat_ngram_size <= 0 for sampling_params in self.sampling_params_list
        )
        self.no_tfs = all(sampling_params.tfs >= 1.0 for sampling_params in self.sampling_params_list)
        self.no_eta_cutoff = all(sampling_params.eta_cutoff <= 0.0 for sampling_params in self.sampling_params_list)
        self.no_epsilon_cutoff = all(
            sampling_params.epsilon_cutoff <= 0.0 for sampling_params in self.sampling_params_list
        )
        self.no_typical_p = all(sampling_params.typical_p >= 1.0 for sampling_params in self.sampling_params_list)
        self.no_quadratic = all(
            sampling_params.smoothing_factor <= 0.0 for sampling_params in self.sampling_params_list
        )
        self.no_xtc = all(sampling_params.xtc_probability <= 0.0 for sampling_params in self.sampling_params_list)
        self.no_top_nsigma = all(sampling_params.nsigma <= 0.0 for sampling_params in self.sampling_params_list)
        self.no_mirostat = all(sampling_params.mirostat_mode == 0 for sampling_params in self.sampling_params_list)
        self.no_skew = all(sampling_params.skew == 0.0 for sampling_params in self.sampling_params_list)
        self.no_allowed_token_ids = all(
            not sampling_params.allowed_token_ids for sampling_params in self.sampling_params_list
        )
        self.no_bad_words = all(
            not sampling_params.bad_words_token_ids for sampling_params in self.sampling_params_list
        )
        self.no_logit_bias = all(not sampling_params.logit_bias for sampling_params in self.sampling_params_list)
        self.no_logprob_token_ids = all(
            sampling_params.logprob_token_ids is None for sampling_params in self.sampling_params_list
        )
        self.no_penalties = all(
            sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            for sampling_params in self.sampling_params_list
        )

    @staticmethod
    def can_use_native_greedy(
        sampling_params_list: Sequence[SamplingParams],
        *,
        vocab_size: int | None = None,
    ) -> bool:
        """Return whether MLX argmax matches the requested sampling behavior."""
        return all(
            sampling_params.temperature < GREEDY_TEMPERATURE_EPS
            and (sampling_params.top_k <= 0 or (vocab_size is not None and sampling_params.top_k >= vocab_size))
            and sampling_params.top_p == 1.0
            and sampling_params.min_p == 0.0
            and sampling_params.top_a == 0.0
            and sampling_params.tfs == 1.0
            and sampling_params.eta_cutoff == 0.0
            and sampling_params.epsilon_cutoff == 0.0
            and sampling_params.typical_p == 1.0
            and sampling_params.smoothing_factor == 0.0
            and sampling_params.xtc_probability == 0.0
            and sampling_params.nsigma == 0.0
            and sampling_params.mirostat_mode == 0
            and sampling_params.skew == 0.0
            and sampling_params.dynatemp_min == 0.0
            and sampling_params.dynatemp_max == 0.0
            and sampling_params.dry_multiplier == 0.0
            and sampling_params.no_repeat_ngram_size == 0
            and sampling_params.frequency_penalty == 0.0
            and sampling_params.presence_penalty == 0.0
            and sampling_params.repetition_penalty == 1.0
            and sampling_params.min_tokens == 0
            and not sampling_params.allowed_token_ids
            and not sampling_params.bad_words_token_ids
            and not sampling_params.logit_bias
            and sampling_params.logprobs is None
            and sampling_params.logprob_token_ids is None
            for sampling_params in sampling_params_list
        )

    def _make_temperature(self) -> torch.Tensor | None:
        if self.all_greedy:
            return None

        return torch.tensor(
            [sampling_params.temperature for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )

    def _make_top_p(self) -> torch.Tensor | None:
        if self.no_top_p:
            return None

        return torch.tensor(
            [sampling_params.top_p for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )

    def _make_top_k(self) -> torch.Tensor | None:
        if self.no_top_k:
            return None

        return torch.tensor(
            [
                sampling_params.top_k if 0 < sampling_params.top_k < self.vocab_size else self.vocab_size
                for sampling_params in self.sampling_params_list
            ],
            dtype=torch.int32,
            device=self.device,
        )

    def _make_float_tensor(
        self,
        attr: str,
        *,
        disabled: bool,
        source_attr: str | None = None,
    ) -> torch.Tensor | None:
        if disabled:
            return None
        field_name = source_attr or attr
        return torch.tensor(
            [float(getattr(sampling_params, field_name)) for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )

    def _make_int_tensor(
        self,
        attr: str,
        *,
        disabled: bool,
        dtype: torch.dtype = torch.int32,
        source_attr: str | None = None,
    ) -> torch.Tensor | None:
        if disabled:
            return None
        field_name = source_attr or attr
        return torch.tensor(
            [int(getattr(sampling_params, field_name)) for sampling_params in self.sampling_params_list],
            dtype=dtype,
            device=self.device,
        )

    def _make_prompt_token_ids(self) -> torch.Tensor | None:
        if self.no_penalties and self.no_dry:
            return None

        return make_tensor_with_pad(
            self.prompt_token_id_lists,
            pad=self.vocab_size,
            device=self.device,
            dtype=torch.int64,
            pin_memory=False,
        )

    def _make_penalty_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frequency_penalties = torch.tensor(
            [sampling_params.frequency_penalty for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )
        presence_penalties = torch.tensor(
            [sampling_params.presence_penalty for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )
        repetition_penalties = torch.tensor(
            [sampling_params.repetition_penalty for sampling_params in self.sampling_params_list],
            dtype=torch.float32,
            device=self.device,
        )
        return frequency_penalties, presence_penalties, repetition_penalties

    def _make_dry_sequence_breaker_ids(self) -> torch.Tensor | None:
        if self.no_dry:
            return None

        max_len = max(
            (
                len(sampling_params.dry_sequence_breaker_ids)
                for sampling_params in self.sampling_params_list
                if sampling_params.dry_sequence_breaker_ids
            ),
            default=0,
        )
        if max_len == 0:
            return torch.empty(
                (len(self.sampling_params_list), 0),
                dtype=torch.long,
                device=self.device,
            )

        rows = []
        for sampling_params in self.sampling_params_list:
            breaker_ids = sampling_params.dry_sequence_breaker_ids or []
            rows.append(breaker_ids + [self.vocab_size] * (max_len - len(breaker_ids)))
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _make_allowed_token_ids_mask(self) -> torch.Tensor | None:
        if self.no_allowed_token_ids:
            return None

        mask = torch.zeros(
            (len(self.sampling_params_list), self.vocab_size),
            dtype=torch.bool,
            device=self.device,
        )
        for index, sampling_params in enumerate(self.sampling_params_list):
            if sampling_params.allowed_token_ids:
                mask[index].fill_(True)
                mask[index, sampling_params.allowed_token_ids] = False
        return mask

    def _make_bad_words_token_ids(self) -> dict[int, list[list[int]]]:
        return {
            index: sampling_params.bad_words_token_ids
            for index, sampling_params in enumerate(self.sampling_params_list)
            if sampling_params.bad_words_token_ids
        }

    def _make_logit_bias(self) -> dict[int, dict[int, float]]:
        return {
            index: sampling_params.logit_bias
            for index, sampling_params in enumerate(self.sampling_params_list)
            if sampling_params.logit_bias
        }

    def _make_logprob_token_ids(self) -> dict[int, list[int]] | None:
        if self.no_logprob_token_ids:
            return None
        return {
            index: sampling_params.logprob_token_ids
            for index, sampling_params in enumerate(self.sampling_params_list)
            if sampling_params.logprob_token_ids is not None
        }

    def _make_max_num_logprobs(self) -> int | None:
        values = [
            self.vocab_size if sampling_params.logprobs == -1 else sampling_params.logprobs
            for sampling_params in self.sampling_params_list
            if sampling_params.logprobs is not None
        ]
        return max(values) if values else None

    def _refresh_logits_processors(self) -> None:
        previous_batch_size = int(getattr(self.logitsprocs, "_metal_batch_size", 0))
        batch_size = len(self.sampling_params_list)
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=list(range(batch_size, previous_batch_size)),
            added=[
                (
                    index,
                    sampling_params,
                    self.prompt_token_id_lists[index],
                    self.output_token_id_lists[index],
                )
                for index, sampling_params in enumerate(self.sampling_params_list)
            ],
            moved=[],
        )
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        setattr(self.logitsprocs, "_metal_batch_size", batch_size)

    def can_use_native_greedy_for_batch(self) -> bool:
        return self.can_use_native_greedy(
            self.sampling_params_list,
            vocab_size=self.vocab_size,
        )

    def can_use_native_random_for_batch(self) -> bool:
        """Return whether MLX can handle this random-sampling request.

        This fast path intentionally covers the common serving case:
        temperature + top-k/top-p sampling without request-specific logits
        processors, penalties, constraints, or logprob output.  Everything
        else stays on Aphrodite's torch sampler for feature parity.
        """
        if not self.all_random:
            return False
        if not (
            self.no_dynatemp
            and self.no_top_a
            and self.no_dry
            and self.no_no_repeat_ngram
            and self.no_tfs
            and self.no_eta_cutoff
            and self.no_epsilon_cutoff
            and self.no_typical_p
            and self.no_quadratic
            and self.no_xtc
            and self.no_top_nsigma
            and self.no_mirostat
            and self.no_skew
            and self.no_allowed_token_ids
            and self.no_bad_words
            and self.no_logit_bias
            and self.no_logprob_token_ids
            and self.no_penalties
        ):
            return False
        return all(
            sampling_params.min_p == 0.0
            and sampling_params.min_tokens == 0
            and sampling_params.logprobs is None
            and sampling_params.logprob_token_ids is None
            and sampling_params.seed is None
            and not sampling_params.temperature_last
            for sampling_params in self.sampling_params_list
        )

    def make_sampling_metadata(self) -> SamplingMetadata:
        """Create Aphrodite ``SamplingMetadata`` for this batch."""
        self._refresh_logits_processors()
        (
            frequency_penalties,
            presence_penalties,
            repetition_penalties,
        ) = self._make_penalty_tensors()

        return SamplingMetadata(
            temperature=self._make_temperature(),
            dynatemp_min=self._make_float_tensor("dynatemp_min", disabled=self.no_dynatemp),
            dynatemp_max=self._make_float_tensor("dynatemp_max", disabled=self.no_dynatemp),
            dynatemp_exp=self._make_float_tensor(
                "dynatemp_exp",
                disabled=self.no_dynatemp,
                source_attr="dynatemp_exponent",
            ),
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self._make_top_p(),
            top_k=self._make_top_k(),
            top_a=self._make_float_tensor("top_a", disabled=self.no_top_a),
            dry_multiplier=self._make_float_tensor("dry_multiplier", disabled=self.no_dry),
            dry_base=self._make_float_tensor("dry_base", disabled=self.no_dry),
            dry_allowed_length=self._make_int_tensor("dry_allowed_length", disabled=self.no_dry),
            dry_sequence_breaker_ids=self._make_dry_sequence_breaker_ids(),
            dry_ranges=self._make_int_tensor(
                "dry_ranges",
                disabled=self.no_dry,
                source_attr="dry_range",
            ),
            dry_max_ngram=self._make_int_tensor("dry_max_ngram", disabled=self.no_dry),
            dry_max_occurrences=self._make_int_tensor("dry_max_occurrences", disabled=self.no_dry),
            dry_early_exit_match_len=self._make_int_tensor("dry_early_exit_match_len", disabled=self.no_dry),
            no_repeat_ngram_size=self._make_int_tensor("no_repeat_ngram_size", disabled=self.no_no_repeat_ngram),
            tfs=self._make_float_tensor("tfs", disabled=self.no_tfs),
            eta_cutoff=self._make_float_tensor("eta_cutoff", disabled=self.no_eta_cutoff),
            epsilon_cutoff=self._make_float_tensor("epsilon_cutoff", disabled=self.no_epsilon_cutoff),
            typical_p=self._make_float_tensor("typical_p", disabled=self.no_typical_p),
            quadratic_smoothing_factor=self._make_float_tensor(
                "quadratic_smoothing_factor",
                disabled=self.no_quadratic,
                source_attr="smoothing_factor",
            ),
            quadratic_smoothing_curve=self._make_float_tensor(
                "quadratic_smoothing_curve",
                disabled=self.no_quadratic,
                source_attr="smoothing_curve",
            ),
            xtc_threshold=self._make_float_tensor("xtc_threshold", disabled=self.no_xtc),
            xtc_probability=self._make_float_tensor("xtc_probability", disabled=self.no_xtc),
            top_nsigma=self._make_float_tensor(
                "top_nsigma",
                disabled=self.no_top_nsigma,
                source_attr="nsigma",
            ),
            mirostat_mode=self._make_int_tensor("mirostat_mode", disabled=self.no_mirostat),
            mirostat_tau=self._make_float_tensor("mirostat_tau", disabled=self.no_mirostat),
            mirostat_eta=self._make_float_tensor("mirostat_eta", disabled=self.no_mirostat),
            skew=self._make_float_tensor("skew", disabled=self.no_skew),
            generators=self.generators,
            max_num_logprobs=self._make_max_num_logprobs(),
            prompt_token_ids=self._make_prompt_token_ids(),
            output_token_ids=self.output_token_id_lists,
            frequency_penalties=frequency_penalties,
            presence_penalties=presence_penalties,
            repetition_penalties=repetition_penalties,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=self._make_allowed_token_ids_mask(),
            bad_words_token_ids=self._make_bad_words_token_ids(),
            logit_bias=self._make_logit_bias(),
            logitsprocs=self.logitsprocs,
            logprob_token_ids=self._make_logprob_token_ids(),
            temperature_last=[sampling_params.temperature_last for sampling_params in self.sampling_params_list],
            persistent_data={index: {} for index in range(len(self.sampling_params_list))},
            spec_token_ids=None,
        )


# ---------------------------------------------------------------------------
# Pure sampling functions
# ---------------------------------------------------------------------------


def _mlx_greedy_sample(logits: mx.array) -> mx.array:
    """Native MLX greedy sampling — avoids PyTorch round-trip."""
    return mx.argmax(logits, axis=-1)


def _mlx_random_sample(logits: mx.array, batch: SamplingBatch) -> mx.array:
    """Native MLX temperature/top-k/top-p sampling for the common path."""
    logits = logits.astype(mx.float32)
    temperatures = mx.array(
        [sampling_params.temperature for sampling_params in batch.sampling_params_list],
        dtype=mx.float32,
    )[:, None]
    logits = logits / temperatures

    if not batch.no_top_k:
        top_ks = [
            sampling_params.top_k if 0 < sampling_params.top_k < batch.vocab_size else batch.vocab_size
            for sampling_params in batch.sampling_params_list
        ]
        max_top_k = max(top_ks)
        if max_top_k < batch.vocab_size:
            topk_indices = mx.argpartition(-logits, max_top_k - 1, axis=-1)[:, :max_top_k]
            logits = mx.take_along_axis(logits, topk_indices, axis=-1)
            if len(set(top_ks)) != 1:
                positions = mx.arange(max_top_k)[None, :]
                row_top_ks = mx.array(top_ks, dtype=mx.int32)[:, None]
                logits = mx.where(positions < row_top_ks, logits, -float("inf"))

            if not batch.no_top_p:
                sorted_positions = mx.argsort(-logits, axis=-1)
                sorted_logits = mx.take_along_axis(logits, sorted_positions, axis=-1)
                sorted_indices = mx.take_along_axis(topk_indices, sorted_positions, axis=-1)
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                top_ps = mx.array(
                    [sampling_params.top_p for sampling_params in batch.sampling_params_list],
                    dtype=mx.float32,
                )[:, None]
                # Keep the first token that crosses top-p, matching nucleus
                # sampling's usual "cumulative probability before this token"
                # test.
                remove = (mx.cumsum(sorted_probs, axis=-1) - sorted_probs) > top_ps
                sorted_logits = mx.where(remove, -float("inf"), sorted_logits)
                sampled_positions = mx.random.categorical(sorted_logits, axis=-1)
                return mx.take_along_axis(sorted_indices, sampled_positions[:, None], axis=-1)[:, 0]

            sampled_positions = mx.random.categorical(logits, axis=-1)
            return mx.take_along_axis(topk_indices, sampled_positions[:, None], axis=-1)[:, 0]

        topk_values = mx.topk(logits, max_top_k, axis=-1)
        topk_thresholds = mx.min(topk_values, axis=-1, keepdims=True)
        logits = mx.where(logits < topk_thresholds, -float("inf"), logits)

    if not batch.no_top_p:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        top_ps = mx.array(
            [sampling_params.top_p for sampling_params in batch.sampling_params_list],
            dtype=mx.float32,
        )[:, None]
        # Keep the first token that crosses top-p, matching nucleus sampling's
        # usual "cumulative probability before this token" test.
        remove = (mx.cumsum(sorted_probs, axis=-1) - sorted_probs) > top_ps
        sorted_logits = mx.where(remove, -float("inf"), sorted_logits)
        sampled_positions = mx.random.categorical(sorted_logits, axis=-1)
        return mx.take_along_axis(sorted_indices, sampled_positions[:, None], axis=-1)[:, 0]

    return mx.random.categorical(logits, axis=-1)


def sample_from_logits(
    logits_2d: mx.array,
    batch: SamplingBatch,
    sampler: Sampler,
    device: torch.device,
) -> MetalSamplerResult:
    """Sample tokens from pre-sliced 2D logits ``(batch_size, vocab)``.

    Single entry point for all sampling paths.  Chooses native MLX greedy
    when possible, otherwise bridges to the Aphrodite torch sampler.
    """
    if batch.can_use_native_greedy_for_batch():
        tokens = _mlx_greedy_sample(logits_2d)
        mx.eval(tokens)
        if tokens.ndim == 0:
            return MetalSamplerResult([int(tokens.item())], None)
        return MetalSamplerResult(tokens.tolist(), None)  # type: ignore[arg-type]

    if batch.can_use_native_random_for_batch():
        tokens = _mlx_random_sample(logits_2d, batch)
        mx.eval(tokens)
        if tokens.ndim == 0:
            return MetalSamplerResult([int(tokens.item())], None)
        return MetalSamplerResult(tokens.tolist(), None)  # type: ignore[arg-type]

    mx.eval(logits_2d)
    logits_torch = mlx_to_torch(logits_2d.astype(mx.float32), device=device)
    metadata = batch.make_sampling_metadata()
    output = sampler.forward(logits_torch, metadata)
    return MetalSamplerResult(
        output.sampled_token_ids[:, 0].tolist(),
        output.logprobs_tensors,
    )


def sample_decode_tokens(
    logits: mx.array,
    decode_reqs: list[tuple[str, object]],
    num_decode: int,
    sampler: Sampler,
    device: torch.device,
    *,
    vocab_size: int,
    logitsprocs: LogitsProcessors | None = None,
) -> MetalSamplerResult:
    """Sample one token per decode request from evaluated logits.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        decode_reqs: ``(req_id, RequestState)`` pairs for decode requests.
        num_decode: Number of decode requests (prefix of the token dimension).
        sampler: Aphrodite Sampler instance.
        device: PyTorch device for the torch bridge path.
        vocab_size: Model vocabulary size.
        logitsprocs: Optional logits processors.

    Returns:
        List of sampled token IDs, one per decode request.
    """
    if not decode_reqs:
        return MetalSamplerResult([], None)

    decode_logits = logits[0, :num_decode, :]  # (num_decode, vocab)

    sampling_params_list = [state.sampling_params for _, state in decode_reqs]
    prompt_token_ids_list = [state.token_ids[: state.prompt_len] for _, state in decode_reqs]
    output_tokens_list = [state.token_ids[state.prompt_len :] for _, state in decode_reqs]
    generators = {i: state.generator for i, (_, state) in enumerate(decode_reqs) if state.generator is not None}

    batch = SamplingBatch(
        sampling_params_list,
        prompt_token_ids_list,
        output_tokens_list,
        vocab_size=vocab_size,
        device=device,
        logitsprocs=logitsprocs,
        generators=generators,
    )
    return sample_from_logits(decode_logits, batch, sampler, device)


def sample_prefill_tokens(
    logits: mx.array,
    prefill_reqs: list,
    cu_seqlens: list[int],
    num_decode: int,
    sampler: Sampler,
    device: torch.device,
    *,
    vocab_size: int,
    logitsprocs: LogitsProcessors | None = None,
) -> MetalSamplerResult:
    """Sample one token per prefill request from the last logit position.

    Args:
        logits: Full logits array, shape ``(1, total_tokens, vocab)``.
        prefill_reqs: List of ``PrefillRequest`` objects.
        cu_seqlens: Cumulative sequence lengths for logit position lookup.
        num_decode: Number of decode requests (offset into cu_seqlens).
        sampler: Aphrodite Sampler instance.
        device: PyTorch device for the torch bridge path.
        vocab_size: Model vocabulary size.
        logitsprocs: Optional logits processors.

    Returns:
        List of sampled token IDs, one per prefill request.
    """
    prefill_next_tokens: list[int] = []
    prefill_logprobs: list[LogprobsTensors | None] = []
    for j, pr in enumerate(prefill_reqs):
        last_idx = cu_seqlens[num_decode + j + 1] - 1
        last_logits = logits[0, last_idx : last_idx + 1, :]  # (1, vocab)

        if pr.full_prompt_token_ids is not None:
            prompt_len = len(pr.full_prompt_token_ids)
        elif pr.prompt_len is not None:
            prompt_len = pr.prompt_len
        else:
            prompt_len = len(pr.token_ids)

        prompt_for_meta = pr.full_prompt_token_ids if pr.full_prompt_token_ids is not None else pr.token_ids
        generators = {} if pr.generator is None else {0: pr.generator}

        batch = SamplingBatch(
            [pr.sampling_params],
            [prompt_for_meta[:prompt_len]],
            [prompt_for_meta[prompt_len:]],
            vocab_size=vocab_size,
            device=device,
            logitsprocs=logitsprocs,
            generators=generators,
        )
        result = sample_from_logits(last_logits, batch, sampler, device)
        [next_token] = result.token_ids
        prefill_next_tokens.append(next_token)
        prefill_logprobs.append(result.logprobs_tensors)

    logprobs_tensors = _merge_single_row_logprobs(prefill_logprobs)
    return MetalSamplerResult(prefill_next_tokens, logprobs_tensors)


def _merge_single_row_logprobs(
    rows: Sequence[LogprobsTensors | None],
) -> LogprobsTensors | None:
    non_none = [row for row in rows if row is not None]
    if not non_none:
        return None

    width = max(row.logprobs.shape[1] for row in non_none)
    token_rows: list[torch.Tensor] = []
    logprob_rows: list[torch.Tensor] = []
    rank_rows: list[torch.Tensor] = []

    token_dtype = non_none[0].logprob_token_ids.dtype
    logprob_dtype = non_none[0].logprobs.dtype
    rank_dtype = non_none[0].selected_token_ranks.dtype
    device = non_none[0].logprobs.device
    for row in rows:
        if row is None:
            token_rows.append(torch.zeros((1, width), dtype=token_dtype, device=device))
            logprob_rows.append(torch.zeros((1, width), dtype=logprob_dtype, device=device))
            rank_rows.append(torch.zeros((1,), dtype=rank_dtype, device=device))
            continue
        pad = width - row.logprobs.shape[1]
        if pad:
            token_rows.append(torch.nn.functional.pad(row.logprob_token_ids[:1], (0, pad)))
            logprob_rows.append(torch.nn.functional.pad(row.logprobs[:1], (0, pad)))
        else:
            token_rows.append(row.logprob_token_ids[:1])
            logprob_rows.append(row.logprobs[:1])
        rank_rows.append(row.selected_token_ranks[:1])

    return LogprobsTensors(
        torch.cat(token_rows, dim=0),
        torch.cat(logprob_rows, dim=0),
        torch.cat(rank_rows, dim=0),
    )
