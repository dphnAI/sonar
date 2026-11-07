# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonar's ordered samplers for V2, isolated from the native sampling fast path.

Committed history is read from RequestState, not an optimistic CPU output mirror.
Draft prefixes are local to a verification call. Only accepted outputs commit
Mirostat updates; its checkpoints survive preemption and roll back on readmission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import numpy as np
import torch

from sonar.sampling_params import SamplerID, SamplingParams
from sonar.v1.sample.logits_processor import LogitsProcessors
from sonar.v1.sample.metadata import SamplingMetadata
from sonar.v1.sample.ops import SamplingOps
from sonar.v1.sample.ops.temperatures import apply_all_temperatures
from sonar.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

if TYPE_CHECKING:
    # Annotation-only: the standalone CPU runner reuses this module without
    # importing GPU worker state.
    from sonar.v1.worker.gpu.states import RequestState


class ForkRequestStates(Protocol):
    """Request-state surface shared by the GPU and standalone CPU runners."""

    @property
    def device(self) -> torch.device: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def index_to_req_id(self) -> dict[int, str]: ...

    @property
    def req_id_to_index(self) -> dict[str, int]: ...


class ForkLogitsContext(Protocol):
    """Logits-context fields read by the ordered samplers."""

    @property
    def expanded_idx_mapping(self) -> torch.Tensor: ...

    @property
    def pos(self) -> torch.Tensor: ...

    @property
    def input_ids(self) -> torch.Tensor: ...


StatesT = TypeVar("StatesT", bound=ForkRequestStates)

DEFAULT_ORDER = (
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
)


def needs_fork_sampling(params: SamplingParams) -> bool:
    return bool(
        params.dry_multiplier
        or params.no_repeat_ngram_size
        or params.dynatemp_min
        or params.dynatemp_max
        or params.temperature_last
        or params.top_a
        or params.tfs != 1.0
        or params.eta_cutoff
        or params.epsilon_cutoff
        or params.typical_p != 1.0
        or params.smoothing_factor
        or params.xtc_probability
        or params.nsigma
        or params.mirostat_mode
        or params.skew
    )


@dataclass
class ForkRequest:
    params: SamplingParams
    metadata: SamplingMetadata
    order: tuple[SamplerID, ...]
    seed: int
    generator: torch.Generator | None = None
    # mu before sampling the token at this output offset.
    checkpoints: dict[int, float] = field(default_factory=dict)
    # (output offset, surprise before truncation, provisional mu).
    pending: list[tuple[int, torch.Tensor, float]] = field(default_factory=list)


class ForkSamplingState(ABC, Generic[StatesT]):
    def __init__(self, req_states: StatesT):
        self.req_states = req_states
        self.requests: dict[int, ForkRequest] = {}
        self.retained: dict[str, ForkRequest] = {}
        self.ops = SamplingOps()

    def any(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(self.requests) and any(int(idx) in self.requests for idx in idx_mapping_np)

    @abstractmethod
    def _prompt_len(self, req_idx: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _prefill_len(self, req_idx: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _token_history(self, req_idx: int, end: int) -> list[int]:
        """Committed token ids [0, end) of a request slot."""
        raise NotImplementedError

    def add_request(self, req_idx: int, params: SamplingParams, seed: int) -> bool:
        self.requests.pop(req_idx, None)
        if not needs_fork_sampling(params):
            return False
        req_id = self.req_states.index_to_req_id[req_idx]
        output_len = self._prefill_len(req_idx) - self._prompt_len(req_idx)
        previous = self.retained.pop(req_id, None)
        device = self.req_states.device

        def tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.tensor([value], dtype=dtype, device=device)

        metadata = SamplingMetadata(
            temperature=tensor(params.temperature),
            dynatemp_min=tensor(params.dynatemp_min),
            dynatemp_max=tensor(params.dynatemp_max),
            dynatemp_exp=tensor(params.dynatemp_exponent),
            all_greedy=params.temperature == 0,
            all_random=params.temperature != 0,
            top_p=tensor(params.top_p),
            top_k=tensor(
                min(params.top_k, self.req_states.vocab_size) if params.top_k > 0 else self.req_states.vocab_size,
                torch.int32,
            ),
            generators={},
            max_num_logprobs=None,
            no_penalties=(
                params.repetition_penalty == 1 and params.presence_penalty == 0 and params.frequency_penalty == 0
            ),
            prompt_token_ids=None,
            frequency_penalties=tensor(params.frequency_penalty),
            presence_penalties=tensor(params.presence_penalty),
            repetition_penalties=tensor(params.repetition_penalty),
            output_token_ids=[[]],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
            min_p=tensor(params.min_p),
            persistent_data={} if previous is None else previous.metadata.persistent_data,
        )
        fields = {
            "top_a": params.top_a,
            "tfs": params.tfs,
            "eta_cutoff": params.eta_cutoff,
            "epsilon_cutoff": params.epsilon_cutoff,
            "typical_p": params.typical_p,
            "quadratic_smoothing_factor": params.smoothing_factor,
            "quadratic_smoothing_curve": params.smoothing_curve,
            "xtc_threshold": params.xtc_threshold,
            "xtc_probability": params.xtc_probability,
            "top_nsigma": params.nsigma,
            "skew": params.skew,
            "dry_multiplier": params.dry_multiplier,
            "dry_base": params.dry_base,
        }
        for name, value in fields.items():
            setattr(metadata, name, tensor(value))
        for name, value in {
            "no_repeat_ngram_size": params.no_repeat_ngram_size,
            "dry_allowed_length": params.dry_allowed_length,
            "dry_ranges": params.dry_range,
            "dry_max_ngram": params.dry_max_ngram,
            "dry_max_occurrences": params.dry_max_occurrences,
            "dry_early_exit_match_len": params.dry_early_exit_match_len,
        }.items():
            setattr(metadata, name, tensor(value, torch.int64))
        if params.dry_multiplier:
            metadata.dry_multiplier_cpu = torch.tensor([params.dry_multiplier], dtype=torch.float32)
            for name, value in {
                "dry_allowed_length": params.dry_allowed_length,
                "dry_ranges": params.dry_range,
                "dry_max_ngram": params.dry_max_ngram,
                "dry_max_occurrences": params.dry_max_occurrences,
                "dry_early_exit_match_len": params.dry_early_exit_match_len,
            }.items():
                setattr(metadata, name + "_cpu", torch.tensor([value], dtype=torch.int32))
            metadata.dry_sequence_breaker_ids_cpu = torch.tensor([params.dry_sequence_breaker_ids], dtype=torch.int64)
        metadata.dry_sequence_breaker_ids = torch.tensor(
            [params.dry_sequence_breaker_ids], dtype=torch.int64, device=device
        )
        order = list(DEFAULT_ORDER)
        if params.temperature_last:
            order.remove(SamplerID.TEMPERATURE)
            order.append(SamplerID.TEMPERATURE)
        checkpoints = {0: 2 * params.mirostat_tau} if previous is None else previous.checkpoints
        checkpoints = {offset: mu for offset, mu in checkpoints.items() if offset <= output_len}
        checkpoints.setdefault(output_len, 2 * params.mirostat_tau)
        self.requests[req_idx] = ForkRequest(
            params,
            metadata,
            tuple(order),
            seed=seed if previous is None else previous.seed,
            generator=torch.Generator(device=device) if params.xtc_probability else None,
            checkpoints=checkpoints,
        )
        return True

    def remove_request(self, req_idx: int) -> None:
        state = self.requests.pop(req_idx, None)
        if state is not None:
            state.pending.clear()
            self.retained[self.req_states.index_to_req_id[req_idx]] = state

    def finish_request(self, req_id: str) -> None:
        self.retained.pop(req_id, None)
        req_idx = self.req_states.req_id_to_index.get(req_id)
        if req_idx is not None:
            self.requests.pop(req_idx, None)

    def apply(self, logits: torch.Tensor, ctx: ForkLogitsContext) -> torch.Tensor:
        slots = ctx.expanded_idx_mapping.tolist()
        positions = ctx.pos.tolist()
        inputs = ctx.input_ids.tolist()
        histories: dict[int, list[int]] = {}
        previous_surprises: dict[int, torch.Tensor] = {}
        provisional_mu: dict[int, float] = {}
        for row, (slot, position, token) in enumerate(zip(slots, positions, inputs)):
            state = self.requests[slot]
            params, metadata = state.params, state.metadata
            prompt_len = self._prompt_len(slot)
            if slot not in histories:
                state.pending.clear()
                histories[slot] = self._token_history(slot, position + 1)
            else:
                histories[slot].append(token)
            history = histories[slot]
            output_offset = position + 1 - prompt_len
            metadata.prompt_token_ids = torch.tensor([history[:prompt_len]], dtype=torch.int64, device=logits.device)
            metadata.output_token_ids = [history[prompt_len:]]
            if params.dry_multiplier:
                # Feed the stable-ABI native scanner the exact committed/draft
                # prefix; no optimistic or rejected token can leak into DRY.
                metadata.token_history_ids_cpu = torch.tensor([history], dtype=torch.int32)
                metadata.token_history_lens_cpu = torch.tensor([len(history)], dtype=torch.int32)
            row_logits = logits[row : row + 1]
            if params.mirostat_mode == 2:
                if slot in previous_surprises:
                    surprise = float(previous_surprises[slot][token].item())
                    mu = provisional_mu[slot] - params.mirostat_eta * (surprise - params.mirostat_tau)
                else:
                    mu = state.checkpoints.get(output_offset, 2 * params.mirostat_tau)
                # Keep a distribution, not legacy Mirostat's pre-sampled one-hot:
                # probabilistic rejection needs the actual target probabilities.
                surprise = -torch.log_softmax(row_logits[0], dim=-1) / np.log(2.0)
                mask = surprise > mu
                mask[surprise.argmin()] = False
                row_logits.masked_fill_(mask.unsqueeze(0), -float("inf"))
                state.pending.append((output_offset, surprise, mu))
                previous_surprises[slot] = surprise
                provisional_mu[slot] = mu
            else:
                row_logits = self._apply_order(row_logits, state, position)
            logits[row : row + 1] = row_logits
        return logits

    def _apply_order(self, logits: torch.Tensor, state: ForkRequest, position: int) -> torch.Tensor:
        params, metadata = state.params, state.metadata
        operations = {
            SamplerID.DRY: (params.dry_multiplier != 0, self.ops.apply_dry),
            SamplerID.NO_REPEAT_NGRAM: (params.no_repeat_ngram_size != 0, self.ops.apply_no_repeat_ngram),
            SamplerID.TOP_NSIGMA: (params.nsigma != 0, self.ops.apply_top_nsigma),
            SamplerID.TOP_A: (params.top_a != 0, self.ops.apply_top_a),
            SamplerID.MIN_P: (params.min_p != 0, self.ops.apply_min_p),
            SamplerID.TFS: (params.tfs != 1, self.ops.apply_tfs),
            SamplerID.ETA_CUTOFF: (params.eta_cutoff != 0, self.ops.apply_eta_cutoff),
            SamplerID.EPSILON_CUTOFF: (params.epsilon_cutoff != 0, self.ops.apply_epsilon_cutoff),
            SamplerID.TYPICAL_P: (params.typical_p != 1, self.ops.apply_typical_p),
            SamplerID.QUADRATIC: (params.smoothing_factor != 0, self.ops.apply_quadratic),
        }
        for stage in state.order:
            if stage == SamplerID.TEMPERATURE:
                if params.temperature != 0:
                    logits = apply_all_temperatures(logits, metadata)
            elif stage == SamplerID.PENALTIES:
                if not metadata.no_penalties:
                    logits = self.ops.apply_penalties(logits, metadata, metadata.output_token_ids)
            elif stage == SamplerID.TOP_P_TOP_K:
                if params.temperature != 0 and (params.top_k > 0 or params.top_p < 1):
                    logits = apply_top_k_top_p(logits, metadata.top_k, metadata.top_p)
            elif stage == SamplerID.XTC:
                if params.xtc_probability:
                    assert state.generator is not None
                    # Position-keyed RNG makes retries, chunking and batch ordering independent.
                    state.generator.manual_seed((state.seed + position * 0x9E3779B1) % (2**63 - 1))
                    if torch.rand((), device=logits.device, generator=state.generator) < params.xtc_probability:
                        probs, indices = logits.softmax(-1).sort(descending=True, dim=-1)
                        count = int((probs[0, 1:] >= params.xtc_threshold).sum().item())
                        logits[0, indices[0, :count]] = -float("inf")
            else:
                enabled, operation = operations[stage]
                if enabled:
                    logits = operation(logits, metadata)
        if params.skew:
            # The legacy skew operation returns probabilities, not logits.
            support = torch.isfinite(logits)
            probabilities = self.ops.apply_skew(logits, metadata).clamp_min_(0)
            logits = probabilities.masked_fill_(~support, 0).log_()
        return logits

    def update(self, input_batch, sampler_output) -> None:
        """Commit accepted Mirostat outputs and synchronize TP sampler owners.

        Ownership can change when a preempted request is assigned a new slot.
        Every TP rank therefore retains the same accepted mu checkpoints even
        when only its sampling shard saw the target logits.
        """
        mirostat_requests = [
            (batch_idx, self.requests[int(slot)])
            for batch_idx, slot in enumerate(input_batch.idx_mapping_np)
            if int(slot) in self.requests and self.requests[int(slot)].params.mirostat_mode == 2
        ]
        if not mirostat_requests:
            return
        width = sampler_output.sampled_token_ids.shape[1]
        updates = torch.zeros((len(mirostat_requests), width, 3), dtype=torch.float64, device=self.req_states.device)
        for row, (batch_idx, state) in enumerate(mirostat_requests):
            if state.pending:
                count = int(sampler_output.num_sampled[batch_idx].item())
                tokens = sampler_output.sampled_token_ids[batch_idx, :count].tolist()
                for col, (token, (offset, surprise, mu)) in enumerate(zip(tokens, state.pending)):
                    next_mu = mu - state.params.mirostat_eta * (
                        float(surprise[token].item()) - state.params.mirostat_tau
                    )
                    updates[row, col] = updates.new_tensor((offset + 1, next_mu, 1))
            state.pending.clear()
        if torch.distributed.is_initialized():
            from sonar.distributed.parallel_state import get_tp_group

            group = get_tp_group()
            if group.world_size > 1:
                torch.distributed.all_reduce(updates, group=group.device_group)
        for (_, state), entries in zip(mirostat_requests, updates.tolist()):
            for offset_sum, mu_sum, contributors in entries:
                if contributors:
                    state.checkpoints[round(offset_sum / contributors)] = mu_sum / contributors


class GPUForkSamplingState(ForkSamplingState):
    """Ordered samplers over the GPU runner's UVA-backed RequestState."""

    req_states: RequestState

    def _prompt_len(self, req_idx: int) -> int:
        return int(self.req_states.prompt_len.np[req_idx])

    def _prefill_len(self, req_idx: int) -> int:
        return int(self.req_states.prefill_len.np[req_idx])

    def _token_history(self, req_idx: int, end: int) -> list[int]:
        return self.req_states.all_token_ids.gpu[req_idx, :end].tolist()
