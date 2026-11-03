# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sonar's ordered samplers and phrase-retry masks on the standalone CPU runner."""

import numpy as np
import pytest
import torch

from aphrodite.config import AphroditeConfig
from aphrodite.platforms import CpuArchEnum, current_platform
from aphrodite.sampling_params import SamplingParams
from aphrodite.triton_utils import HAS_TRITON
from aphrodite.v1.worker.cpu.sampler import CPUSampler
from aphrodite.v1.worker.cpu.states import CPURequestState

if current_platform.is_cpu():
    import aphrodite._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu() or HAS_TRITON or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

VOCAB_SIZE = 128
NEG_INF = float("-inf")


def _make_sampler(params: SamplingParams, tokens: list[int], prompt_len: int) -> tuple[CPUSampler, torch.Tensor]:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    sampler = CPUSampler(
        aphrodite_config=AphroditeConfig(),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
        req_states=req_states,
    )
    req_states.add_request("req", prompt_len, tokens, len(tokens), 16)
    req_idx = req_states.req_id_to_index["req"]
    sampler.add_request(req_idx, params)
    return sampler, torch.tensor([req_idx], dtype=torch.int64)


def _batch_args(idx_mapping: torch.Tensor, pos: int, input_id: int) -> tuple:
    return (
        idx_mapping,
        idx_mapping,
        idx_mapping.numpy(),
        torch.tensor([pos], dtype=torch.int64),
        torch.tensor([input_id], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([pos + 1], dtype=np.int64),
    )


def _process(sampler: CPUSampler, idx_mapping: torch.Tensor, logits: torch.Tensor, pos: int, input_id: int):
    return sampler.apply_sampling_params(logits.clone(), *_batch_args(idx_mapping, pos, input_id))


def _sample(sampler: CPUSampler, idx_mapping: torch.Tensor, logits: torch.Tensor, pos: int, input_id: int):
    sampled, _ = sampler.sample(logits.clone(), *_batch_args(idx_mapping, pos, input_id))
    return sampled.tolist()


def _peaked_logits(values: dict[int, float]) -> torch.Tensor:
    logits = torch.full((1, VOCAB_SIZE), -20.0)
    for token, value in values.items():
        logits[0, token] = value
    return logits


def test_cpu_top_a_truncates_relative_to_top_probability() -> None:
    sampler, idx_mapping = _make_sampler(SamplingParams(temperature=1.0, top_a=0.5), [1], prompt_len=1)
    assert sampler.fork_state is not None and sampler.fork_state.any(idx_mapping.numpy())

    logits = _process(sampler, idx_mapping, _peaked_logits({5: 10.0, 6: 9.9, 7: 0.0}), pos=0, input_id=1)

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [5, 6]


def test_cpu_dry_penalizes_repeated_sequence_continuation() -> None:
    history = [1, 2, 3, 1, 2]
    params = SamplingParams(temperature=1.0, dry_multiplier=1.0, dry_base=1.75, dry_allowed_length=2)
    sampler, idx_mapping = _make_sampler(params, history, prompt_len=len(history))

    logits = _process(sampler, idx_mapping, torch.zeros(1, VOCAB_SIZE), pos=len(history) - 1, input_id=history[-1])

    assert logits[0, 3] < 0
    others = torch.cat([logits[0, :3], logits[0, 4:]])
    assert torch.all(others == 0)


def test_cpu_dry_owns_penalties_without_applying_them_twice() -> None:
    params = SamplingParams(temperature=1.0, dry_multiplier=1.0, repetition_penalty=2.0)
    sampler, idx_mapping = _make_sampler(params, [4], prompt_len=1)

    assert float(sampler.penalties_state.repetition_penalty[int(idx_mapping[0])]) == 1.0
    logits = _process(sampler, idx_mapping, _peaked_logits({4: 2.0}), pos=0, input_id=4)

    # Penalized exactly once by the ordered pipeline.
    assert logits[0, 4].item() == pytest.approx(1.0)


def test_cpu_xtc_removes_top_choices_above_threshold() -> None:
    params = SamplingParams(temperature=1.0, xtc_threshold=0.1, xtc_probability=1.0)
    sampler, idx_mapping = _make_sampler(params, [1], prompt_len=1)

    logits = _process(sampler, idx_mapping, _peaked_logits({5: 10.0, 6: 9.9, 7: 9.8}), pos=0, input_id=1)

    assert logits[0, 5] == NEG_INF
    assert logits[0, 6] == NEG_INF
    assert torch.isfinite(logits[0, 7])


@pytest.mark.parametrize("fork_sampling", [False, True])
def test_cpu_phrase_retry_blocks_rejected_token_at_retry_position(fork_sampling: bool) -> None:
    params = SamplingParams(
        temperature=0.0,
        top_a=0.01 if fork_sampling else 0.0,
        extra_args={"_sonar_phrase_retry": (1, [3])},
    )
    sampler, idx_mapping = _make_sampler(params, [1], prompt_len=1)
    assert (sampler.fork_state is not None) == fork_sampling
    logits = _peaked_logits({2: 4.0, 3: 5.0})

    # prompt_len + retry position - 1 == 1: the rejected token is masked.
    assert _sample(sampler, idx_mapping, logits, pos=1, input_id=1) == [2]
    # Other positions sample normally.
    assert _sample(sampler, idx_mapping, logits, pos=2, input_id=2) == [3]


def test_cpu_phrase_retry_reopens_first_token_when_everything_is_blocked() -> None:
    params = SamplingParams(temperature=0.0, extra_args={"_sonar_phrase_retry": (1, [3])})
    sampler, idx_mapping = _make_sampler(params, [1], prompt_len=1)
    logits = torch.full((1, VOCAB_SIZE), NEG_INF)
    logits[0, 3] = 1.0

    assert _sample(sampler, idx_mapping, logits, pos=1, input_id=1) == [3]


def test_cpu_fork_state_survives_preemption_until_finished() -> None:
    sampler, idx_mapping = _make_sampler(SamplingParams(temperature=1.0, top_a=0.5), [1], prompt_len=1)
    assert sampler.fork_state is not None
    req_idx = int(idx_mapping[0])

    sampler.remove_request(req_idx)
    assert req_idx not in sampler.fork_state.requests
    assert "req" in sampler.fork_state.retained

    sampler.finish_request("req")
    assert "req" not in sampler.fork_state.retained
