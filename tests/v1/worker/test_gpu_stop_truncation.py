# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncating speculative acceptance at the request's stop (Model Runner V2).

With `speculative_config.truncate_acceptance_at_stop`,
`get_num_sampled_and_rejected` cuts each request's sampled tokens at its
stop. The reference is the scheduler itself: it appends a step's tokens one at
a time and stops at `check_stop`, the way `_update_request_with_output`
trims them. A step whose first input token already ended its request accepts
at most that token (the scheduler then keeps none).
"""

import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sonar.sampling_params import SamplingParams
from sonar.v1.core.sched.utils import check_stop
from sonar.v1.request import Request
from sonar.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from sonar.v1.worker.gpu.sample.logit_bias import MAX_NUM_STOP_TOKEN_IDS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
DEVICE = torch.device("cuda")
NUM_DRAFTS = 3
MAX_MODEL_LEN = 4096
# Tokens of the prompt and of the outputs before the step: none is a stop.
FILLER = 7


@dataclass
class Row:
    prompt_len: int
    # Position of the step's first input token.
    start: int
    first_input: int
    sampled: list[int]
    num_sampled: int
    max_tokens: int
    eos_token_id: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    num_logits: int = 1 + NUM_DRAFTS
    # Larger than start + num_logits for a chunked-prefill row.
    prefill_len: int | None = None

    def params(self) -> SamplingParams:
        params = SamplingParams(
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.eos_token_id is None,
        )
        params.update_from_generation_config({}, self.eos_token_id)
        return params

    @property
    def max_len(self) -> int:
        return min(self.prompt_len + self.max_tokens, MAX_MODEL_LEN)


def scheduler_keeps(row: Row) -> int:
    """The tokens of the step the device must accept: those the scheduler
    keeps, or at most the first input token when that ended the request."""
    if row.prefill_len is not None and row.prefill_len > row.start + row.num_logits:
        return 0
    num_outputs = row.start + 1 - row.prompt_len
    request = Request(
        request_id="r",
        prompt_token_ids=[FILLER] * row.prompt_len,
        sampling_params=row.params(),
        pooling_params=None,
    )
    if num_outputs > 0:
        request.append_output_token_ids([FILLER] * (num_outputs - 1))
        request.append_output_token_ids(row.first_input)
        if check_stop(request, MAX_MODEL_LEN):
            return min(row.num_sampled, 1)
    for i, token in enumerate(row.sampled[: row.num_sampled]):
        request.append_output_token_ids(token)
        if check_stop(request, MAX_MODEL_LEN):
            return i + 1
    return row.num_sampled


def run_kernel(rows: list[Row], truncate: bool) -> tuple[list[int], list[int]]:
    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=DEVICE)

    cu_num_logits = [0]
    for row in rows:
        cu_num_logits.append(cu_num_logits[-1] + row.num_logits)
    # Deliberately reverse persistent request slots relative to batch rows.
    state_rows = rows[::-1]
    stop_token_ids = torch.full((len(rows), MAX_NUM_STOP_TOKEN_IDS), -1, dtype=torch.int32, device=DEVICE)
    for i, row in enumerate(state_rows):
        stop_token_ids[i, : len(row.stop_token_ids)] = tensor(row.stop_token_ids)
    # The tensors StopConditions keeps on the device, one row per request.
    stop_conditions = SimpleNamespace(
        eos_token_id=SimpleNamespace(
            gpu=tensor([-1 if r.eos_token_id is None else r.eos_token_id for r in state_rows])
        ),
        max_len=SimpleNamespace(gpu=tensor([r.max_len for r in state_rows])),
        num_stop_token_ids=SimpleNamespace(gpu=tensor([len(r.stop_token_ids) for r in state_rows])),
        stop_token_ids=SimpleNamespace(gpu=stop_token_ids),
    )
    kwargs = {}
    if truncate:
        kwargs = dict(
            stop_conditions=stop_conditions,
            sampled=tensor([r.sampled for r in rows], torch.int64),
            last_sampled_tokens=tensor([[r.first_input] for r in state_rows], torch.int64),
            prompt_len=tensor([r.prompt_len for r in state_rows]),
        )
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        tensor([r.num_sampled for r in rows]),
        tensor([r.start + r.num_logits for r in rows]),
        tensor(cu_num_logits),
        torch.arange(len(rows) - 1, -1, -1, dtype=torch.int32, device=DEVICE),
        tensor([r.prefill_len or r.prompt_len for r in state_rows]),
        **kwargs,
    )
    return num_sampled.tolist(), num_rejected.tolist()


def random_row(rng: random.Random) -> Row:
    vocab = list(range(10, 30))
    prompt_len = rng.randrange(1, 60)
    verify = rng.random() < 0.8
    num_logits = 1 + NUM_DRAFTS if verify else 1
    # A verify step starts past the prompt; a final prefill chunk ends at it.
    start = rng.randrange(prompt_len, prompt_len + 40) if verify else prompt_len - 1
    num_outputs = start + 1 - prompt_len
    return Row(
        prompt_len=prompt_len,
        start=start,
        first_input=rng.choice(vocab),
        sampled=[rng.choice(vocab) for _ in range(1 + NUM_DRAFTS)],
        num_sampled=rng.randrange(1, num_logits + 1),
        # A request has not passed its length limit before the step.
        max_tokens=rng.randrange(max(1, num_outputs), num_outputs + 8),
        eos_token_id=rng.choice([None, rng.choice(vocab)]),
        stop_token_ids=rng.sample(vocab, rng.randrange(0, 3)),
        num_logits=num_logits,
    )


@pytest.mark.parametrize("seed", range(6))
def test_truncation_matches_the_scheduler(seed: int):
    rng = random.Random(seed)
    rows = [random_row(rng) for _ in range(64)]
    num_sampled, num_rejected = run_kernel(rows, truncate=True)
    for row, got, rejected in zip(rows, num_sampled, num_rejected):
        want = scheduler_keeps(row)
        assert got == want, row
        assert rejected == row.num_logits - want


def test_named_cases():
    # The step's first input token is at position 20; its sampled tokens make
    # the request 22, 23, 24 and 25 tokens long.
    base = dict(
        prompt_len=10,
        start=20,
        first_input=50,
        sampled=[11, 12, 13, 14],
        num_sampled=4,
        max_tokens=1000,
    )
    cases: list[tuple[dict[str, Any], int]] = [
        ({}, 4),  # no stop
        (dict(max_tokens=13), 2),  # the length limit at the second token
        (dict(max_tokens=12), 1),  # at the first
        (dict(num_sampled=2, max_tokens=15), 2),  # past the accepted tokens
        (dict(eos_token_id=13), 3),  # EOS accepted as a draft
        (dict(eos_token_id=13, max_tokens=12), 1),  # the earlier stop wins
        (dict(stop_token_ids=[12]), 2),  # a stop token id
        (dict(first_input=99, eos_token_id=99), 1),  # the input ended the request
        (dict(num_sampled=3, max_tokens=11), 1),  # length reached before the step
        # The first input is a prompt token: a stop id there ends nothing.
        (dict(prompt_len=12, start=9, first_input=99, eos_token_id=99), 4),
    ]
    rows = [Row(**{**base, **over}) for over, _ in cases]
    assert [scheduler_keeps(row) for row in rows] == [want for _, want in cases]
    num_sampled, _ = run_kernel(rows, truncate=True)
    assert num_sampled == [want for _, want in cases]


def test_truncation_ignores_nonfinal_prefill_and_caps_model_length():
    rows = [
        Row(32, 4, 99, [99, 99, 99, 99], 1, 10, eos_token_id=99, num_logits=1),
        Row(10, MAX_MODEL_LEN - 3, 7, [11, 12, 13, 14], 4, MAX_MODEL_LEN),
        Row(10, 20, 99, [11, 12, 13, 14], 0, 100, eos_token_id=99),
    ]
    num_sampled, num_rejected = run_kernel(rows, truncate=True)
    assert num_sampled == [0, 2, 0]
    assert num_rejected == [0, 2, 4]


def test_without_truncation_only_chunked_prefill_rows_change():
    rng = random.Random(7)
    rows = [random_row(rng) for _ in range(32)]
    rows[3].prefill_len = rows[3].start + rows[3].num_logits + 5
    num_sampled, num_rejected = run_kernel(rows, truncate=False)
    for i, (row, got, rejected) in enumerate(zip(rows, num_sampled, num_rejected)):
        if i == 3:
            assert (got, rejected) == (0, 0)
        else:
            assert (got, rejected) == (row.num_sampled, row.num_logits - got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs UVA buffers")
def test_stop_conditions_follow_the_sampling_params():
    from sonar.v1.worker.gpu.spec_decode.stop_conditions import StopConditions
    from sonar.v1.worker.gpu.states import RequestState

    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=64,
        num_speculative_steps=NUM_DRAFTS,
        vocab_size=100,
        device=DEVICE,
    )
    stop_conditions = StopConditions(req_states)
    rows = {
        "eos": Row(10, 0, 0, [], 0, max_tokens=5, eos_token_id=2, stop_token_ids=[9]),
        "ignore_eos": Row(10, 0, 0, [], 0, max_tokens=100, eos_token_id=None),
    }
    for req_id, row in rows.items():
        req_states.add_request(req_id, row.prompt_len, [1] * row.prompt_len, 0, 5)
        stop_conditions.add_request(req_states.req_id_to_index[req_id], row.params())
    req_states.apply_staged_writes()
    stop_conditions.apply_staged_writes()
    eos = req_states.req_id_to_index["eos"]
    ignore_eos = req_states.req_id_to_index["ignore_eos"]
    assert stop_conditions.eos_token_id.gpu[eos].item() == 2
    assert stop_conditions.eos_token_id.gpu[ignore_eos].item() == -1
    assert stop_conditions.max_len.gpu[eos].item() == 15
    # The model length bounds a longer max_tokens.
    assert stop_conditions.max_len.gpu[ignore_eos].item() == 64
    assert stop_conditions.num_stop_token_ids.gpu[eos].item() == 1
    assert stop_conditions.stop_token_ids.gpu[eos, 0].item() == 9

    # Reusing a slot must discard its old EOS, stop IDs and length bound.
    req_states.remove_request("eos")
    req_states.add_request("replacement", 12, [1] * 12, 0, 40)
    replacement = req_states.req_id_to_index["replacement"]
    assert replacement == eos
    stop_conditions.add_request(replacement, SamplingParams(ignore_eos=True, max_tokens=40))
    req_states.apply_staged_writes()
    stop_conditions.apply_staged_writes()
    assert stop_conditions.eos_token_id.gpu[replacement].item() == -1
    assert stop_conditions.num_stop_token_ids.gpu[replacement].item() == 0
    assert stop_conditions.max_len.gpu[replacement].item() == 52

    too_many = SamplingParams(stop_token_ids=list(range(MAX_NUM_STOP_TOKEN_IDS + 1)))
    with pytest.raises(ValueError, match="Too many stop tokens"):
        stop_conditions.add_request(eos, too_many)
