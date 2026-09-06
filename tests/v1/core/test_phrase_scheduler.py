# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import numpy as np
import pytest

from aphrodite.v1.outputs import LogprobsLists, ModelRunnerOutput
from aphrodite.v1.phrase_guard.logprobs import PendingLogprobs
from aphrodite.v1.phrase_guard.matcher import PendingText
from aphrodite.v1.phrase_guard.processor import PhraseRetryProcessor
from aphrodite.v1.phrase_guard.scheduler import PhraseScheduler
from aphrodite.v1.request import RequestStatus
from tests.v1.core.utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class FakeGuard:
    def __init__(self, request, tokenizer):
        self.pending = PendingText(("Once upon time",))
        self.retries = 0
        self.retry_position = -1
        self.blocked = []
        self.logprobs = None

    def decode(self, token):
        return {10: "Once", 11: " upon", 12: " time", 13: "Fine."}[token]

    def reset_decoder(self):
        pass


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setenv("APHRODITE_USE_V2_MODEL_RUNNER", "0")
    with patch("tests.v1.core.utils.AsyncScheduler", PhraseScheduler):
        result = create_scheduler(async_scheduling=True, enable_prefix_caching=True, use_v2_model_runner=False)
    result.tokenizer = object()
    result.aphrodite_config.model_config.logits_processors = [PhraseRetryProcessor]
    return result


def add_requests(scheduler):
    a, b = create_requests(2, num_tokens=48, same_prompt=True, max_tokens=24)
    a.sampling_params = a.sampling_params.clone()
    a.sampling_params.extra_args = {"banned_strings": ["Once upon time"]}
    with patch("aphrodite.v1.phrase_guard.scheduler.RequestGuard", FakeGuard):
        scheduler.add_request(a)
        scheduler.add_request(b)
    return a, b


def accept(scheduler, scheduled, tokens):
    req_ids = list(scheduled.num_scheduled_tokens)
    return scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[tokens[rid]] for rid in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_rewind_drains_old_frames_without_stalling_other_requests(scheduler):
    a, b = add_requests(scheduler)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert a.request_id in first.num_scheduled_tokens
    assert a.request_id in second.num_scheduled_tokens
    assert b.request_id in second.num_scheduled_tokens
    assert set(scheduler.running) == {a, b}
    accept(scheduler, first, {a.request_id: 10, b.request_id: 20})
    third = scheduler.schedule()
    accept(scheduler, second, {a.request_id: 11, b.request_id: 20})
    fourth = scheduler.schedule()
    old_tail = scheduler.kv_cache_manager.get_blocks(a.request_id).blocks[0][-1]
    accept(scheduler, third, {a.request_id: 12, b.request_id: 20})
    assert a.num_in_flight_tokens > 0
    assert old_tail.ref_cnt > 0
    fenced = scheduler.schedule()
    assert a.request_id not in fenced.num_scheduled_tokens
    assert b.request_id in fenced.num_scheduled_tokens
    assert scheduler.has_requests()
    accept(scheduler, fourth, {a.request_id: 10, b.request_id: 20})
    assert old_tail.ref_cnt == 0
    assert not a.output_token_ids
    assert a.num_in_flight_tokens == 0
    accept(scheduler, fenced, {b.request_id: 20})
    resumed = scheduler.schedule()
    assert a.request_id in resumed.phrase_rewinds
    accept(scheduler, resumed, {a.request_id: 13, b.request_id: 20})
    assert list(a.output_token_ids) == [13]
    assert list(b.output_token_ids) == [20] * 6


@pytest.mark.parametrize("v2", [False, True])
def test_rewind_preserves_other_request_and_resumes_prefix(scheduler, v2):
    scheduler.use_v2_model_runner = v2
    a, b = add_requests(scheduler)
    published = []
    for token in [10, 11, 12]:
        scheduled = scheduler.schedule()
        before = scheduler.kv_cache_manager.get_block_ids(b.request_id)
        batches = accept(scheduler, scheduled, {a.request_id: token, b.request_id: 20})
        for batch in batches.values():
            for output in batch.outputs:
                if output.request_id == a.request_id:
                    published.extend(output.new_token_ids)
        assert scheduler.kv_cache_manager.get_block_ids(b.request_id) == before
    assert published == []
    assert a.status == RequestStatus.PREEMPTED
    assert list(a.output_token_ids) == []
    assert list(b.output_token_ids) == [20, 20, 20]
    assert a.num_output_placeholders == 0
    assert a.num_in_flight_tokens == 0
    resumed = scheduler.schedule()
    if v2:
        restored = next(r for r in resumed.scheduled_new_reqs if r.req_id == a.request_id)
        assert restored.sampling_params.extra_args["_sonar_phrase_retry"] == (0, [10])
        assert a.sampling_params is restored.sampling_params
        assert restored.num_computed_tokens >= 32
    else:
        assert a.request_id in resumed.phrase_rewinds
        row = resumed.scheduled_cached_reqs.req_ids.index(a.request_id)
        assert resumed.scheduled_cached_reqs.num_computed_tokens[row] >= 32
    batches = accept(scheduler, resumed, {a.request_id: 13, b.request_id: 20})
    outputs = [o for batch in batches.values() for o in batch.outputs]
    assert next(o for o in outputs if o.request_id == a.request_id).new_token_ids == [13]


@pytest.mark.parametrize("v2", [False, True])
def test_exhausted_retry_finishes_only_affected_request(scheduler, v2):
    scheduler.use_v2_model_runner = v2
    a, b = add_requests(scheduler)
    scheduler.guards[a.request_id].pending = PendingText(("Once",))
    a.sampling_params.allowed_token_ids = [10]
    accept(scheduler, scheduler.schedule(), {a.request_id: 10, b.request_id: 20})
    assert a.status == RequestStatus.PREEMPTED
    batches = accept(scheduler, scheduler.schedule(), {a.request_id: 10, b.request_id: 20})
    output = next(o for batch in batches.values() for o in batch.outputs if o.request_id == a.request_id)
    assert a.status == RequestStatus.FINISHED_ERROR
    assert output.finish_reason is not None
    assert not output.new_token_ids
    assert not a.output_token_ids
    assert a.request_id not in scheduler.rewinds
    assert b.status == RequestStatus.RUNNING
    assert list(b.output_token_ids) == [20, 20]


@pytest.mark.parametrize("include_stop", [False, True])
@pytest.mark.parametrize("eos", [False, True])
def test_stop_token_completing_phrase_terminates_without_retry(scheduler, include_stop, eos):
    a, b = add_requests(scheduler)
    a.sampling_params.include_stop_str_in_output = include_stop
    if eos:
        a.sampling_params._eos_token_id = 12
    else:
        a.sampling_params.stop_token_ids = [12]
    for token in [10, 11, 12]:
        batches = accept(scheduler, scheduler.schedule(), {a.request_id: token, b.request_id: 20})
    output = next(o for batch in batches.values() for o in batch.outputs if o.request_id == a.request_id)
    assert a.request_id not in scheduler.rewinds
    if include_stop:
        assert a.status == RequestStatus.FINISHED_ERROR
        assert output.new_token_ids == []
    else:
        assert a.status == RequestStatus.FINISHED_STOPPED
        assert output.new_token_ids == [10, 11, 12]
    assert b.status == RequestStatus.RUNNING


def test_complete_ban_checked_before_length_limit(scheduler):
    a, b = add_requests(scheduler)
    a.max_tokens = 3
    for token in [10, 11, 12]:
        accept(scheduler, scheduler.schedule(), {a.request_id: token, b.request_id: 20})
    assert a.status == RequestStatus.PREEMPTED
    assert not a.output_token_ids


def test_rewind_carries_exact_output_position(scheduler):
    a, b = add_requests(scheduler)
    for token in [13, 10, 11, 12]:
        accept(scheduler, scheduler.schedule(), {a.request_id: token, b.request_id: 20})
    assert list(a.output_token_ids) == [13]
    assert scheduler.rewinds[a.request_id][1].extra_args["_sonar_phrase_retry"] == (1, [10])


def test_ban_detected_in_output_of_memory_preempted_request(scheduler):
    a, b = add_requests(scheduler)
    for token in [10, 11]:
        accept(scheduler, scheduler.schedule(), {a.request_id: token, b.request_id: 20})
    last = scheduler.schedule()
    scheduler.running.remove(a)
    scheduler._preempt_request(a, 0.0)
    accept(scheduler, last, {a.request_id: 12, b.request_id: 20})
    assert a.num_output_placeholders == 0
    resumed = scheduler.schedule()
    assert a.request_id in resumed.phrase_rewinds
    accept(scheduler, resumed, {a.request_id: 13, b.request_id: 20})
    assert list(a.output_token_ids) == [13]


def test_abort_during_rewind_drains_without_resuming(scheduler):
    a, b = add_requests(scheduler)
    for token in [10, 11]:
        accept(scheduler, scheduler.schedule(), {a.request_id: token, b.request_id: 20})
    hit = scheduler.schedule()
    stale = scheduler.schedule()
    accept(scheduler, hit, {a.request_id: 12, b.request_id: 20})
    scheduler.finish_requests(a.request_id, RequestStatus.FINISHED_ABORTED)
    assert a.request_id not in scheduler.guards
    assert a.request_id not in scheduler.rewinds
    accept(scheduler, stale, {a.request_id: 10, b.request_id: 20})
    assert a.request_id not in scheduler.schedule().num_scheduled_tokens


def test_logprobs_follow_held_tokens_and_discard_rejected_rows(scheduler):
    a, b = add_requests(scheduler)
    a.sampling_params.logprobs = 1
    guard = scheduler.guards[a.request_id]
    guard.logprobs = PendingLogprobs()
    guard.prompt_logprobs = None
    scheduler.logprob_guards.add(a.request_id)
    published = []
    prompt_rows = object()
    published_prompt_rows = []
    for token in [10, 11, 12, 13]:
        scheduled = scheduler.schedule()
        ids = list(scheduled.num_scheduled_tokens)
        tokens = [[token if rid == a.request_id else 20] for rid in ids]
        batches = scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=tokens,
                logprobs=LogprobsLists(np.array(tokens), np.full((len(ids), 1), -0.5), np.ones(len(ids))),
                prompt_logprobs_dict={a.request_id: prompt_rows} if token == 10 else {},
                pooler_output=[],
            ),
        )
        for batch in batches.values():
            for output in batch.outputs:
                if output.request_id == a.request_id:
                    assert output.new_logprobs.logprob_token_ids[:, 0].tolist() == output.new_token_ids
                    published.extend(output.new_token_ids)
                    if output.new_prompt_logprobs_tensors is not None:
                        published_prompt_rows.append(output.new_prompt_logprobs_tensors)
    assert published == [13]
    assert published_prompt_rows == [prompt_rows]
    assert not guard.logprobs.rows


def test_speculative_packet_discards_tokens_after_first_match(scheduler):
    a, _ = add_requests(scheduler)
    scheduler.schedule()
    a.num_output_placeholders = 4
    ready, stopped = scheduler._update_request_with_output(a, [10, 11, 12, 13])
    assert ready == []
    assert not stopped
    assert not a.output_token_ids
    assert a.status == RequestStatus.PREEMPTED
    assert scheduler.rewinds[a.request_id][1].extra_args["_sonar_phrase_retry"] == (0, [10])
