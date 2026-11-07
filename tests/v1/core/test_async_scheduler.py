# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from unittest.mock import Mock

import pytest

from sonar.v1.core.sched.async_scheduler import AsyncScheduler
from sonar.v1.core.sched.output import CachedRequestData, SchedulerOutput
from sonar.v1.outputs import ModelRunnerOutput
from sonar.v1.request import RequestStatus
from sonar.v1.structured_output import StructuredOutputGrammar
from sonar.v1.utils import ConstantList

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_no_spec_decode_padding_up_to_max_model_len():
    """Uniform spec-decode padding must leave room for the sampled token.

    Padding a single-token decode up to exactly max_model_len makes the
    running-loop max_model_len cap negative on the next step (async
    scheduling schedules it before the output that would stop the request),
    and a negative num_scheduled_tokens crashes the model runner. The
    request is scheduled un-padded instead.

    Modelled on a disaggregated decode worker, where every request arrives
    with all but the last prompt token already computed.
    """
    max_model_len = 32
    num_spec = 1
    scheduler = create_scheduler(
        async_scheduling=True,
        max_model_len=max_model_len,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )

    # A running decode, so padding's "something already scheduled and no
    # prefill this step" precondition holds for the request below.
    (filler,) = create_requests(num_requests=1, num_tokens=4, req_ids=["filler"])
    scheduler.add_request(filler)
    sched_output = scheduler.schedule()
    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    (request,) = create_requests(num_requests=1, num_tokens=max_model_len - num_spec, req_ids=["boundary"])
    request.num_computed_tokens = request.num_tokens - 1
    scheduler.add_request(request)

    sched_output = scheduler.schedule()
    assert sched_output.num_scheduled_tokens["boundary"] == 1
    assert "boundary" not in sched_output.scheduled_spec_decode_tokens
    assert request.num_computed_tokens < max_model_len

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_connector_metadata_precedes_async_placeholder_advance(monkeypatch):
    """Mirroring must see earlier unresolved outputs, excluding the current step."""
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    (request,) = create_requests(num_requests=1, num_tokens=4, max_tokens=4)
    scheduler.add_request(request)

    assert scheduler.connector is not None
    build_connector_meta = scheduler.connector.build_connector_meta
    observed_placeholders = []

    def record_placeholders(scheduler_output):
        observed_placeholders.append(request.num_output_placeholders)
        return build_connector_meta(scheduler_output)

    monkeypatch.setattr(scheduler.connector, "build_connector_meta", record_placeholders)
    scheduler.schedule()
    scheduler.schedule()

    assert observed_placeholders == [0, 1]


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert sched_output.num_scheduled_tokens[req1.request_id] == num_prompt_tokens % BLOCK_SIZE

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert sched_output.num_scheduled_tokens[added_req.request_id] == num_prompt_tokens % BLOCK_SIZE

    assert scheduler.get_num_unfinished_requests() == 0


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(requests[i].output_token_ids)
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (req.num_prompt_tokens % BLOCK_SIZE)


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.accept_tokens.return_value = False
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = {}
    scheduler.encoder_cache_mismatch_reqs = set()
    scheduler.sonar_config = Mock()
    scheduler.aux_output_connector = None
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.num_spec_tokens = 0
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


def test_no_placeholder_underflow_on_stale_spec_frame():
    num_spec = 5
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )
    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.num_computed_tokens = req.num_tokens
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    req.status = RequestStatus.RUNNING

    req.num_output_placeholders = num_spec + 1
    req.num_in_flight_tokens = num_spec + 1
    scheduler.running.remove(req)
    scheduler._preempt_request(req, 0.0)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: num_spec + 1},
        total_num_scheduled_tokens=num_spec + 1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={req.request_id: [10] * num_spec},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[999]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert req.num_output_placeholders == 0
    assert req.num_computed_tokens == 0
    assert req.num_stale_output_tokens == 0
    assert list(req.output_token_ids) == [999]
    assert req.status == RequestStatus.PREEMPTED


def test_requires_kv_delivery_defaults_to_producer_role():
    # No connector: nothing is handed off, so keep the lossless deliver-stale
    # path on preemption.
    assert create_scheduler(async_scheduling=True).requires_kv_delivery is False
    # Only a producer hands KV off when a request completes.
    for role, expected in (
        ("kv_producer", True),
        ("kv_both", True),
        ("kv_consumer", False),
    ):
        scheduler = create_scheduler(async_scheduling=True, use_kv_connector=True, kv_role=role)
        assert scheduler.requires_kv_delivery is expected, role


@pytest.mark.parametrize(
    ("kv_role", "defer_free"),
    [
        ("kv_producer", False),
        ("kv_consumer", False),
        ("kv_consumer", True),
    ],
)
def test_kv_pressure_preempt_mid_handoff(kv_role: str, defer_free: bool):
    """P/D race: KV pressure hits while the output of a request's final
    prefill chunk -- the hand-off token that would finish it -- is in flight.

    When the victim's blocks free immediately, the request is preempted. On a
    producer, that output must be dropped so the request recomputes; delivering
    it would finish the request and hand off blocks the preemption already
    freed, so the consumer pulls garbage. A consumer hands nothing off, so it
    keeps the lossless deliver-stale path.

    A consumer with overlapping batches (async scheduling or PP) instead fences
    the victim's free behind its in-flight output, so the allocation retry
    stops instead of preempting; the request then finishes from that output
    once it lands. The gate depends on the platform (async scheduling is
    force-disabled on CPU), so force the flag to cover both paths everywhere.
    """
    is_producer = kv_role == "kv_producer"
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=True,
        kv_role=kv_role,
        num_blocks=5,
        block_size=16,
        max_num_batched_tokens=512,
    )
    assert scheduler.requires_kv_delivery is is_producer
    # The production gate requires overlapping batches, which async
    # scheduling only provides off-CPU.
    scheduler.defer_block_free = defer_free

    # 32-token prompts fill 2 blocks each, exhausting the usable pool, so the
    # next decode allocation targets the tail of the running queue (the
    # handoff request) while its prefill output is still in flight.
    decoder = create_requests(num_requests=1, num_tokens=32, max_tokens=8, req_ids=["decoder"])[0]
    handoff = create_requests(num_requests=1, num_tokens=32, max_tokens=1, req_ids=["handoff"])[0]
    scheduler.add_request(decoder)
    scheduler.add_request(handoff)
    sched_output = scheduler.schedule()
    assert handoff.status == RequestStatus.RUNNING
    assert handoff.num_output_placeholders == 1

    scheduler.schedule()
    if defer_free:
        # The victim's blocks are fenced behind its in-flight output, so
        # preempting it could not satisfy the allocation; the retry stops
        # instead of preempting.
        assert handoff.status == RequestStatus.RUNNING
        assert handoff.num_stale_output_tokens == 0
    else:
        # Blocks free immediately, so the handoff request is preempted.
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_stale_output_tokens == handoff.num_prompt_tokens
        assert handoff.drop_stale_output is is_producer

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    assert handoff.num_stale_output_tokens == 0
    if is_producer:
        assert not handoff.is_finished()
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_output_tokens == 0
        assert handoff.request_id in scheduler.requests
    else:
        assert handoff.is_finished()
        assert handoff.num_output_tokens == 1


def test_resumable_request_handoff():
    """New streaming input resumes with the correct context and stale results
    are ignored.
    """
    scheduler = create_scheduler(
        async_scheduling=True,
        use_v2_model_runner=True,
    )

    old_prompt_len = 10
    continuation_len = 12
    request_id = "resumable-race"

    request = create_requests(1, num_tokens=old_prompt_len, max_tokens=4, req_ids=[request_id])[0]
    request.resumable = True
    continuation = create_requests(1, num_tokens=continuation_len, max_tokens=4, req_ids=[request_id])[0]
    continuation.resumable = True

    scheduler.add_request(request)

    pending = deque(scheduler.schedule() for _ in range(3))
    first_step = pending.popleft()
    first_output = _make_model_runner_output(first_step)
    first_output.sampled_token_ids = [[EOS_TOKEN_ID]]
    scheduler.update_from_output(first_step, first_output)
    scheduler.add_request(continuation)
    new_step = scheduler.schedule()
    assert new_step.scheduled_new_reqs[0].num_computed_tokens == old_prompt_len
    pending.append(new_step)

    all_token_ids_before_stale = list(request.all_token_ids)
    for _ in range(2):
        stale_step = pending.popleft()
        scheduler.update_from_output(stale_step, _make_model_runner_output(stale_step))
    assert list(request.output_token_ids) == []
    assert list(request.all_token_ids) == all_token_ids_before_stale

    resumed_step = pending.popleft()
    scheduler.update_from_output(resumed_step, _make_model_runner_output(resumed_step))
    assert len(request.output_token_ids) == 1
