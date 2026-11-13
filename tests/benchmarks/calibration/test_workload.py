# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from sonar.benchmarks.calibration.workload import (
    Session,
    TokenSampler,
    Turn,
    WorkloadError,
    load_workload,
    next_prompt,
    parse_trace,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def request(t, inp, out, hashes, *, api_time=1.0, **extra):
    return {"t": t, "type": "n", "model": "m", "in": inp, "out": out, "hash_ids": hashes, "api_time": api_time} | extra


TRACE = {
    "id": "trace-1",
    "models": ["m"],
    "block_size": 4,
    "hash_id_scope": "local",
    "unknown_field": True,
    "requests": [
        request(0.0, 10, 3, [1, 2]),
        # Shares both full blocks of the previous prompt: a continuation.
        request(5.0, 16, 2, [1, 2, 3, 4], api_time=2.0),
        {
            "t": 9.0,
            "type": "subagent",
            "agent_id": "agent_001",
            "status": "completed",
            "requests": [request(9.5, 12, 2, [1, 2, 9]), request(30.0, 15, 2, [1, 2, 9])],
        },
        # Compaction: shares only the first block.
        request(100.0, 9, 2, [1, 7, 8], think_time=3.0),
    ],
}


def test_weka_trace_derives_continuations_reuse_and_delays():
    session, truncated = parse_trace(TRACE, max_model_len=1000, max_idle_seconds=10.0)
    assert not truncated
    assert session is not None and session.id == "trace-1"
    first, second, subagent, last = session.steps
    assert first == Turn(10, 3, False, 0, 0.0)
    assert isinstance(second, Turn) and isinstance(last, Turn) and isinstance(subagent, Session)
    assert second.continues and second.delay_seconds == pytest.approx(4.0)
    sub_first, sub_second = subagent.steps
    assert isinstance(sub_first, Turn) and isinstance(sub_second, Turn)
    # A subagent starts its own context, sharing two blocks with the parent.
    assert not sub_first.continues and sub_first.reuse_tokens == 8
    assert sub_first.delay_seconds == pytest.approx(9.5 - 5.0 - 2.0)
    assert sub_second.continues
    assert sub_second.delay_seconds == pytest.approx(10.0)  # capped
    assert not last.continues and last.reuse_tokens == 4


def test_missing_timestamps_fall_back_to_think_time():
    trace = {
        "id": "t",
        "block_size": 4,
        "requests": [request(None, 8, 1, [1, 2]), request(None, 12, 1, [1, 2, 3], think_time=2.5)],
    }
    session, _ = parse_trace(trace, max_model_len=100)
    assert session is not None
    second = session.steps[1]
    assert isinstance(second, Turn) and second.delay_seconds == pytest.approx(2.5)


def test_requests_beyond_max_model_len_end_the_session():
    session, truncated = parse_trace(TRACE, max_model_len=18)
    assert truncated
    assert session is not None and len(session.steps) == 1


def test_trace_files_and_directories_load(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(TRACE))
    (tmp_path / "b.json").write_text(json.dumps(TRACE | {"id": "trace-2"}))
    workload = load_workload(str(tmp_path), max_model_len=1000, cache_dir=tmp_path / "cache")
    assert [session.id for session in workload.sessions] == ["trace-1", "trace-2"]
    assert workload.stats.requests == 10
    lines = tmp_path / "traces.jsonl"
    lines.write_text(json.dumps(TRACE) + "\n" + json.dumps(TRACE | {"id": "x"}) + "\n")
    assert load_workload(str(lines), max_model_len=1000).stats.sessions == 2


def test_unknown_workload_spec_is_rejected(tmp_path):
    with pytest.raises(WorkloadError, match="not 'coding-agents'"):
        load_workload(str(tmp_path / "missing"), max_model_len=1000)


def test_synthetic_workload_is_seeded_and_validated():
    spec = "synthetic:sessions=5,turns=4,context=1000,new=100,output=50"
    a = load_workload(spec, max_model_len=100000, seed=1)
    b = load_workload(spec, max_model_len=100000, seed=1)
    assert a.sessions == b.sessions and a.stats.sessions == 5
    for session in a.sessions:
        assert all(isinstance(turn, Turn) and turn.continues for turn in session.steps[1:])
    with pytest.raises(WorkloadError, match="Valid keys"):
        load_workload("synthetic:bogus=1", max_model_len=1000)
    with pytest.raises(WorkloadError, match="needs a number"):
        load_workload("synthetic:turns=many", max_model_len=1000)


def test_token_sampler_is_deterministic_and_avoids_special_tokens():
    sampler = TokenSampler(10, [0, 1, 2], seed=4)
    tokens = sampler.fresh(500, key=("a", 1))
    assert tokens == sampler.fresh(500, key=("a", 1))
    assert tokens != sampler.fresh(500, key=("a", 2))
    assert set(tokens) <= set(range(3, 10))


def test_next_prompt_continues_reuses_or_starts_fresh():
    sampler = TokenSampler(1000, [], seed=0)
    fresh = next_prompt(None, None, Turn(5, 2, True, 0, 0.0), sampler, ("k", 0))
    assert len(fresh) == 5
    continued = next_prompt(fresh, [7, 8], Turn(10, 2, True, 0, 0.0), sampler, ("k", 1))
    assert continued[:7] == fresh + [7, 8] and len(continued) == 10
    reused = next_prompt(continued, [9], Turn(6, 2, False, 4, 0.0), sampler, ("k", 2))
    assert reused[:4] == continued[:4] and len(reused) == 6
