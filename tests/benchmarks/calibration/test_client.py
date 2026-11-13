# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import socket

import aiohttp
import pytest
from aiohttp import web

from sonar.benchmarks.calibration.client import (
    CANCELLED,
    RequestRecord,
    TokenTimeline,
    WorkloadRunner,
    complete_once,
    summarize_window,
)
from sonar.benchmarks.calibration.workload import Session, TokenSampler, Turn, Workload, compute_stats

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class FakeServer:
    """Streams `max_tokens` output tokens, one per chunk."""

    def __init__(self, *, fail_on_length: int | None = None, delay: float = 0.0) -> None:
        self.prompts: list[list[int]] = []
        self.fail_on_length = fail_on_length
        self.delay = delay

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body["prompt"]
        self.prompts.append(prompt)
        assert body["ignore_eos"] and body["return_token_ids"] and body["stream"]
        if len(prompt) == self.fail_on_length:
            return web.Response(status=500, text="boom")
        response = web.StreamResponse()
        await response.prepare(request)
        for index in range(body["max_tokens"]):
            await asyncio.sleep(self.delay)
            choice = {"index": 0, "token_ids": [900 + index], "text": "x"}
            if index == 0:
                choice["prompt_token_ids"] = prompt
            await response.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
        await response.write(b'data: {"choices": [], "usage": {}}\n\ndata: [DONE]\n\n')
        return response


async def serve(fake: FakeServer):
    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/completions", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, f"http://127.0.0.1:{port}"


def test_complete_once_streams_exact_tokens_even_with_huge_prompt_echoes():
    # The first chunk echoes the prompt; 200k token IDs make a line over 1 MB.
    long_prompt = list(range(200_000))

    async def scenario():
        fake = FakeServer()
        runner, url = await serve(fake)
        try:
            timeline = TokenTimeline(0.0)
            async with aiohttp.ClientSession() as http:
                record, output = await complete_once(http, url, "m", [1, 2, 3], 4, timeout=10, timeline=timeline)
                long, long_output = await complete_once(http, url, "m", long_prompt, 2, timeout=30)
                failed, _ = await complete_once(http, url + "/missing", "m", [1], 1, timeout=10)
        finally:
            await runner.cleanup()
        return record, output, long, long_output, failed

    record, output, long, long_output, failed = asyncio.run(scenario())
    assert record.error is None and output == [900, 901, 902, 903]
    assert record.output_tokens == 4 and record.prompt_tokens == 3
    assert record.start <= record.first_token <= record.end
    assert long.error is None and long_output == [900, 901]
    assert failed.error is not None and failed.error.startswith("HTTP 404")


def _workload(sessions):
    return Workload(name="t", description="t", sessions=tuple(sessions), stats=compute_stats(list(sessions), 0))


def test_runner_continues_sessions_from_previous_output():
    session = Session("s", (Turn(5, 2, False, 0, 0.0), Turn(10, 2, True, 0, 0.0), Turn(6, 1, False, 3, 0.0)))

    async def scenario():
        fake = FakeServer(delay=0.002)
        runner, url = await serve(fake)
        try:
            workload_runner = WorkloadRunner(
                url,
                "m",
                _workload([session]),
                lanes=1,
                sampler=TokenSampler(500, [], seed=0),
                max_model_len=100,
                seed=0,
                trajectory_start=(0.0, 0.01),
            )
            records = await workload_runner.run(0.5)
        finally:
            await runner.cleanup()
        return fake.prompts, records

    prompts, records = asyncio.run(scenario())
    first, second, third = prompts[:3]
    assert len(first) == 5
    assert second[:7] == first + [900, 901] and len(second) == 10
    assert third[:3] == second[:3] and len(third) == 6
    assert any(record.error is None for record in records)


def test_failed_requests_end_the_session_and_cancel_at_the_deadline():
    session = Session("s", (Turn(5, 2, False, 0, 0.0), Turn(10, 2, True, 0, 0.0), Turn(20, 50, True, 0, 0.0)))

    async def scenario():
        fake = FakeServer(fail_on_length=10, delay=0.001)
        runner, url = await serve(fake)
        try:
            workload_runner = WorkloadRunner(
                url,
                "m",
                _workload([session]),
                lanes=2,
                sampler=TokenSampler(500, [], seed=0),
                max_model_len=1000,
                seed=0,
                trajectory_start=(0.0, 0.01),
            )
            records = await workload_runner.run(0.3)
            snapshot = workload_runner.snapshot()
        finally:
            await runner.cleanup()
        return fake.prompts, records, snapshot

    prompts, records, snapshot = asyncio.run(scenario())
    # The third turn never runs: its session ended when the second failed.
    assert all(len(prompt) in (5, 10) for prompt in prompts)
    assert snapshot.failed >= 1
    assert all(record.end is not None for record in records)


def test_summarize_window_counts_by_arrival_and_skips_cancelled():
    timeline = TokenTimeline(0.0)
    for second in range(100):
        timeline.add(second + 0.5, 10)
    records = [
        RequestRecord(lane=0, session="a", start=10.0, first_token=11.0, end=12.0, output_tokens=5),
        RequestRecord(lane=0, session="a", start=20.0, first_token=24.0, end=25.0, output_tokens=5),
        RequestRecord(lane=0, session="b", start=5.0, first_token=6.0, end=7.0),  # before the window
        RequestRecord(lane=1, session="c", start=30.0, first_token=31.0, end=99.0, error=CANCELLED),
        RequestRecord(lane=1, session="d", start=40.0, end=41.0, error="HTTP 500: x"),
    ]
    summary = summarize_window(records, timeline, 10.0, 75.0, interval=30.0)
    assert summary.output_tokens_per_second == pytest.approx(10.0)
    # Two full 30 s intervals; the 5 s remainder is under half an interval.
    assert summary.interval_output_tokens_per_second == (10.0, 10.0)
    assert summary.completed_requests == 2
    assert summary.failed_requests == 1
    assert summary.ttft_p90 == pytest.approx(1.0 + 0.9 * 3.0)
