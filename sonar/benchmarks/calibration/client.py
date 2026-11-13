# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-loop replay of a workload against a running Sonar server."""

import asyncio
import json
import math
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

import aiohttp

from sonar.benchmarks.calibration.workload import Session, TokenSampler, Turn, Workload, next_prompt

CANCELLED = "cancelled"


@dataclass(slots=True)
class RequestRecord:
    lane: int
    session: str
    start: float
    first_token: float | None = None
    end: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class TokenTimeline:
    """Output tokens counted per second of arrival."""

    def __init__(self, origin: float) -> None:
        self.origin = origin
        self._counts: list[int] = []

    def add(self, when: float, tokens: int) -> None:
        index = max(0, int(when - self.origin))
        if index >= len(self._counts):
            self._counts.extend([0] * (index + 1 - len(self._counts)))
        self._counts[index] += tokens

    def tokens(self, start: float, end: float) -> int:
        first = max(0, int(start - self.origin))
        last = max(first, int(end - self.origin))
        return sum(self._counts[first:last])


async def _sse_events(response: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
    """The `data:` payloads of a server-sent event stream. The first chunk
    echoes the prompt's token IDs, so lines can be far longer than aiohttp's
    line reader allows."""
    buffer = bytearray()
    async for block in response.content.iter_any():
        # Search only the new bytes, so a long line is not rescanned per block.
        position = len(buffer)
        buffer += block
        while (newline := buffer.find(b"\n", position)) >= 0:
            line = bytes(buffer[:newline]).strip()
            del buffer[: newline + 1]
            position = 0
            if line.startswith(b"data:"):
                yield line[5:].strip()
    line = bytes(buffer).strip()
    if line.startswith(b"data:"):
        yield line[5:].strip()


async def complete_once(
    http: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    *,
    timeout: float,
    lane: int = -1,
    session: str = "",
    timeline: TokenTimeline | None = None,
    on_start: Callable[[RequestRecord], None] | None = None,
) -> tuple[RequestRecord, list[int]]:
    """Stream one completion of exactly `max_tokens` tokens for `prompt`."""
    record = RequestRecord(lane=lane, session=session, start=time.monotonic(), prompt_tokens=len(prompt))
    if on_start is not None:
        on_start(record)
    output: list[int] = []
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with http.post(
            f"{base_url}/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                record.error = f"HTTP {response.status}: {(await response.text())[:300]}"
                return record, output
            async for data in _sse_events(response):
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    record.error = str(chunk["error"])[:300]
                    break
                tokens = 0
                for choice in chunk.get("choices") or ():
                    delta = choice.get("token_ids") or ()
                    output.extend(delta)
                    tokens += len(delta)
                if tokens:
                    now = time.monotonic()
                    if record.first_token is None:
                        record.first_token = now
                    record.output_tokens += tokens
                    if timeline is not None:
                        timeline.add(now, tokens)
    except asyncio.CancelledError:
        record.error = CANCELLED
        raise
    except Exception as error:
        # A failed request ends its session; it must not stop the calibration.
        record.error = f"{type(error).__name__}: {error}"[:300]
    finally:
        record.end = time.monotonic()
    if record.error is None and record.output_tokens < max_tokens:
        record.error = f"stream ended after {record.output_tokens} of {max_tokens} tokens"
    return record, output


@dataclass(frozen=True, slots=True)
class RunnerSnapshot:
    elapsed: float
    active_lanes: int
    in_flight: int
    completed: int
    failed: int
    output_tokens_per_second: float
    ttft_p90_recent: float | None
    sessions_started: int


@dataclass(slots=True)
class _Counters:
    in_flight: int = 0
    completed: int = 0
    failed: int = 0
    sessions_started: int = 0
    active_lanes: int = 0
    recent_ttft: deque = field(default_factory=deque)


class _SessionEnded(Exception):
    def __init__(self, failed: bool = False) -> None:
        super().__init__()
        self.failed = failed


class WorkloadRunner:
    """Replays `workload` with `lanes` concurrent sessions, each starting the
    next request after the previous one finishes."""

    def __init__(
        self,
        base_url: str,
        model: str,
        workload: Workload,
        *,
        lanes: int,
        sampler: TokenSampler,
        max_model_len: int,
        seed: int,
        trajectory_start: tuple[float, float] = (0.25, 0.75),
        request_timeout: float = 3600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workload = workload
        self.lanes = lanes
        self.sampler = sampler
        self.max_model_len = max_model_len
        self.seed = seed
        self.trajectory_start = trajectory_start
        self.request_timeout = request_timeout
        order = list(range(len(workload.sessions)))
        random.Random(seed).shuffle(order)
        self._order = order
        self._next_instance = 0
        self._records: list[RequestRecord] = []
        self._counters = _Counters()
        self.origin = time.monotonic()
        self.timeline = TokenTimeline(self.origin)
        self._deadline = math.inf

    @property
    def records(self) -> list[RequestRecord]:
        return self._records

    async def run(self, duration: float) -> list[RequestRecord]:
        self.origin = time.monotonic()
        self.timeline = TokenTimeline(self.origin)
        self._deadline = self.origin + duration
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as http:
            lanes = [asyncio.create_task(self._lane(lane, http)) for lane in range(self.lanes)]
            try:
                await asyncio.sleep(max(0.0, self._deadline - time.monotonic()))
            finally:
                for task in lanes:
                    task.cancel()
                await asyncio.gather(*lanes, return_exceptions=True)
        now = time.monotonic()
        for record in self._records:
            if record.end is None:
                record.end = now
                record.error = CANCELLED
        return self._records

    def snapshot(self) -> RunnerSnapshot:
        now = time.monotonic()
        counters = self._counters
        while counters.recent_ttft and counters.recent_ttft[0][0] < now - 60:
            counters.recent_ttft.popleft()
        ttfts = sorted(value for _, value in counters.recent_ttft)
        window = min(30.0, now - self.origin)
        rate = self.timeline.tokens(now - window, now) / window if window >= 1 else 0.0
        return RunnerSnapshot(
            elapsed=now - self.origin,
            active_lanes=counters.active_lanes,
            in_flight=counters.in_flight,
            completed=counters.completed,
            failed=counters.failed,
            output_tokens_per_second=rate,
            ttft_p90_recent=_percentile(ttfts, 0.9) if ttfts else None,
            sessions_started=counters.sessions_started,
        )

    async def _lane(self, lane: int, http: aiohttp.ClientSession) -> None:
        while time.monotonic() < self._deadline:
            instance = self._next_instance
            self._next_instance += 1
            session = self.workload.sessions[self._order[instance % len(self._order)]]
            self._counters.sessions_started += 1
            self._counters.active_lanes += 1
            try:
                await self._run_session(lane, http, session, instance)
            except _SessionEnded as ended:
                if ended.failed:
                    # Back off so an unavailable server is not flooded.
                    await asyncio.sleep(1.0)
            finally:
                self._counters.active_lanes -= 1

    async def _run_session(self, lane: int, http: aiohttp.ClientSession, session: Session, instance: int) -> None:
        steps = session.steps
        rng = random.Random(f"{self.seed}:{instance}")
        low, high = self.trajectory_start
        start = int(rng.uniform(low, high) * len(steps))
        while start < len(steps) and not isinstance(steps[start], Turn):
            start += 1
        if start >= len(steps):
            start = next(i for i, step in enumerate(steps) if isinstance(step, Turn))
        await self._run_chain(lane, http, session.id, steps[start:], None, None, (instance, start))

    async def _run_chain(
        self,
        lane: int,
        http: aiohttp.ClientSession,
        session_id: str,
        steps: Sequence["Turn | Session"],
        prompt: list[int] | None,
        output: list[int] | None,
        key: tuple,
    ) -> None:
        for index, step in enumerate(steps):
            if isinstance(step, Session):
                # A subagent shares a prefix with the parent's latest prompt;
                # the parent continues from its own context afterwards.
                await self._run_chain(lane, http, session_id, step.steps, prompt, None, (*key, index))
                continue
            if prompt is not None and step.delay_seconds > 0:
                await asyncio.sleep(step.delay_seconds)
            if time.monotonic() >= self._deadline:
                raise _SessionEnded
            next_tokens = next_prompt(prompt, output, step, self.sampler, (*key, index))
            if len(next_tokens) + step.output_tokens > self.max_model_len - 1:
                raise _SessionEnded
            record, generated = await self._request(lane, http, session_id, next_tokens, step.output_tokens)
            if record.error is not None:
                raise _SessionEnded(failed=True)
            prompt, output = next_tokens, generated

    async def _request(
        self, lane: int, http: aiohttp.ClientSession, session_id: str, prompt: list[int], max_tokens: int
    ) -> tuple[RequestRecord, list[int]]:
        counters = self._counters
        counters.in_flight += 1
        try:
            record, generated = await complete_once(
                http,
                self.base_url,
                self.model,
                prompt,
                max_tokens,
                timeout=self.request_timeout,
                lane=lane,
                session=session_id,
                timeline=self.timeline,
                on_start=self._records.append,
            )
        finally:
            counters.in_flight -= 1
        if record.error is None:
            counters.completed += 1
            assert record.first_token is not None
            counters.recent_ttft.append((record.first_token, record.first_token - record.start))
        else:
            counters.failed += 1
        return record, generated


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True, slots=True)
class WindowSummary:
    output_tokens_per_second: float
    interval_output_tokens_per_second: tuple[float, ...]
    ttft_p50: float
    ttft_p90: float
    completed_requests: int
    failed_requests: int


def summarize_window(
    records: Sequence[RequestRecord],
    timeline: TokenTimeline,
    start: float,
    end: float,
    *,
    interval: float = 60.0,
) -> WindowSummary:
    """Throughput and TTFT over `[start, end)`. Tokens count by arrival time;
    TTFT covers successful requests whose first token arrived in the window."""
    length = end - start
    if length <= 0:
        raise ValueError("The measurement window is empty.")
    rates = []
    position = start
    while position < end:
        span = min(interval, end - position)
        if span < interval / 2:
            break
        rates.append(timeline.tokens(position, position + span) / span)
        position += span
    ttfts = sorted(
        record.first_token - record.start
        for record in records
        if record.error is None and record.first_token is not None and start <= record.first_token < end
    )
    failed = sum(1 for record in records if record.error not in (None, CANCELLED) and start <= record.start < end)
    return WindowSummary(
        output_tokens_per_second=timeline.tokens(start, end) / length,
        interval_output_tokens_per_second=tuple(rates),
        ttft_p50=_percentile(ttfts, 0.5) if ttfts else math.nan,
        ttft_p90=_percentile(ttfts, 0.9) if ttfts else math.nan,
        completed_requests=len(ttfts),
        failed_requests=failed,
    )
