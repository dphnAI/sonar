# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure uncached prefill time at several prompt lengths."""

import asyncio
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import aiohttp

from sonar.benchmarks.calibration.client import complete_once
from sonar.benchmarks.calibration.fit import PrefillSample
from sonar.benchmarks.calibration.server import ManagedServer
from sonar.benchmarks.calibration.workload import TokenSampler

PREFILL_SUM = "sonar:request_prefill_time_seconds_sum"
PREFILL_COUNT = "sonar:request_prefill_time_seconds_count"
MIN_PROFILE_TOKENS = 256
SATURATED_PROFILE_TOKENS = 8192
NUM_LENGTHS = 8
REPEATS = 3


@dataclass(frozen=True, slots=True)
class ProfileMeasurement:
    tokens: int
    seconds: float
    # "server": scheduled-to-first-token time from the server's histogram.
    # "client": time to first token seen by the client, used when the server
    # histogram did not record exactly this request.
    source: str

    def sample(self) -> PrefillSample:
        return PrefillSample(self.tokens, self.seconds)


def profile_lengths(max_tokens: int, count: int = NUM_LENGTHS) -> list[int]:
    """Geometrically spaced prompt lengths up to `max_tokens`. They start at
    8192 tokens when possible: shorter prefills are dominated by per-step
    costs that do not grow with length, and would bias the fit."""
    if max_tokens < MIN_PROFILE_TOKENS * 16:
        raise ValueError(f"--profile-max-tokens must be at least {MIN_PROFILE_TOKENS * 16}, got {max_tokens}.")
    low = min(SATURATED_PROFILE_TOKENS, max_tokens // 16)
    ratio = (max_tokens / low) ** (1 / (count - 1))
    return sorted({round(low * ratio**i) for i in range(count)})


def profile_schedule(max_tokens: int, seed: int) -> list[int]:
    """Every length `REPEATS` times, shuffled so drift does not bias one length."""
    schedule = [length for length in profile_lengths(max_tokens) for _ in range(REPEATS)]
    random.Random(seed).shuffle(schedule)
    return schedule


def prefill_seconds(
    before: Mapping[str, float], after: Mapping[str, float], client_seconds: float
) -> tuple[float, str]:
    """The server's prefill time for the single request between two metric
    scrapes, or the client's time to first token when the server did not
    record exactly one request."""
    count = after.get(PREFILL_COUNT, math.nan) - before.get(PREFILL_COUNT, math.nan)
    seconds = after.get(PREFILL_SUM, math.nan) - before.get(PREFILL_SUM, math.nan)
    if count == 1 and seconds > 0:
        return seconds, "server"
    return client_seconds, "client"


async def _measure(
    server: ManagedServer, http: aiohttp.ClientSession, model: str, prompt: list[int], timeout: float
) -> tuple[float, str]:
    before = await server.metrics(http)
    record, _ = await complete_once(http, server.base_url, model, prompt, 1, timeout=timeout)
    if record.error is not None or record.first_token is None:
        raise RuntimeError(f"Profiling request of {len(prompt)} tokens failed: {record.error}")
    client_seconds = record.first_token - record.start
    # The histogram is recorded when the request finishes, which can trail the
    # end of the stream slightly.
    deadline = time.monotonic() + 3.0
    while True:
        after = await server.metrics(http)
        seconds, source = prefill_seconds(before, after, client_seconds)
        if source == "server" or time.monotonic() > deadline:
            return seconds, source
        await asyncio.sleep(0.1)


async def run_profile(
    server: ManagedServer,
    model: str,
    sampler: TokenSampler,
    schedule: list[int],
    done: list[ProfileMeasurement],
    *,
    on_measurement: Callable[[ProfileMeasurement], None],
    on_status: Callable[[str], None],
    timeout: float = 3600.0,
) -> list[ProfileMeasurement]:
    """Measure the requests of `schedule` that `done` does not cover yet."""
    async with aiohttp.ClientSession() as http:
        # One unmeasured request per length first: kernels compiled or tuned on
        # their first use would otherwise make that length look slower.
        lengths = sorted(set(schedule))
        for index, tokens in enumerate(lengths):
            on_status(f"Warming up ({index + 1}/{len(lengths)})")
            prompt = sampler.fresh(tokens, key=("profile-warmup", index, time.time_ns()))
            await _measure(server, http, model, prompt, timeout)
        for index in range(len(done), len(schedule)):
            tokens = schedule[index]
            on_status(f"Prefilling {tokens:,} tokens")
            # Fresh tokens per request and run, so nothing is served from cache.
            prompt = sampler.fresh(tokens, key=("profile", index, time.time_ns()))
            seconds, source = await _measure(server, http, model, prompt, timeout)
            measurement = ProfileMeasurement(tokens, seconds, source)
            done.append(measurement)
            on_measurement(measurement)
    return done
