# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoggingStatLogger via StatLoggerManager.record() and log()."""

import logging

import pytest
import regex as re

from sonar.config import SonarConfig
from sonar.config.device import DeviceConfig
from sonar.v1.engine import FinishReason
from sonar.v1.metrics.loggers import LoggingStatLogger, StatLoggerManager
from sonar.v1.metrics.stats import FinishedRequestStats, IterationStats, SchedulerStats

LOGGER_NAME = "sonar.v1.metrics.loggers"


class _StubPrometheusStatLogger:
    """Stand-in for PrometheusStatLogger, which requires a registry."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass

    def log(self) -> None:
        pass


@pytest.fixture()
def _mock_prometheus(monkeypatch):
    monkeypatch.setattr("sonar.v1.metrics.loggers.PrometheusStatLogger", _StubPrometheusStatLogger)
    monkeypatch.setenv("SONAR_REQUEST_LEVEL_METRICS", "0")


@pytest.fixture()
def manager(_mock_prometheus, caplog_sonar):
    return StatLoggerManager(SonarConfig(device_config=DeviceConfig(device="cpu")))


@pytest.fixture()
def aggregated_manager(_mock_prometheus, caplog_sonar):
    return StatLoggerManager(
        SonarConfig(device_config=DeviceConfig(device="cpu")),
        aggregate_engine_logging=True,
    )


def _iteration(prompt_tokens: int = 0, gen_tokens: int = 0, finished: int = 0) -> IterationStats:
    it = IterationStats()
    it.prompt_token_stats.computed = prompt_tokens
    it.num_generation_tokens = gen_tokens
    it.finished_requests = [FinishedRequestStats(finish_reason=FinishReason.LENGTH) for _ in range(finished)]
    return it


def _log_and_capture(manager, caplog) -> logging.LogRecord:
    before = len(caplog.records)
    manager.log()
    stats = [r for r in caplog.records[before:] if "Avg prompt throughput" in r.getMessage()]
    assert stats, "expected the periodic serving-stats line"
    return stats[-1]


def _throughput(line: str, field: str) -> float:
    match = re.search(rf"{field} throughput: ([\d.]+) tokens/s", line)
    assert match, line
    return float(match.group(1))


def test_iteration_counters_log_then_reset(manager, caplog_sonar):
    caplog_sonar.set_level(logging.DEBUG, logger=LOGGER_NAME)
    manager.record(None, _iteration(prompt_tokens=100, gen_tokens=200, finished=3))
    manager.record(None, _iteration(prompt_tokens=100, gen_tokens=200, finished=1))

    record = _log_and_capture(manager, caplog_sonar)
    line = record.getMessage()
    assert record.levelno == logging.INFO
    assert "Finished: 4 reqs" in line
    assert _throughput(line, "Avg prompt") > 0
    assert _throughput(line, "Avg generation") > 0

    # A new interval starts from zero: nothing recorded, nothing carried.
    record = _log_and_capture(manager, caplog_sonar)
    line = record.getMessage()
    assert "Finished: 0 reqs" in line
    assert _throughput(line, "Avg prompt") == 0.0
    assert _throughput(line, "Avg generation") == 0.0


def test_scheduler_values_reflect_latest_snapshot(manager, caplog_sonar):
    caplog_sonar.set_level(logging.DEBUG, logger=LOGGER_NAME)
    manager.record(
        SchedulerStats(num_running_reqs=3, num_waiting_reqs=1),
        _iteration(gen_tokens=10),
    )
    manager.record(
        SchedulerStats(num_running_reqs=7, num_waiting_reqs=2),
        _iteration(gen_tokens=10),
    )

    line = _log_and_capture(manager, caplog_sonar).getMessage()
    assert "Running: 7 reqs" in line
    assert "Waiting: 2 reqs" in line


def test_preemptions_and_corrupted_render(manager, caplog_sonar, monkeypatch):
    caplog_sonar.set_level(logging.DEBUG, logger=LOGGER_NAME)
    monkeypatch.setattr("sonar.v1.metrics.loggers.envs.SONAR_COMPUTE_NANS_IN_LOGITS", True)
    it = _iteration(gen_tokens=10)
    it.num_preempted_reqs = 5
    it.num_corrupted_reqs = 2
    manager.record(None, it)

    line = _log_and_capture(manager, caplog_sonar).getMessage()
    assert "Preemptions: 5" in line
    assert "Corrupted: 2 reqs" in line


def test_aggregated_logger_sums_engines(aggregated_manager, caplog_sonar):
    caplog_sonar.set_level(logging.DEBUG, logger=LOGGER_NAME)
    aggregated_manager.record(
        SchedulerStats(num_running_reqs=3, num_waiting_reqs=1),
        _iteration(gen_tokens=100, finished=2),
        engine_idx=0,
    )

    record = _log_and_capture(aggregated_manager, caplog_sonar)
    line = record.getMessage()
    assert "Engines Aggregated" in line
    assert "Running: 3 reqs" in line
    assert "Finished: 2 reqs" in line
    assert _throughput(line, "Avg generation") > 0


def test_quiet_intervals_drop_to_debug(manager, caplog_sonar):
    caplog_sonar.set_level(logging.DEBUG, logger=LOGGER_NAME)

    # Prompt-only activity logs at INFO.
    manager.record(None, _iteration(prompt_tokens=1000))
    assert _log_and_capture(manager, caplog_sonar).levelno == logging.INFO

    # One quiet interval keeps INFO (the previous interval was active)...
    assert _log_and_capture(manager, caplog_sonar).levelno == logging.INFO

    # ...the second quiet interval drops to DEBUG.
    assert _log_and_capture(manager, caplog_sonar).levelno == logging.DEBUG

    # Generation-only activity restores INFO.
    manager.record(None, _iteration(gen_tokens=1000))
    record = _log_and_capture(manager, caplog_sonar)
    assert record.levelno == logging.INFO
    assert _throughput(record.getMessage(), "Avg generation") > 0


def test_request_metrics_remain_the_default(monkeypatch, caplog_sonar):
    monkeypatch.delenv("SONAR_REQUEST_LEVEL_METRICS", raising=False)
    caplog_sonar.set_level(logging.INFO, logger=LOGGER_NAME)
    stat_logger = LoggingStatLogger(SonarConfig(device_config=DeviceConfig(device="cpu")))
    try:
        assert stat_logger.request_level_metrics
        stats = IterationStats()
        stats.finished_requests.append(
            FinishedRequestStats(
                finish_reason=FinishReason.LENGTH,
                e2e_latency=3.0,
                first_token_latency=0.5,
                num_prompt_tokens=100,
                num_generation_tokens=20,
                num_cached_tokens=40,
                prefill_time=1.0,
                decode_time=2.0,
            )
        )
        stat_logger.record(None, stats)
        stat_logger.log()
    finally:
        stat_logger.log_queue.put(None)
        stat_logger.log_thread.join(timeout=5)
    assert not stat_logger.log_thread.is_alive()
    assert "TTFT: 0.50s" in caplog_sonar.text
    assert "Prefill: 100 tokens (100.0 tokens/s)" in caplog_sonar.text
    assert "Decode: 20 tokens (10.0 tokens/s)" in caplog_sonar.text
    assert "Cache hits: 40 tokens (40.0%)" in caplog_sonar.text
    assert "Avg prompt throughput" not in caplog_sonar.text
