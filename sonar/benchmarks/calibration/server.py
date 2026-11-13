# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start, watch and stop the `sonar serve` processes used by calibration."""

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import aiohttp
import regex as re

# Serve options the calibration sets itself, with the number of values each takes.
_REPLACED_OPTIONS = {"--host": 1, "--port": 1, "--prefill-cost-coefficients": 2}

# Startup log lines and the phase each one starts. The latest match wins:
# engines and ranks log in parallel, so the order varies.
_PHASES = (
    (re.compile(r"Resolved architecture"), "Resolving the model"),
    (re.compile(r"DP group leader|distributed_init_method"), "Starting workers"),
    (re.compile(r"Loading model|Initializing model|Loading safetensors"), "Loading weights"),
    (re.compile(r"Model loading took"), "Profiling memory"),
    (re.compile(r"GPU KV cache size"), "Allocating the KV cache"),
    (re.compile(r"Autotuning process starts|Warming up"), "Warming up kernels"),
    (re.compile(r"Compiling a graph|Dynamo bytecode transform"), "Compiling"),
    (re.compile(r"Capturing CUDA graph"), "Capturing CUDA graphs"),
    (re.compile(r"init engine .* took"), "Starting the API server"),
    (re.compile(r"Application startup complete|Starting Sonar server on"), "Ready"),
)
_KV_CAPACITY = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_METRIC_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


class ServerStartError(RuntimeError):
    """`sonar serve` exited or did not become healthy."""


def normalise_option(argument: str) -> str:
    name, _, _ = argument.partition("=")
    return name.replace("_", "-") if name.startswith("--") else name


def rewrite_serve_args(
    serve_args: Sequence[str], *, host: str, port: int, coefficients: tuple[float, float]
) -> list[str]:
    """`serve_args` without the options calibration controls, followed by its
    own values for them."""
    result = []
    skip = 0
    for argument in serve_args:
        if skip:
            skip -= 1
            continue
        name = normalise_option(argument)
        if name in _REPLACED_OPTIONS:
            if "=" not in argument:
                skip = _REPLACED_OPTIONS[name]
            continue
        result.append(argument)
    return [
        *result,
        "--host",
        host,
        "--port",
        str(port),
        "--prefill-cost-coefficients",
        repr(coefficients[0]),
        repr(coefficients[1]),
    ]


def replaced_options(serve_args: Sequence[str]) -> list[str]:
    """The options in `serve_args` that calibration will replace."""
    return sorted({normalise_option(arg) for arg in serve_args} & set(_REPLACED_OPTIONS))


def parse_prometheus(text: str) -> dict[str, float]:
    """Samples by metric name, summed over label sets (engines, ranks)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        name, labels, value = match.groups()
        if labels and 'le="' in labels:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + number
    return values


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ManagedServer:
    """One `sonar serve` process group with its log in `log_path`."""

    def __init__(
        self,
        serve_args: Sequence[str],
        coefficients: tuple[float, float],
        log_path: Path,
        *,
        startup_timeout: float,
    ) -> None:
        self.port = free_port()
        self.host = "127.0.0.1"
        self.base_url = f"http://{self.host}:{self.port}"
        self.command = [
            sys.executable,
            "-m",
            "sonar.entrypoints.cli.main",
            "serve",
            *rewrite_serve_args(serve_args, host=self.host, port=self.port, coefficients=coefficients),
        ]
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.phase = "Launching"
        self.last_line = ""
        self.kv_capacity_tokens: int | None = None
        self.started_at = time.monotonic()
        self.ready_at: float | None = None
        self._tail: deque[str] = deque(maxlen=40)
        self._process: subprocess.Popen | None = None
        self._log_offset = 0

    @property
    def startup_seconds(self) -> float | None:
        return None if self.ready_at is None else self.ready_at - self.started_at

    def tail(self) -> list[str]:
        return list(self._tail)

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.log_path.parent / "command.txt").write_text(" ".join(self.command) + "\n")
        self.started_at = time.monotonic()
        with self.log_path.open("wb") as log:
            self._process = subprocess.Popen(
                self.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

    def poll_log(self) -> None:
        """Read new log lines and advance the startup phase."""
        try:
            with self.log_path.open("rb") as log:
                log.seek(self._log_offset)
                data = log.read()
        except FileNotFoundError:
            return
        if not data:
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        self._log_offset += end + 1
        for raw in data[: end + 1].splitlines():
            # Progress bars redraw with carriage returns; keep the latest state.
            line = raw.decode(errors="replace").rsplit("\r", 1)[-1].rstrip()
            if not line:
                continue
            self._tail.append(line)
            self.last_line = line
            if self.kv_capacity_tokens is None and (match := _KV_CAPACITY.search(line)):
                self.kv_capacity_tokens = int(match.group(1).replace(",", ""))
            for pattern, phase in _PHASES:
                if pattern.search(line):
                    self.phase = phase
                    break

    async def wait_ready(self) -> None:
        assert self._process is not None
        deadline = self.started_at + self.startup_timeout
        async with aiohttp.ClientSession() as http:
            while True:
                self.poll_log()
                if self._process.poll() is not None:
                    raise ServerStartError(
                        f"sonar serve exited with code {self._process.returncode}. Last log lines:\n"
                        + "\n".join(self.tail()[-30:])
                    )
                with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError):
                    async with http.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            self.ready_at = time.monotonic()
                            self.phase = "Ready"
                            return
                if time.monotonic() > deadline:
                    raise ServerStartError(
                        f"sonar serve was not healthy after {self.startup_timeout:.0f} s "
                        "(raise --startup-timeout). Last log lines:\n" + "\n".join(self.tail()[-30:])
                    )
                await asyncio.sleep(1.0)

    async def metrics(self, http: aiohttp.ClientSession) -> dict[str, float]:
        async with http.get(f"{self.base_url}/metrics", timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            return parse_prometheus(await response.text())

    async def model_info(self, http: aiohttp.ClientSession) -> tuple[str, int | None]:
        async with http.get(f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            model = (await response.json())["data"][0]
            return model["id"], model.get("max_model_len")

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)
            try:
                process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        # Engine and worker processes share the group; make sure none remain.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        self.poll_log()

    async def __aenter__(self) -> "ManagedServer":
        self.start()
        try:
            await self.wait_ready()
        except BaseException:
            await asyncio.to_thread(self.stop)
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await asyncio.to_thread(self.stop)
