# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-turn workloads for calibration trials.

A workload is a set of sessions. Each request in a session either continues
the previous one (its prompt is the previous prompt and output followed by
new tokens) or reuses a prefix of the previous prompt. Prompts are random
token IDs, so the workload reproduces the cache behaviour of a recorded
trace without its text.

Trace files use the Weka agentic-trace format of the public
`semianalysisai/cc-traces-weka-*` datasets."""

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sonar.benchmarks.calibration.options import CODING_AGENTS

CODING_AGENTS_REPO = "semianalysisai/cc-traces-weka-062126"
CODING_AGENTS_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
CODING_AGENTS_FILE = "traces.jsonl"
# Idle gaps between a session's requests (tool runs, user think time) are
# capped like the reference AgentX replay (`trace_idle_gap_cap_seconds`). The
# gaps leave contexts idle in the cache, which is what eviction decides about;
# without them every session is always running and nothing can be kept.
DEFAULT_MAX_IDLE_SECONDS = 300.0
_PARSER_VERSION = 2

Progress = Callable[[str, float | None], None]


@dataclass(frozen=True, slots=True)
class Turn:
    input_tokens: int
    output_tokens: int
    continues: bool
    reuse_tokens: int
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    steps: tuple["Turn | Session", ...]


@dataclass(frozen=True, slots=True)
class WorkloadStats:
    sessions: int
    requests: int
    mean_turns: float
    prompt_tokens_p50: int
    prompt_tokens_p95: int
    prompt_tokens_max: int
    output_tokens_mean: float
    mean_session_context: float
    truncated_sessions: int


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    description: str
    sessions: tuple[Session, ...]
    stats: WorkloadStats


class WorkloadError(ValueError):
    """The workload specification or trace cannot be used."""


class TokenSampler:
    """Random prompt tokens, avoiding special tokens. Each key always yields
    the same tokens."""

    def __init__(self, vocab_size: int, excluded_ids: Iterable[int], seed: int) -> None:
        allowed = np.setdiff1d(np.arange(vocab_size, dtype=np.int64), np.fromiter(excluded_ids, dtype=np.int64))
        if allowed.size == 0:
            raise WorkloadError("The tokenizer has no ordinary tokens to sample prompts from.")
        self._allowed = allowed
        self._seed = seed

    def fresh(self, n: int, *, key: tuple) -> list[int]:
        if n <= 0:
            return []
        digest = hashlib.blake2b(repr((self._seed, key)).encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        return self._allowed[rng.integers(0, self._allowed.size, size=n)].tolist()


def next_prompt(
    prev_prompt: list[int] | None,
    prev_output: list[int] | None,
    turn: Turn,
    sampler: TokenSampler,
    key: tuple,
) -> list[int]:
    """The prompt for `turn` after a request that sent `prev_prompt` and
    received `prev_output`."""
    if prev_prompt is None:
        return sampler.fresh(turn.input_tokens, key=key)
    if turn.continues and prev_output is not None:
        context = prev_prompt + prev_output
        return context + sampler.fresh(turn.input_tokens - len(context), key=key)
    reuse = min(turn.reuse_tokens, len(prev_prompt), turn.input_tokens)
    return prev_prompt[:reuse] + sampler.fresh(turn.input_tokens - reuse, key=key)


# Weka traces


def _common_blocks(a: list[int], b: list[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def _delay(current: dict, previous: dict | None, max_idle: float) -> float:
    if previous is None:
        return 0.0
    start, end, api_time = current.get("t"), previous.get("t"), previous.get("api_time")
    if start is not None and end is not None and api_time is not None:
        gap = start - end - api_time
    elif current.get("think_time") is not None:
        gap = current["think_time"]
    else:
        gap = 0.0
    if not math.isfinite(gap):
        return 0.0
    return min(max(gap, 0.0), max_idle)


class _TraceParser:
    def __init__(self, max_model_len: int, max_idle: float) -> None:
        self.max_model_len = max_model_len
        self.max_idle = max_idle
        self.truncated = False
        # The latest request in time order, for idle gaps across subagents.
        self.last_request: dict | None = None

    def chain(self, requests: list, block_size: int, parent: dict | None) -> tuple["Turn | Session", ...]:
        steps: list[Turn | Session] = []
        previous = parent
        first = True
        for index, request in enumerate(requests):
            kind = request.get("type")
            if kind == "subagent":
                if previous is None:
                    continue
                sub_steps = self.chain(request.get("requests", []), block_size, previous)
                if sub_steps:
                    steps.append(Session(id=str(request.get("agent_id", index)), steps=sub_steps))
                if self.truncated:
                    break
                continue
            if kind not in ("n", "s"):
                continue
            input_tokens, output_tokens = int(request["in"]), int(request["out"])
            if input_tokens <= 0 or output_tokens <= 0:
                continue
            if input_tokens + output_tokens > self.max_model_len - 1:
                self.truncated = True
                break
            hash_ids = request.get("hash_ids") or []
            if previous is None:
                continues, reuse = False, 0
            else:
                previous_in = previous["in"]
                shared = _common_blocks(previous["hash_ids"], hash_ids)
                if first:
                    # A subagent starts its own context and shares only a
                    # prefix with its parent.
                    continues = False
                else:
                    continues = shared >= previous_in // block_size and input_tokens >= previous_in
                reuse = min(shared * block_size, previous_in, input_tokens)
            steps.append(
                Turn(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    continues=continues,
                    reuse_tokens=0 if continues else reuse,
                    delay_seconds=_delay(request, self.last_request, self.max_idle) if steps or parent else 0.0,
                )
            )
            previous = {"in": input_tokens, "hash_ids": hash_ids}
            self.last_request = {"t": request.get("t"), "api_time": request.get("api_time")}
            first = False
        return tuple(steps)


def parse_trace(
    trace: dict, *, max_model_len: int, max_idle_seconds: float = DEFAULT_MAX_IDLE_SECONDS
) -> tuple[Session | None, bool]:
    """One session from a Weka trace, and whether `max_model_len` cut it."""
    block_size = int(trace.get("block_size") or 0)
    if block_size <= 0:
        raise WorkloadError(f"Trace {trace.get('id')!r} has no positive block_size.")
    parser = _TraceParser(max_model_len, max_idle_seconds)
    steps = parser.chain(trace.get("requests", []), block_size, None)
    if not any(isinstance(step, Turn) for step in steps):
        return None, parser.truncated
    return Session(id=str(trace.get("id")), steps=steps), parser.truncated


def _iter_trace_lines(path: Path, progress: Progress | None) -> Iterator[dict]:
    size = path.stat().st_size or 1
    with path.open("rb") as handle:
        read = 0
        last_report = -1.0
        for line in handle:
            read += len(line)
            if line.strip():
                yield json.loads(line)
            if progress is not None and read / size - last_report >= 0.01:
                last_report = read / size
                progress(f"Parsing {path.name}", last_report)


def _iter_traces(path: Path, progress: Progress | None) -> Iterator[dict]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise WorkloadError(f"No *.json trace files in {path}.")
        for index, file in enumerate(files):
            if progress is not None:
                progress(f"Parsing {file.name}", index / len(files))
            yield json.loads(file.read_text())
    elif path.suffix == ".jsonl":
        yield from _iter_trace_lines(path, progress)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        yield from data if isinstance(data, list) else [data]
    else:
        raise WorkloadError(f"Expected a .json or .jsonl trace file or a directory of .json traces, got {path}.")


def _sessions_from_traces(traces: Iterable[dict], max_model_len: int, max_idle: float) -> tuple[list[Session], int]:
    sessions = []
    truncated = 0
    for trace in traces:
        session, cut = parse_trace(trace, max_model_len=max_model_len, max_idle_seconds=max_idle)
        truncated += cut
        if session is not None:
            sessions.append(session)
    if not sessions:
        raise WorkloadError(
            f"No usable sessions: every trace exceeds max_model_len {max_model_len} at its first request."
        )
    return sessions, truncated


# Serialisation of the parsed form


def _encode(step: "Turn | Session") -> object:
    if isinstance(step, Turn):
        return [step.input_tokens, step.output_tokens, int(step.continues), step.reuse_tokens, step.delay_seconds]
    return {"id": step.id, "steps": [_encode(child) for child in step.steps]}


def _decode(data: object) -> "Turn | Session":
    if isinstance(data, list):
        input_tokens, output_tokens, continues, reuse, delay = data
        return Turn(int(input_tokens), int(output_tokens), bool(continues), int(reuse), float(delay))
    assert isinstance(data, dict)
    return Session(id=data["id"], steps=tuple(_decode(child) for child in data["steps"]))


def _load_coding_agents(
    max_model_len: int, max_idle: float, cache_dir: Path, progress: Progress | None
) -> tuple[list[Session], int]:
    cache = cache_dir / (
        f"coding-agents-{CODING_AGENTS_REVISION[:12]}-p{_PARSER_VERSION}-len{max_model_len}-idle{max_idle:g}.json"
    )
    if cache.exists():
        data = json.loads(cache.read_text())
        return [_decode(entry) for entry in data["sessions"]], int(data["truncated"])  # type: ignore[misc]
    from sonar.transformers_utils.repo_utils import hf_api

    if progress is not None:
        progress(f"Downloading {CODING_AGENTS_REPO} ({CODING_AGENTS_FILE}, about 1.9 GB, once)", None)
    path = Path(
        hf_api().hf_hub_download(
            CODING_AGENTS_REPO, CODING_AGENTS_FILE, repo_type="dataset", revision=CODING_AGENTS_REVISION
        )
    )
    sessions, truncated = _sessions_from_traces(_iter_trace_lines(path, progress), max_model_len, max_idle)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps({"sessions": [_encode(s) for s in sessions], "truncated": truncated}))
    os.replace(temporary, cache)
    return sessions, truncated


# Synthetic sessions

_SYNTHETIC_DEFAULTS = {
    "sessions": 200.0,
    "turns": 40.0,
    "context": 20000.0,
    "new": 1500.0,
    "output": 400.0,
    "delay": 20.0,
}


def _parse_synthetic(spec: str) -> dict[str, float]:
    options = dict(_SYNTHETIC_DEFAULTS)
    _, _, body = spec.partition(":")
    for item in filter(None, (part.strip() for part in body.split(","))):
        key, sep, value = item.partition("=")
        if not sep or key not in options:
            raise WorkloadError(
                f"Unknown synthetic workload option {item!r}. Valid keys: {', '.join(_SYNTHETIC_DEFAULTS)}."
            )
        try:
            options[key] = float(value)
        except ValueError:
            raise WorkloadError(f"Synthetic workload option {key} needs a number, got {value!r}.") from None
        if options[key] < 0 or (key != "delay" and options[key] <= 0):
            raise WorkloadError(f"Synthetic workload option {key} must be positive, got {value}.")
    return options


def _synthetic_sessions(options: dict[str, float], max_model_len: int, seed: int) -> tuple[list[Session], int]:
    rng = random.Random(seed)

    def draw(mean: float, sigma: float) -> int:
        return max(1, int(rng.lognormvariate(math.log(mean) - sigma * sigma / 2, sigma)))

    sessions = []
    truncated = 0
    for index in range(int(options["sessions"])):
        turns: list[Turn | Session] = []
        prompt = draw(options["context"], 0.5)
        for _ in range(max(2, draw(options["turns"], 0.6))):
            output = draw(options["output"], 0.8)
            if prompt + output > max_model_len - 1:
                truncated += 1
                break
            delay = rng.expovariate(1 / options["delay"]) if options["delay"] > 0 else 0.0
            turns.append(Turn(prompt, output, bool(turns), 0, min(delay, DEFAULT_MAX_IDLE_SECONDS) if turns else 0.0))
            prompt += output + draw(options["new"], 1.0)
        if turns:
            sessions.append(Session(id=f"synthetic-{index}", steps=tuple(turns)))
    if not sessions:
        raise WorkloadError(f"Synthetic sessions do not fit max_model_len {max_model_len}; lower context=.")
    return sessions, truncated


def describe_synthetic() -> str:
    return ", ".join(f"{key}={value:g}" for key, value in _SYNTHETIC_DEFAULTS.items())


# Statistics and entry point


def _walk_turns(steps: Iterable["Turn | Session"]) -> Iterator[Turn]:
    for step in steps:
        if isinstance(step, Turn):
            yield step
        else:
            yield from _walk_turns(step.steps)


def compute_stats(sessions: list[Session], truncated: int) -> WorkloadStats:
    prompts: list[int] = []
    outputs: list[int] = []
    top_level_turns = []
    contexts: list[int] = []
    for session in sessions:
        top = [step for step in session.steps if isinstance(step, Turn)]
        top_level_turns.append(len(top))
        contexts.extend(turn.input_tokens + turn.output_tokens for turn in top)
        for turn in _walk_turns(session.steps):
            prompts.append(turn.input_tokens)
            outputs.append(turn.output_tokens)
    ordered = np.sort(np.array(prompts))
    return WorkloadStats(
        sessions=len(sessions),
        requests=len(prompts),
        mean_turns=float(np.mean(top_level_turns)),
        prompt_tokens_p50=int(np.percentile(ordered, 50)),
        prompt_tokens_p95=int(np.percentile(ordered, 95)),
        prompt_tokens_max=int(ordered[-1]),
        output_tokens_mean=float(np.mean(outputs)),
        mean_session_context=float(np.mean(contexts)),
        truncated_sessions=truncated,
    )


def load_workload(
    spec: str,
    *,
    max_model_len: int,
    seed: int = 0,
    cache_dir: Path | None = None,
    progress: Progress | None = None,
    max_idle_seconds: float = DEFAULT_MAX_IDLE_SECONDS,
) -> Workload:
    """Load `coding-agents`, `synthetic[:key=value,...]`, or a Weka trace
    file or directory."""
    cache_dir = cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sonar" / "calibration"
    if spec == CODING_AGENTS:
        sessions, truncated = _load_coding_agents(max_model_len, max_idle_seconds, cache_dir, progress)
        description = f"{{}} coding-agent sessions ({CODING_AGENTS_REPO}@{CODING_AGENTS_REVISION[:7]})"
        name = CODING_AGENTS
    elif spec == "synthetic" or spec.startswith("synthetic:"):
        options = _parse_synthetic(spec)
        sessions, truncated = _synthetic_sessions(options, max_model_len, seed)
        name = "synthetic"
        description = (
            "{} synthetic agent sessions ("
            + ", ".join(f"{key}={value:g}" for key, value in options.items() if key != "sessions")
            + ")"
        )
    else:
        path = Path(spec).expanduser()
        if not path.exists():
            raise WorkloadError(
                f"Workload {spec!r} is not 'coding-agents', 'synthetic[:key=value,...]' or an existing trace path."
            )
        sessions, truncated = _sessions_from_traces(_iter_traces(path, progress), max_model_len, max_idle_seconds)
        name = path.name
        description = f"{{}} recorded sessions from {path}"
    stats = compute_stats(sessions, truncated)
    return Workload(name=name, description=description.format(len(sessions)), sessions=tuple(sessions), stats=stats)
