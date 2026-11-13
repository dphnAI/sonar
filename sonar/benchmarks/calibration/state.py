# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibration progress on disk, so an interrupted run can resume."""

import dataclasses
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonar.benchmarks.calibration.fit import CostFit
from sonar.benchmarks.calibration.profile import ProfileMeasurement
from sonar.benchmarks.calibration.search import TrialMeasurement

STATE_FILE = "state.json"
STATE_VERSION = 1


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class TrialRecord:
    index: int
    coefficients: tuple[float, float]
    sessions: int
    started: str
    finished: str
    directory: str
    measurement: TrialMeasurement


@dataclass(slots=True)
class CalibrationState:
    invocation: dict[str, Any]
    fingerprint: dict[str, Any]
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    profile_schedule: list[int] | None = None
    profile: list[ProfileMeasurement] = field(default_factory=list)
    fit: CostFit | None = None
    max_model_len: int | None = None
    kv_capacity_tokens: int | None = None
    sessions: int | None = None
    startup_seconds: list[float] = field(default_factory=list)
    trials: list[TrialRecord] = field(default_factory=list)
    finished: bool = False

    @property
    def measurements(self) -> list[TrialMeasurement]:
        return [trial.measurement for trial in self.trials]

    def save(self, directory: Path) -> None:
        self.updated = _now()
        data = {"version": STATE_VERSION, **dataclasses.asdict(self)}
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / (STATE_FILE + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, directory / STATE_FILE)

    @classmethod
    def load(cls, directory: Path) -> "CalibrationState":
        path = directory / STATE_FILE
        if not path.exists():
            raise FileNotFoundError(f"No calibration to resume in {directory} ({STATE_FILE} is missing).")
        data = json.loads(path.read_text())
        if data.pop("version", None) != STATE_VERSION:
            raise ValueError(f"{path} was written by an incompatible Sonar version; start a new calibration.")
        data["profile"] = [ProfileMeasurement(**entry) for entry in data["profile"]]
        if data["fit"] is not None:
            fit = data["fit"]
            fit["ratio_interval"] = tuple(fit["ratio_interval"])
            data["fit"] = CostFit(**fit)
        trials = []
        for entry in data["trials"]:
            measurement = entry["measurement"]
            measurement["interval_output_tokens_per_second"] = tuple(measurement["interval_output_tokens_per_second"])
            entry["measurement"] = TrialMeasurement(**measurement)
            entry["coefficients"] = tuple(entry["coefficients"])
            trials.append(TrialRecord(**entry))
        data["trials"] = trials
        return cls(**data)


def fingerprint_differences(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable differences between two configuration fingerprints."""
    differences = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            differences.append(f"{key}: saved {saved.get(key)!r}, now {current.get(key)!r}")
    return differences
