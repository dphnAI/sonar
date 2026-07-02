# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

import aphrodite.envs as envs


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode(monkeypatch: pytest.MonkeyPatch):
    """Automatically enable batch invariant kernel overrides for all tests."""
    monkeypatch.setattr(envs, "APHRODITE_BATCH_INVARIANT", True)
    monkeypatch.setenv("APHRODITE_BATCH_INVARIANT", "1")
