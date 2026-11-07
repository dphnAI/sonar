# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

from sonar.envs import get_sonar_port


def test_get_sonar_port_not_set():
    """Test when SONAR_PORT is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_sonar_port() is None


def test_get_sonar_port_valid():
    """Test when SONAR_PORT is set to a valid integer."""
    with patch.dict(os.environ, {"SONAR_PORT": "5678"}, clear=True):
        assert get_sonar_port() == 5678


def test_get_sonar_port_invalid():
    """Test when SONAR_PORT is set to a non-integer value."""
    with (
        patch.dict(os.environ, {"SONAR_PORT": "abc"}, clear=True),
        pytest.raises(ValueError, match="must be a valid integer"),
    ):
        get_sonar_port()


def test_get_sonar_port_uri():
    """Test when SONAR_PORT is set to a URI."""
    with (
        patch.dict(os.environ, {"SONAR_PORT": "tcp://localhost:5678"}, clear=True),
        pytest.raises(ValueError, match="appears to be a URI"),
    ):
        get_sonar_port()
