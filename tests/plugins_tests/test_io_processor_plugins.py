# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from aphrodite.config import AphroditeConfig
from aphrodite.inputs import PromptType
from aphrodite.outputs import PoolingRequestOutput
from aphrodite.plugins.io_processors import get_io_processor
from aphrodite.plugins.io_processors.interface import IOProcessor
from aphrodite.renderers import BaseRenderer


class DummyIOProcessor(IOProcessor):
    """Minimal IOProcessor used as the target of the mocked plugin entry point."""

    def pre_process(
        self,
        prompt: object,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> object:
        raise NotImplementedError


@pytest.fixture
def my_plugin_entry_points():
    """Patch importlib.metadata.entry_points to expose a single 'my_plugin'
    entry point backed by DummyIOProcessor, exercising the full plugin-loading
    code path: entry_points → plugin.load() → func() →
    resolve_obj_by_qualname → IOProcessor.__init__."""
    qualname = f"{DummyIOProcessor.__module__}.{DummyIOProcessor.__qualname__}"
    ep = MagicMock()
    ep.name = "my_plugin"
    ep.value = qualname
    ep.load.return_value = lambda: qualname
    with patch("importlib.metadata.entry_points", return_value=[ep]):
        yield


def test_loading_missing_plugin():
    aphrodite_config = AphroditeConfig()
    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(
            aphrodite_config, renderer=renderer, plugin_from_init="wrong_plugin"
        )


def test_loading_plugin(my_plugin_entry_points):
    # Plugin name supplied via plugin_from_init.
    aphrodite_config = MagicMock(spec=AphroditeConfig)
    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(
        aphrodite_config, renderer=renderer, plugin_from_init="my_plugin"
    )

    assert isinstance(result, DummyIOProcessor)


def test_loading_missing_plugin_from_model_config():
    # Build a mock AphroditeConfig whose hf_config advertises a plugin name,
    # exercising the model-config code path without loading a real model.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "wrong_plugin"}

    aphrodite_config = MagicMock(spec=AphroditeConfig)
    aphrodite_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(aphrodite_config, renderer=renderer)


def test_loading_plugin_from_model_config(my_plugin_entry_points):
    # Plugin name supplied via the model's hf_config.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "my_plugin"}

    aphrodite_config = MagicMock(spec=AphroditeConfig)
    aphrodite_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(aphrodite_config, renderer=renderer)

    assert isinstance(result, DummyIOProcessor)
