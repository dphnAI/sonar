# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import transformers.configuration_utils as hf_configuration_utils
from transformers import PretrainedConfig

import aphrodite.transformers_utils.config as config_module
from aphrodite.transformers_utils.config import (
    get_config_parser,
    register_config_parser,
)
from aphrodite.transformers_utils.config_parser_base import ConfigParserBase


@register_config_parser("custom_config_parser")
class CustomConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        raise NotImplementedError


def test_register_config_parser():
    assert isinstance(get_config_parser("custom_config_parser"), CustomConfigParser)


def test_invalid_config_parser():
    with pytest.raises(ValueError):

        @register_config_parser("invalid_config_parser")
        class InvalidConfigParser:
            pass


def test_patch_hf_transformers_allowed_layer_types(monkeypatch):
    extra_layer_type = "deepseek_sparse_attention"
    hf_layer_types = tuple(
        layer_type for layer_type in hf_configuration_utils.ALLOWED_LAYER_TYPES if layer_type != extra_layer_type
    )
    aphrodite_layer_types = tuple(
        layer_type for layer_type in config_module.ALLOWED_LAYER_TYPES if layer_type != extra_layer_type
    )
    monkeypatch.setattr(hf_configuration_utils, "ALLOWED_LAYER_TYPES", hf_layer_types)
    monkeypatch.setattr(config_module, "ALLOWED_LAYER_TYPES", aphrodite_layer_types)

    config_module._patch_hf_transformers_allowed_layer_types((extra_layer_type,))

    assert extra_layer_type in hf_configuration_utils.ALLOWED_LAYER_TYPES
    assert config_module.ALLOWED_LAYER_TYPES is hf_configuration_utils.ALLOWED_LAYER_TYPES
