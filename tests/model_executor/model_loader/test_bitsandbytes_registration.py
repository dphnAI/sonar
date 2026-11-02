# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the built-in BitsAndBytes loader available alongside loader plugins."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from aphrodite.config import LoadConfig
from aphrodite.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from aphrodite.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from aphrodite.model_executor.layers.quantization import get_quantization_config
from aphrodite.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from aphrodite.model_executor.model_loader import get_model_loader
from aphrodite.model_executor.model_loader.bitsandbytes_loader import BitsAndBytesModelLoader
from aphrodite.model_executor.model_loader.weight_utils import get_quant_config


def test_bitsandbytes_loader_registration():
    loader = get_model_loader(LoadConfig(load_format="bitsandbytes"))
    assert isinstance(loader, BitsAndBytesModelLoader)


def test_bitsandbytes_quantization_registration():
    config_cls = get_quantization_config("bitsandbytes")
    assert issubclass(config_cls, BitsAndBytesConfig)
    assert config_cls.get_name() == "bitsandbytes"


def test_bitsandbytes_inflight_config_needs_no_checkpoint_file():
    model_config = SimpleNamespace(
        quantization="bitsandbytes",
        quantization_config=None,
        hf_config=SimpleNamespace(),
        hf_overrides={},
    )
    assert get_quant_config(model_config, LoadConfig()).get_name() == "bitsandbytes"


@pytest.mark.parametrize("linear_cls", [ColumnParallelLinear, RowParallelLinear])
def test_bitsandbytes_already_sharded_tensor_is_not_sliced_again(linear_cls):
    layer = object.__new__(linear_cls)
    torch.nn.Module.__init__(layer)
    layer.tp_rank = 1
    layer.tp_size = 2
    param = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
    param.output_dim = 0
    param.input_dim = 1
    param.use_bitsandbytes_4bit = True
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    layer.weight_loader(param, weight)

    torch.testing.assert_close(param, weight)


def test_bitsandbytes_merged_shard_uses_packed_offsets():
    layer = object.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_rank = 1
    layer.tp_size = 2
    layer.output_sizes = [8, 8]
    layer.output_size = 16
    param = torch.nn.Parameter(torch.zeros(8, 1), requires_grad=False)
    param.output_dim = 0
    param.use_bitsandbytes_4bit = True

    layer.weight_loader(param, torch.ones(4, 1), 1)

    torch.testing.assert_close(param[:4], torch.zeros(4, 1))
    torch.testing.assert_close(param[4:], torch.ones(4, 1))


@pytest.mark.parametrize("loaded_size", [4, 6])
def test_bitsandbytes_moe_packed_down_projection(loaded_size):
    # Duck-typed stand-in exposing only what the BitsAndBytes path reads.
    layer = cast(
        RoutedExperts,
        SimpleNamespace(
            quant_config=None,
            quant_method=SimpleNamespace(),
            _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
        ),
    )
    param = torch.nn.Parameter(torch.zeros(1, 4, 1, dtype=torch.uint8), requires_grad=False)
    param.use_bitsandbytes_4bit = True
    weight = torch.ones(loaded_size, 1, dtype=torch.uint8)
    if loaded_size != 4:
        with pytest.raises(ValueError, match="BitsAndBytes"):
            RoutedExperts.weight_loader(layer, param, weight, "w2", "w2", 0, return_success=False)
    else:
        RoutedExperts.weight_loader(layer, param, weight, "w2", "w2", 0, return_success=False)
        torch.testing.assert_close(param[0], weight)
