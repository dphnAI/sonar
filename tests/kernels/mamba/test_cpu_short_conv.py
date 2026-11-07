# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sonar.config import (
    CompilationConfig,
    SonarConfig,
    set_current_sonar_config,
)
from sonar.forward_context import set_forward_context
from sonar.model_executor.layers.mamba.short_conv import ShortConv
from sonar.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from sonar.platforms import current_platform
from sonar.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)


@pytest.fixture(autouse=True)
def mock_dist():
    with (
        patch(
            "sonar.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "sonar.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "sonar.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        ),
        patch(
            "sonar.distributed.parallel_state.get_tp_group",
            return_value=MagicMock(rank_in_group=0),
        ),
    ):
        yield


@pytest.fixture
def sonar_config():
    # ShortConv only needs compilation_config from the current Sonar config,
    # so a minimal config (model_config=None) avoids mocking ModelConfig and the
    # associated SonarConfig validation churn.
    return SonarConfig(compilation_config=CompilationConfig())


def test_short_conv_forward_native_prefill(sonar_config):
    prefix = "test_layer"
    config = SimpleNamespace(conv_L_cache=4, conv_bias=True)
    dim = 16

    with set_current_sonar_config(sonar_config):
        layer = ShortConv(config=config, dim=dim, layer_idx=0, prefix=prefix)

    layer.to("cpu")
    # Sonar Linear layers allocate weights with torch.empty (uninitialized).
    # On ARM these come back as zero-filled pages, so in_proj output is zero and
    # the prefill state stays zero. Seed + init to make the test platform-safe.
    torch.manual_seed(0)
    for p in layer.parameters():
        torch.nn.init.normal_(p)
    dispatch_cpu_unquantized_gemm(layer.in_proj, remove_weight=False)
    dispatch_cpu_unquantized_gemm(layer.out_proj, remove_weight=False)

    num_prefills = 1
    num_prefill_tokens = 5
    query_start_loc_p = torch.tensor([0, 5], dtype=torch.int32)
    state_indices_tensor_p = torch.tensor([0], dtype=torch.int32)

    attn_metadata = ShortConvAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=1,
        query_start_loc_p=query_start_loc_p,
        has_initial_states_p=torch.tensor([False]),
        state_indices_tensor_p=state_indices_tensor_p,
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=None,
        seq_lens=torch.tensor([5]),
    )

    conv_state = torch.zeros((1, config.conv_L_cache - 1, dim))
    layer.kv_cache = (conv_state,)

    hidden_states = torch.randn((num_prefill_tokens, dim))
    output = torch.zeros_like(hidden_states)

    attn_metadata_dict = {prefix: attn_metadata}
    with set_forward_context(attn_metadata=attn_metadata_dict, sonar_config=sonar_config):
        layer.forward_native(hidden_states, output)

    assert not torch.allclose(conv_state, torch.zeros_like(conv_state))


def test_short_conv_forward_native_decode(sonar_config):
    prefix = "test_layer_decode"
    config = SimpleNamespace(conv_L_cache=4, conv_bias=True)
    dim = 16

    with set_current_sonar_config(sonar_config):
        layer = ShortConv(config=config, dim=dim, layer_idx=0, prefix=prefix)

    layer.to("cpu")
    torch.manual_seed(0)
    for p in layer.parameters():
        torch.nn.init.normal_(p)
    dispatch_cpu_unquantized_gemm(layer.in_proj, remove_weight=False)
    dispatch_cpu_unquantized_gemm(layer.out_proj, remove_weight=False)

    num_decodes = 2
    state_indices_tensor_d = torch.tensor([0, 1], dtype=torch.int32)

    attn_metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty((0,), dtype=torch.int32),
        state_indices_tensor_d=state_indices_tensor_d,
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1]),
    )

    conv_state = torch.randn((2, config.conv_L_cache - 1, dim))
    layer.kv_cache = (conv_state,)

    hidden_states = torch.randn((num_decodes, dim))
    output = torch.zeros_like(hidden_states)

    old_conv_state = conv_state.clone()

    attn_metadata_dict = {prefix: attn_metadata}
    with set_forward_context(attn_metadata=attn_metadata_dict, sonar_config=sonar_config):
        layer.forward_native(hidden_states, output)

    assert not torch.allclose(conv_state, old_conv_state)


def test_dispatch_cpu_unquantized_gemm_conv_layer():
    # Convolution layers have >2D weights; dispatch should skip them gracefully.
    # Shape/dtype are AMX-pack safe (bf16, width==4, dim % block_size == 0) so
    # the AMX prepack branch does not raise on AMX-capable CPUs.
    class MockConvLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(32, 1, 4, dtype=torch.bfloat16))
            self.bias = torch.nn.Parameter(torch.randn(32, dtype=torch.bfloat16))

    layer = MockConvLayer()
    dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    assert not hasattr(layer, "cpu_linear")
