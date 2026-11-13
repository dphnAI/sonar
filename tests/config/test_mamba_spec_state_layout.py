# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig

from sonar.config import CacheConfig, KVTransferConfig, ModelConfig, SonarConfig, SpeculativeConfig
from sonar.model_executor.layers.mamba.abstract import MambaBase
from sonar.platforms import current_platform
from sonar.v1.attention.backends.registry import MambaAttentionBackendEnum
from sonar.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def local_model(tmp_path):
    config = OPTConfig(
        hidden_size=32,
        word_embed_proj_dim=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        ffn_dim=64,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["OPTForCausalLM"],
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def make_config(model, *, speculative=True, connector=None):
    return SonarConfig(
        model_config=ModelConfig(model=model, skip_tokenizer_init=True),
        cache_config=CacheConfig(mamba_spec_state_layout="compact"),
        speculative_config=(SpeculativeConfig(method="ngram_gpu", num_speculative_tokens=2) if speculative else None),
        kv_transfer_config=connector,
    )


def test_compact_layout_requires_speculative_decoding(local_model):
    with pytest.raises(ValueError, match="requires speculative decoding"):
        make_config(local_model, speculative=False)


def test_compact_layout_requires_a_supporting_model(local_model):
    with pytest.raises(ValueError, match="not supported for architecture 'OPTForCausalLM'"):
        make_config(local_model)


def test_compact_layout_requires_cuda(local_model, monkeypatch):
    monkeypatch.setattr(ModelConfig, "supports_compact_spec_state", property(lambda self: True))
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        make_config(local_model)


def test_compact_layout_rejects_incompatible_connector(local_model, monkeypatch):
    monkeypatch.setattr(ModelConfig, "supports_compact_spec_state", property(lambda self: True))
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    connector = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both")
    with pytest.raises(ValueError, match="does not support KV connectors other than MooncakeStore"):
        make_config(local_model, connector=connector)


class _Layer(MambaBase):
    mamba_type = MambaAttentionBackendEnum.LINEAR

    def get_state_shape(self):
        return ((4,),)

    def get_state_dtype(self):
        return (torch.float32,)


class _CompactLayer(_Layer):
    supports_compact_spec_state = True


def test_compact_allocation_requires_every_layer_to_support_it():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            use_replayssm=False,
            use_kda_recoverssm=True,
            mamba_block_size=16,
            mamba_page_size_padded=None,
            mamba_cache_mode="align",
        ),
        num_speculative_tokens=3,
    )
    spec = _CompactLayer().get_kv_cache_spec(config)
    assert isinstance(spec, MambaSpec)
    assert spec.num_speculative_blocks == 0
    with pytest.raises(ValueError, match="_Layer does not support"):
        _Layer().get_kv_cache_spec(config)
    config.cache_config.use_replayssm = True
    spec = _Layer().get_kv_cache_spec(config)
    assert isinstance(spec, MambaSpec)
    assert spec.num_speculative_blocks == 0
    config.cache_config.use_replayssm = False
    config.cache_config.use_kda_recoverssm = False
    spec = _Layer().get_kv_cache_spec(config)
    assert isinstance(spec, MambaSpec)
    assert spec.num_speculative_blocks == 3
