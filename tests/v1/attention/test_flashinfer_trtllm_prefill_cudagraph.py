# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: TRTLLM prefill metadata must be CUDA-graph safe."""

import pytest

from sonar.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from sonar.config import SpeculativeConfig, set_current_sonar_config
from sonar.v1.attention.backends import flashinfer as flashinfer_backend
from sonar.v1.attention.backends.flashinfer import (
    FlashInferMetadataBuilder,
    TRTLLMPrefill,
)
from sonar.v1.attention.backends.utils import PerLayerParameters
from sonar.v1.kv_cache_interface import FullAttentionSpec
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_sonar_config,
)


def test_trtllm_prefill_query_offsets_use_a_persistent_buffer(monkeypatch):
    """A uniform batch wider than reorder_batch_threshold (e.g. the draft-model
    prefill, num_speculative_tokens + 2 tokens per request) takes the TRTLLM
    prefill path and can be captured in a FULL CUDA graph. The graph keeps the
    address of cum_seq_lens_q from capture time, so every build must write the
    offsets into the same buffer instead of allocating a new tensor.
    """
    sonar_config = create_sonar_config(max_model_len=1024, max_num_seqs=8)
    sonar_config.speculative_config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    monkeypatch.setattr(flashinfer_backend, "use_trtllm_attention", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False)
        },
    )

    block_size = 16
    kv_cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=sonar_config.model_config.get_num_kv_heads(sonar_config.parallel_config),
        head_size=sonar_config.model_config.get_head_size(),
        dtype=sonar_config.model_config.dtype,
    )
    device = torch.device("cuda")
    with set_current_sonar_config(sonar_config):
        builder = FlashInferMetadataBuilder(kv_cache_spec, ["layer.0"], sonar_config, device)
    query_len = builder.reorder_batch_threshold + 1

    # Keep every build's offsets alive so a fresh allocation cannot reuse the
    # previous address.
    offsets = []
    for num_reqs in (4, 3):
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[64] * num_reqs, query_lens=[query_len] * num_reqs),
            block_size,
            device,
        )
        metadata = builder.build(0, common)
        assert metadata.num_decodes == 0
        assert isinstance(metadata.prefill, TRTLLMPrefill)
        cum_seq_lens_q = metadata.prefill.cum_seq_lens_q
        expected = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
        torch.testing.assert_close(cum_seq_lens_q.cpu(), expected)
        offsets.append(cum_seq_lens_q)
    pointers = {t.data_ptr() for t in offsets}
    assert len(pointers) == 1, "cum_seq_lens_q moved between builds"
