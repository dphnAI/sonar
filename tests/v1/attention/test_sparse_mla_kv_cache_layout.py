# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA allocation and warmup must agree on physical addressing."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sonar.config import CacheConfig
from sonar.model_executor.layers.attention.mla_attention import MLAAttention
from sonar.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from sonar.platforms import current_platform
from sonar.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from sonar.v1.attention.backends.mla.sparse_utils import (
    _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL,
    flat_kv_row_view,
)
from sonar.v1.attention.backends.registry import AttentionBackendEnum
from sonar.v1.attention.backends.utils import get_supported_kv_cache_layouts
from sonar.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from sonar.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from sonar.v1.kv_cache_layout import KVCacheLayout
from sonar.v1.worker.utils import allocate_kv_cache, customize_attention_spec

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("mha_method", ["absent", "unimplemented", "implemented"])
def test_sparse_mha_requires_an_implemented_forward(mha_method):
    from sonar.model_executor.layers.attention.mla_attention import MLAAttentionImpl

    def forward_mha(*args, **kwargs):
        pytest.fail("capability selection must not execute attention")

    methods = {
        "absent": {},
        "unimplemented": {"forward_mha": MLAAttentionImpl.forward_mha},
        "implemented": {"forward_mha": forward_mha},
    }
    impl = type("SparseImpl", (), methods[mha_method])()
    layer = SimpleNamespace(
        impl=impl,
        hisparse_cache=None,
        prefill_backend=None,
        _sonar_config=SimpleNamespace(attention_config=SimpleNamespace(sparse_mla_force_mqa=False)),
    )
    metadata = SimpleNamespace(prefill=SimpleNamespace(use_dense_mha=True))

    assert MLAAttention._use_sparse_mha(layer, metadata) is (mha_method == "implemented")


def test_sparse_mqa_only_backend_declares_no_dense_prefill():
    from sonar.v1.attention.backend import SparseMLAAttentionImpl
    from sonar.v1.attention.backends.mla.sm89_mla_sparse import Sm89MLASparseImpl

    assert not SparseMLAAttentionImpl.supports_dense_mha_prefill
    assert not Sm89MLASparseImpl.supports_dense_mha_prefill
    assert Sm89MLASparseImpl.get_fp8_ds_mla_row_bytes(None) == 656


@pytest.mark.parametrize("allow_local_prefill", [False, True])
@pytest.mark.parametrize(
    "query_lens,seq_lens,expected_counts,fresh_prefill",
    [
        ([1, 1], [9, 12], (2, 0, 2), False),
        ([1, 1, 4, 3], [9, 12, 4, 3], (2, 2, 2), True),
        ([1, 1, 4, 3], [9, 12, 8, 3], (2, 2, 2), False),
    ],
    ids=["decode-only", "mixed-fresh", "mixed-continuation"],
)
def test_sm89_metadata_retains_wrapper_counts(
    monkeypatch, allow_local_prefill, query_lens, seq_lens, expected_counts, fresh_prefill
):
    from sonar.v1.attention.backend import CommonAttentionMetadata
    from sonar.v1.attention.backends.mla import sm89_mla_sparse as sm89

    # Exercise the host metadata path without CUDA allocation or pinned memory.
    monkeypatch.setattr(sm89, "np_to_pinned_tensor", torch.from_numpy)
    starts = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32, device="cpu")
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    num_tokens = sum(query_lens)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=lengths,
        seq_lens_cpu_upper_bound=lengths,
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(len(query_lens), 1, dtype=torch.int32, device="cpu"),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device="cpu"),
    )
    builder = object.__new__(sm89.Sm89MLASparseMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.allow_local_prefill = allow_local_prefill
    builder.reorder_batch_threshold = 1
    builder.req_id_per_token_buffer = torch.full((num_tokens + 3,), -1, dtype=torch.int32, device="cpu")
    builder.local_prefill_slot_buffer = torch.arange(num_tokens, dtype=torch.int64, device="cpu")
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.topk_tokens = 2048
    builder.sonar_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=64))

    metadata = builder.build(0, common)

    assert metadata.seq_lens is lengths
    assert (metadata.num_decodes, metadata.num_prefills, metadata.num_decode_tokens) == expected_counts
    assert metadata.query_start_loc is starts
    assert metadata.slot_mapping is common.slot_mapping
    assert metadata.block_table is common.block_table_tensor
    assert metadata.req_id_per_token.tolist() == [req for req, count in enumerate(query_lens) for _ in range(count)]
    assert builder.req_id_per_token_buffer[num_tokens:].tolist() == [0, 0, 0]
    if allow_local_prefill and fresh_prefill:
        local = metadata.prefill_local
        assert local is not None
        assert local.workspace_starts.tolist() == [0, 4]
        assert local.workspace_request_ids.tolist() == [0, 0, 0, 0, 1, 1, 1]
        assert local.shadow_slot_mapping.tolist() == list(range(num_tokens - metadata.num_decode_tokens))
    else:
        assert metadata.prefill_local is None


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA/ROCm only"),
        ),
    ],
)
@pytest.mark.parametrize("layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC, KVCacheLayout.LBNHC])
@pytest.mark.parametrize(
    "backend,cache_dtype,sm100",
    [
        (AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "auto", True),
        (AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "fp8", True),
        pytest.param(
            AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE_FA4,
            "auto",
            True,
            marks=pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only"),
        ),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "auto", True),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "fp8_ds_mla", False),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "fp8_ds_mla", True),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "nvfp4_ds_mla", True),
    ],
)
def test_allocation_and_warmup_follow_addressing_mode(monkeypatch, layout, backend, cache_dtype, sm100, device):
    backend = backend.get_class()
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda family: sm100)
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(hisparse_config=None),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )
    config.cache_config.kv_cache_layout = layout.name
    layer = SimpleNamespace(
        _sonar_config=config,
        attn_backend=backend,
        impl=object.__new__(backend.get_impl_cls()),
        kv_cache_dtype=cache_dtype,
        head_size=576,
        topk_buffer_width=None,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
    )
    layer._uses_flat_kv_cache = MethodType(MLAAttention._uses_flat_kv_cache, layer)
    spec = customize_attention_spec(backend, MLAAttention.get_kv_cache_spec(layer, config))
    indexer = SimpleNamespace(cache_config=config.cache_config, head_dim=132, dtype=torch.uint8)
    index_spec = DeepseekV32IndexerCache.get_kv_cache_spec(indexer, config)
    assert index_spec.block_stride_alignment is None
    assert layout in get_supported_kv_cache_layouts(
        # indexer backend has no impl class
        [backend, DeepseekV32IndexerBackend]  # type: ignore[type-abstract]
    )
    specs = {"mla": spec, "indexer": index_spec}
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs))
    cache = get_kv_cache_config_from_groups(config, [group], 2**20)
    views = allocate_kv_cache(cache, torch.device(device), layout)
    register = MagicMock()
    monkeypatch.setattr(_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL, "register_warmup", register)
    MLAAttention.bind_kv_cache(layer, views["mla"])
    DeepseekV32IndexerCache.bind_kv_cache(indexer, views["indexer"])
    assert indexer.kv_cache.stride(0) == views["indexer"].stride(0)
    alignment = spec.get_block_stride_alignment()
    assert views["mla"].stride(0) * views["mla"].element_size() % alignment == 0
    if not layer._uses_flat_kv_cache():
        register.assert_not_called()
    else:
        rows, stride = flat_kv_row_view(layer.kv_cache, 64)
        register.assert_called_once_with(config, block_stride_rows=stride)
        layer.kv_cache[1, 0].fill_(3)
        torch.testing.assert_close(rows[stride], layer.kv_cache[1, 0])
    if alignment == 1:
        assert cache.kv_cache_tensors[0].size == cache.num_blocks * sum(item.page_size_bytes for item in specs.values())
