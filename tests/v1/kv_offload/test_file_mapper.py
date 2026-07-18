# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from aphrodite.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from aphrodite.v1.kv_offload.base import (
    OffloadingSpec,
    make_offload_key,
)
from aphrodite.v1.kv_offload.file_mapper import FileMapper

# ---------------------------------------------------------------------------
# Shared mocks (mirrors test_fs_tier.py pattern)
# ---------------------------------------------------------------------------

_MOCK_APHRODITE_CONFIG = MagicMock()
_MOCK_APHRODITE_CONFIG.model_config.model = "test-model"
_MOCK_APHRODITE_CONFIG.cache_config.block_size = 16
_MOCK_APHRODITE_CONFIG.cache_config.cache_dtype = "torch.float32"
_MOCK_APHRODITE_CONFIG.parallel_config.tensor_parallel_size = 1
_MOCK_APHRODITE_CONFIG.parallel_config.pipeline_parallel_size = 1
_MOCK_APHRODITE_CONFIG.parallel_config.prefill_context_parallel_size = 1
_MOCK_APHRODITE_CONFIG.parallel_config.decode_context_parallel_size = 1
_MOCK_APHRODITE_CONFIG.parallel_config.rank = 0

_MOCK_KV_CACHE_CONFIG = MagicMock()
_MOCK_KV_CACHE_CONFIG.kv_cache_groups = []

_MOCK_OFFLOADING_SPEC = MagicMock(spec=OffloadingSpec)
_MOCK_OFFLOADING_SPEC.config = SimpleNamespace(
    model=SimpleNamespace(name="test-model", dtype="float32"),
    cache=SimpleNamespace(tokens_per_hash=16),
    parallel=SimpleNamespace(
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        is_parallelism_agnostic=True,
    ),
    groups=(),
)
_MOCK_OFFLOADING_SPEC.blocks_per_chunk = 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mapper_from_offloading_spec(**kwargs) -> FileMapper:
    """Helper to create FileMapper with customizable mock config."""
    mock_offloading_spec = MagicMock(spec=OffloadingSpec)
    mock_offloading_spec.config = SimpleNamespace(
        model=SimpleNamespace(
            name=kwargs.get("model_name", "test-model"),
            dtype=kwargs.get("dtype", "float16"),
        ),
        cache=SimpleNamespace(tokens_per_hash=kwargs.get("tokens_per_hash", 16)),
        parallel=SimpleNamespace(
            tp_size=kwargs.get("tp_size", 1),
            pp_size=kwargs.get("pp_size", 1),
            pcp_size=kwargs.get("pcp_size", 1),
            dcp_size=kwargs.get("dcp_size", 1),
            rank=kwargs.get("rank", 0),
            is_parallelism_agnostic=kwargs.get("is_parallelism_agnostic", True),
        ),
        groups=tuple(
            SimpleNamespace(
                tokens_per_block=group.kv_cache_spec.block_size,
                layer_names=tuple(group.layer_names),
            )
            for group in kwargs.get("kv_cache_groups", [])
        ),
    )
    mock_offloading_spec.blocks_per_chunk = kwargs.get("blocks_per_chunk", 1)

    return FileMapper.from_offloading_spec(
        root_dir=kwargs.get("root_dir", "/tmp/cache"),
        offloading_spec=mock_offloading_spec,
        blocks_per_file=mock_offloading_spec.blocks_per_chunk,
        parallel_agnostic=kwargs.get("parallel_agnostic", False),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_file_name_full_structure():
    """
    Path must match: <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin

    Concretely:
      - The segment immediately after base_path must end with `_r0`
      - The next segment is the first 3 hex chars of the block hash
      - The next segment is <2 hex chars>_g<group_idx>
      - The final segment is <full hash hex>.bin
    """
    rank = 3
    group_idx = 2
    block_hash = bytes(range(8))  # deterministic, non-zero bytes
    fm = make_mapper_from_offloading_spec(rank=rank)
    key = make_offload_key(block_hash, group_idx)
    path = fm.get_file_name(key)

    expected_path = "/tmp/cache/test-model_588656ebcc66_r3/000/10_g2/0001020304050607.bin"
    assert path == expected_path


def test_get_run_config_fields():
    fm = make_mapper_from_offloading_spec(
        model_name="my-model",
        dtype="bfloat16",
        tp_size=2,
    )
    cfg = fm.get_run_config()
    assert cfg == {
        "model_name": "my-model",
        "tokens_per_hash": 16,
        "blocks_per_file": 1,
        "tp_size": 2,
        "pp_size": 1,
        "pcp_size": 1,
        "dcp_size": 1,
        "dtype": "bfloat16",
        "kv_cache_groups": [],
        "inference_engine": "aphrodite",
    }


def test_get_config_file_path():
    fm = make_mapper_from_offloading_spec()
    config_path = fm.get_config_file_path()
    assert config_path == f"{fm.base_path}/config.json"


# ---------------------------------------------------------------------------
# parallel_agnostic: honored only for a single non-MLA full-attention group
# ---------------------------------------------------------------------------


def _full_attention_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=["layer0"],
        kv_cache_spec=FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=128, dtype=torch.float32),
    )


def _sliding_window_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=["layer0"],
        kv_cache_spec=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.float32,
            sliding_window=128,
        ),
    )


def test_parallel_agnostic_enabled_for_single_full_attention():
    # tp/rank are collapsed out of the namespace so the cache is shared
    # across tensor-parallel sizes.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        kv_cache_groups=[_full_attention_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 1
    assert fm.rank == 0


def test_parallel_agnostic_disabled_for_multiple_groups():
    # More than one KV-cache group (hybrid model) => keep per-layout namespacing.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        kv_cache_groups=[_full_attention_group(), _full_attention_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2


def test_parallel_agnostic_disabled_for_non_full_attention():
    # Single group but not full attention (sliding window) => keep namespacing.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        kv_cache_groups=[_sliding_window_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2


def test_parallel_agnostic_excludes_mla():
    # MLA latent KV is replicated per rank, so its offloaded blocks are not
    # parallelism-invariant: the opt-in must not collapse tp/rank.
    group = KVCacheGroupSpec(
        layer_names=["layer0"],
        kv_cache_spec=MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=576, dtype=torch.float32),
    )
    fm = make_mapper_from_offloading_spec(tp_size=2, rank=1, kv_cache_groups=[group], parallel_agnostic=True)
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1


def test_parallel_agnostic_disabled_on_v2_model_runner():
    # V2's KV layout is not known to be parallelism-invariant: don't collapse.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        kv_cache_groups=[_full_attention_group()],
        use_v2_model_runner=True,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
