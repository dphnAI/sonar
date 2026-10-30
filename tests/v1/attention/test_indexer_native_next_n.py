# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

from types import SimpleNamespace

import pytest

from aphrodite import envs
from aphrodite.model_executor.kernels.attention.dsa.sm120_indexer import use_sm120_dsa_indexer
from aphrodite.platforms import current_platform
from aphrodite.utils.deep_gemm import _paged_mqa_logits_schedule_slots
from aphrodite.v1.attention.backends.mla import indexer

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True, sm120_native: bool = False):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)
    monkeypatch.setattr(indexer, "use_sm120_dsa_indexer", lambda: family == 12 and not sm120_native)


@pytest.mark.parametrize(
    "family,sm120_native,expected_native",
    [
        # SM90 supports next_n=4 via 2-CTA multicast, but not next_n=3.
        (9, False, {1, 2, 4}),
        (10, False, {1, 2, 3, 4, 5, 8}),
        # Only installations with the native SM120 indexer can use multi-atom decode.
        (12, False, {1, 2}),
        (12, True, {1, 2, 3, 4, 5, 8}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, sm120_native, expected_native):
    _set_arch(monkeypatch, family, sm120_native=sm120_native)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize("cuda,deep_gemm", [(False, True), (True, False), (False, False)])
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("max_page_size", [None, 64, 128])
@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_sm120_paged_admission_requires_binary_capability(monkeypatch, tmp_path, page_size, max_page_size):
    import aphrodite.utils.deep_gemm as deep_gemm

    _set_arch(monkeypatch, 12)
    monkeypatch.setattr(envs, "APHRODITE_USE_DEEP_GEMM", True)
    include = tmp_path / "include" / "deep_gemm"
    for header in (
        "impls/sm120_fp8_mqa_logits.cuh",
        "impls/sm120_fp8_paged_mqa_logits.cuh",
        "scheduler/sm120_paged_mqa_logits.cuh",
    ):
        path = include / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    module = SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"),
        fp8_fp4_mqa_logits=lambda: None,
        fp8_fp4_paged_mqa_logits=lambda: None,
        get_paged_mqa_logits_metadata=lambda: None,
    )
    if max_page_size is not None:
        module.get_sm120_fp8_paged_mqa_max_page_size = lambda: max_page_size
    monkeypatch.setattr(deep_gemm, "_import_deep_gemm", lambda: module)
    use_sm120_dsa_indexer.cache_clear()
    try:
        expected_fallback = page_size == 256 or (page_size == 128 and max_page_size != 128)
        assert use_sm120_dsa_indexer(page_size) == expected_fallback
    finally:
        use_sm120_dsa_indexer.cache_clear()
