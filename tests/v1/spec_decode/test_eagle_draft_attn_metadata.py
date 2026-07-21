# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EAGLE speculator's draft attention metadata builder."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from aphrodite.v1.worker.gpu.spec_decode import speculator as base_speculator
from aphrodite.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


def _make_fake_speculator(
    *,
    max_num_reqs: int = 8,
    max_num_tokens: int = 16,
    max_model_len: int = 1024,
    draft_max_seq_len: int = 1024,
) -> SimpleNamespace:
    fake_input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
    )
    fake_block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(max_num_reqs, 4, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, max_num_tokens, dtype=torch.int64),
    )
    return SimpleNamespace(
        arange=torch.arange(max_num_reqs + 1, dtype=torch.int32, device="cpu"),
        block_tables=fake_block_tables,
        input_buffers=fake_input_buffers,
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        max_model_len=max_model_len,
        draft_max_seq_len=draft_max_seq_len,
    )


def _run_build(fake, *, num_reqs, num_reqs_padded, num_tokens_padded, base, step):
    captured: dict[str, object] = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {}

    with patch.object(base_speculator, "build_attn_metadata", fake_build_attn_metadata):
        EagleSpeculator._build_draft_attn_metadata(
            fake,  # type: ignore[arg-type]
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            seq_lens_cpu_upper_bound=base,
            step=step,
        )
    return captured


def test_build_draft_attn_metadata_sets_seq_lens_cpu_upper_bound():
    fake = _make_fake_speculator()
    base = torch.tensor([100, 200, 300, 0], dtype=torch.int32)

    captured = _run_build(fake, num_reqs=3, num_reqs_padded=4, num_tokens_padded=4, base=base, step=2)

    bound = captured["seq_lens_cpu_upper_bound"]
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (4,)
    assert bound.device.type == "cpu"
    assert bound.dtype == torch.int32
    assert torch.equal(bound, torch.tensor([102, 202, 302, 0], dtype=torch.int32))


def test_build_draft_attn_metadata_handles_zero_unpadded_reqs():
    fake = _make_fake_speculator()
    base = torch.zeros(2, dtype=torch.int32)

    captured = _run_build(fake, num_reqs=0, num_reqs_padded=2, num_tokens_padded=2, base=base, step=1)

    bound = captured["seq_lens_cpu_upper_bound"]
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (2,)
    assert torch.equal(bound, torch.zeros(2, dtype=torch.int32))


def test_build_draft_attn_metadata_clamps_to_max_model_len():
    fake = _make_fake_speculator(max_model_len=1024)
    base = torch.tensor([1023, 500], dtype=torch.int32)

    captured = _run_build(fake, num_reqs=2, num_reqs_padded=2, num_tokens_padded=2, base=base, step=3)

    bound = captured["seq_lens_cpu_upper_bound"]
    assert torch.equal(bound, torch.tensor([1024, 503], dtype=torch.int32))
