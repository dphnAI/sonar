# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from sonar.config.speculative import resolve_draft_kv_cache_dtype


def _spec_config(*, draft_uses_mla: bool, kv_cache_dtype: str | None = None):
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        draft_model_config=SimpleNamespace(use_mla=draft_uses_mla),
    )


def test_non_mla_draft_does_not_inherit_packed_mla_cache():
    spec_config = _spec_config(draft_uses_mla=False)

    assert resolve_draft_kv_cache_dtype(spec_config, "fp8_ds_mla") == "fp8_e4m3"


def test_mla_draft_inherits_packed_mla_cache():
    spec_config = _spec_config(draft_uses_mla=True)

    assert resolve_draft_kv_cache_dtype(spec_config, "fp8_ds_mla") == "fp8_ds_mla"


def test_explicit_draft_cache_dtype_takes_precedence():
    spec_config = _spec_config(draft_uses_mla=False, kv_cache_dtype="bfloat16")

    assert resolve_draft_kv_cache_dtype(spec_config, "fp8_ds_mla") == "bfloat16"
