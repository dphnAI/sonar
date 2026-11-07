# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import builtins
import logging
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import pytest
import torch

from sonar.platforms import (
    _is_amd_zen_cpu,
    cpu_platform_plugin,
    resolve_current_platform_cls_qualname,
)


def test_is_amd_zen_cpu_detects_amd_with_avx512():
    cpuinfo = "vendor_id: AuthenticAMD\nflags: avx avx2 avx512f avx512bw"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_for_amd_without_avx512():
    cpuinfo = "vendor_id: AuthenticAMD\nflags: avx avx2"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert not _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_for_intel_with_avx512():
    cpuinfo = "vendor_id: GenuineIntel\nflags: avx avx2 avx512f"
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=cpuinfo)),
    ):
        assert not _is_amd_zen_cpu()


def test_is_amd_zen_cpu_returns_false_when_cpuinfo_missing():
    with patch("os.path.exists", return_value=False):
        assert not _is_amd_zen_cpu()


def test_cpu_target_selects_cpu_platform_from_non_cpu_wheel(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SONAR_TARGET_DEVICE", "cpu")

    with (
        patch("sonar.platforms.sonar_version_matches_substr") as version_matches,
        patch("sonar.platforms._is_amd_zen_cpu", return_value=False),
        patch("sonar.platforms.rocm_platform_plugin") as rocm_plugin,
    ):
        assert resolve_current_platform_cls_qualname() == "sonar.platforms.cpu.CpuPlatform"

    # An explicit target does not depend on the installed wheel's version
    # suffix or host accelerators (a native CI job can reuse a ROCm wheel).
    version_matches.assert_not_called()
    rocm_plugin.assert_not_called()


def test_broken_zentorch_falls_back_to_cpu_platform(caplog_sonar):
    original_import = builtins.__import__

    def import_with_broken_zentorch(name, *args, **kwargs):
        if name == "zentorch":
            raise OSError("incompatible shared library")
        return original_import(name, *args, **kwargs)

    with (
        patch("sonar.platforms.envs.SONAR_TARGET_DEVICE", "cuda"),
        patch.dict(
            "sonar.platforms.builtin_platform_plugins",
            {"cpu": cpu_platform_plugin},
            clear=True,
        ),
        patch("sonar.platforms.load_plugins_by_group", return_value={}),
        patch("sonar.platforms.sonar_version_matches_substr", return_value=True),
        patch("sonar.platforms._is_amd_zen_cpu", return_value=True),
        patch.object(builtins, "__import__", side_effect=import_with_broken_zentorch),
        caplog_sonar.at_level(logging.DEBUG, logger="sonar.platforms"),
    ):
        platform = resolve_current_platform_cls_qualname()

    assert platform == "sonar.platforms.cpu.CpuPlatform"
    assert "zentorch failed to import" in caplog_sonar.text
    assert "OSError: incompatible shared library" in caplog_sonar.text


def test_zentorch_failure_other_than_oserror_is_not_recovered(caplog_sonar):
    original_import = builtins.__import__

    def import_with_broken_zentorch(name, *args, **kwargs):
        if name == "zentorch":
            raise RuntimeError("unknown zentorch failure")
        return original_import(name, *args, **kwargs)

    with (
        patch("sonar.platforms.envs.SONAR_TARGET_DEVICE", "cuda"),
        patch.dict(
            "sonar.platforms.builtin_platform_plugins",
            {"cpu": cpu_platform_plugin},
            clear=True,
        ),
        patch("sonar.platforms.load_plugins_by_group", return_value={}),
        patch("sonar.platforms.sonar_version_matches_substr", return_value=True),
        patch("sonar.platforms._is_amd_zen_cpu", return_value=True),
        patch.object(builtins, "__import__", side_effect=import_with_broken_zentorch),
        caplog_sonar.at_level(logging.DEBUG, logger="sonar.platforms"),
    ):
        platform = resolve_current_platform_cls_qualname()

    assert platform == "sonar.platforms.interface.UnspecifiedPlatform"
    assert "RuntimeError: unknown zentorch failure" in caplog_sonar.text


def test_platform_plugin_failure_is_logged(caplog_sonar):
    def failing_plugin():
        raise RuntimeError("plugin exploded")

    with (
        patch("sonar.platforms.envs.SONAR_TARGET_DEVICE", "cuda"),
        patch.dict(
            "sonar.platforms.builtin_platform_plugins",
            {"cpu": failing_plugin},
            clear=True,
        ),
        patch("sonar.platforms.load_plugins_by_group", return_value={}),
        caplog_sonar.at_level(logging.DEBUG, logger="sonar.platforms"),
    ):
        platform = resolve_current_platform_cls_qualname()

    assert platform == "sonar.platforms.interface.UnspecifiedPlatform"
    assert "Platform plugin cpu failed during detection" in caplog_sonar.text
    assert "RuntimeError: plugin exploded" in caplog_sonar.text


@pytest.mark.parametrize("fp16_supported", [True, False])
def test_zen_cpu_supported_dtypes_track_fp16_capability(fp16_supported):
    from sonar.platforms.zen_cpu import ZenCpuPlatform

    # zentorch is absent on most CI hosts, so torch.ops.zentorch has to be
    # stood up rather than patched in place.
    zentorch_ops = SimpleNamespace(zentorch_is_fp16_supported=lambda: fp16_supported)
    with patch.object(torch.ops, "zentorch", zentorch_ops, create=True):
        dtypes = ZenCpuPlatform().supported_dtypes

    # bf16 stays first so `--dtype auto` resolves to it on every Zen part.
    assert dtypes[0] is torch.bfloat16
    assert (torch.float16 in dtypes) is fp16_supported
