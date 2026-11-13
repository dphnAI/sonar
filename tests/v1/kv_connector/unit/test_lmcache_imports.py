# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Sonar project

import importlib.util
import py_compile
import sys
from typing import Any

import pytest

from sonar.distributed.kv_transfer.kv_connector.v1.lmcache_integration.imports import _LMCacheLoader


@pytest.mark.parametrize("cached", [False, True])
def test_lmcache_imports_share_native_engine_state(tmp_path, monkeypatch, cached):
    from sonar import envs
    from sonar.v1.core import kv_cache_utils

    monkeypatch.setenv("SONAR_RPC_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"native-sonar-hash-state", raising=False)
    source = tmp_path / "adapter.py"
    source.write_text(
        "from vllm.v1.core import kv_cache_utils\n"
        "import vllm.envs as envs\n"
        "paths = ['vllm.v1.core.kv_cache_utils']\n"
        "dynamic = __import__(paths[0], fromlist=['NONE_HASH'])\n"
        "state = (kv_cache_utils.NONE_HASH, dynamic.NONE_HASH, envs.VLLM_RPC_BASE_PATH)\n"
        "engine_type = 'vllm'\n"
        "adapter_path = 'lmcache.integration.vllm'\n"
    )
    original = source.read_bytes()
    bytecode = None
    if cached:
        bytecode_path = py_compile.compile(str(source), doraise=True)
        assert bytecode_path is not None
        with open(bytecode_path, "rb") as f:
            bytecode = f.read()
    before = {name for name in sys.modules if name == "vllm" or name.startswith("vllm.")}
    loader = _LMCacheLoader("lmcache_test_adapter", str(source))
    spec = importlib.util.spec_from_file_location(loader.name, source, loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    assert module.state == (b"native-sonar-hash-state", b"native-sonar-hash-state", envs.SONAR_RPC_BASE_PATH)
    # These identifiers belong to LMCache's protocol and package, not Sonar imports.
    assert module.engine_type == "vllm"
    assert module.adapter_path == "lmcache.integration.vllm"
    assert {name for name in sys.modules if name == "vllm" or name.startswith("vllm.")} == before
    assert source.read_bytes() == original
    if bytecode is not None:
        with open(bytecode_path, "rb") as f:
            assert f.read() == bytecode


def test_lmcache_config_aliases_use_native_context(tmp_path):
    from sonar.config import SonarConfig, set_current_sonar_config

    source = tmp_path / "context.py"
    source.write_text(
        "import vllm.config\n"
        "from vllm.config import VllmConfig as Config, get_current_vllm_config\n"
        "def get_config():\n"
        "    config = get_current_vllm_config()\n"
        "    assert isinstance(config, Config)\n"
        "    assert isinstance(config.parallel_config, vllm.config.ParallelConfig)\n"
        "    return config\n"
    )
    loader = _LMCacheLoader("lmcache_test_context", str(source))
    spec = importlib.util.spec_from_file_location(loader.name, source, loader=loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    config = SonarConfig()
    with set_current_sonar_config(config):
        assert module.get_config() is config


@pytest.mark.parametrize(
    "module_name",
    [
        "lmcache.integration.vllm.vllm_v1_adapter",
        "lmcache.inprocess.integration.vllm.vllm_v1_adapter",
    ],
)
def test_lmcache_scheduler_contract_uses_v2_prefill(tmp_path, module_name):
    from types import SimpleNamespace

    source = tmp_path / "adapter.py"
    source.write_text(
        "def from_new_request(new_request, num_tokens_to_compute):\n"
        "    return new_request.prompt_token_ids[:num_tokens_to_compute].copy()\n"
        "def build_connector_meta(cached_reqs, req_id):\n"
        "    if hasattr(cached_reqs, 'resumed_req_ids'):\n"
        "        preempted = req_id in cached_reqs.resumed_req_ids\n"
        "    elif hasattr(cached_reqs, 'resumed_from_preemption'):\n"
        "        preempted = cached_reqs.resumed_from_preemption[0]\n"
        "    else:\n"
        "        raise AttributeError('unsupported scheduler')\n"
        "    return preempted\n"
    )
    original = source.read_bytes()
    bytecode_path = py_compile.compile(str(source), doraise=True)
    with open(bytecode_path, "rb") as stream:
        bytecode = stream.read()
    namespace: dict[str, Any] = {}
    exec(_LMCacheLoader(module_name, str(source)).get_code(module_name), namespace)
    request = SimpleNamespace(prompt_token_ids=[1, 2], prefill_token_ids=[1, 2, 3, 4])
    assert namespace["from_new_request"](request, 4) == [1, 2, 3, 4]
    assert namespace["build_connector_meta"](SimpleNamespace(), "r") is False
    assert source.read_bytes() == original
    with open(bytecode_path, "rb") as stream:
        assert stream.read() == bytecode


def test_lmcache_mp_appends_all_cached_blocks(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    module_name = "lmcache.integration.vllm.lmcache_mp_connector"
    source = tmp_path / "mp.py"
    source.write_text(
        "def _process_cached_requests(cached_reqs, request_id, request_tracker, new_block_ids):\n"
        "    if request_id not in cached_reqs.resumed_req_ids:\n"
        "        request_tracker.append_block_ids(new_block_ids, 2)\n"
    )
    namespace: dict[str, Any] = {}
    exec(_LMCacheLoader(module_name, str(source)).get_code(module_name), namespace)
    tracker = Mock()
    namespace["_process_cached_requests"](SimpleNamespace(), "r", tracker, ([3, 4],))
    tracker.append_block_ids.assert_called_once_with(([3, 4],), 2)


@pytest.mark.parametrize(
    "source_text",
    [
        "value = cached_reqs.resumed_req_ids\n",
        "if request_id not in cached_reqs.resumed_req_ids:\n    unrelated_operation()\n",
        "if hasattr(cached_reqs, 'resumed_req_ids'):\n    unrelated_operation()\n",
    ],
)
def test_lmcache_unknown_scheduler_contract_fails_clearly(tmp_path, source_text):
    source = tmp_path / "unknown.py"
    source.write_text(source_text)
    module_name = "lmcache.integration.vllm.lmcache_mp_connector"
    with pytest.raises(RuntimeError, match="Unsupported"):
        _LMCacheLoader(module_name, str(source)).get_code(module_name)


def test_lmcache_scheduler_rewrite_is_scoped(tmp_path):
    source = tmp_path / "other.py"
    source.write_text("value = cached_reqs.resumed_req_ids\n")
    namespace = {"cached_reqs": type("Cached", (), {"resumed_req_ids": {"r"}})()}
    module_name = "lmcache.other"
    exec(_LMCacheLoader(module_name, str(source)).get_code(module_name), namespace)
    assert namespace["value"] == {"r"}


@pytest.mark.parametrize("has_inprocess", [False, True])
def test_lmcache_adapter_selects_installed_layout(monkeypatch, has_inprocess):
    from unittest.mock import Mock

    from sonar.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

    importer = Mock()
    monkeypatch.setattr(lmcache_integration, "import_lmcache", importer)
    monkeypatch.setattr(
        lmcache_integration.importlib.util, "find_spec", lambda name: object() if has_inprocess else None
    )
    assert lmcache_integration.get_lmcache_adapter() is importer.return_value
    prefix = "lmcache.inprocess" if has_inprocess else "lmcache"
    assert importer.call_args.args == (f"{prefix}.integration.vllm.vllm_v1_adapter",)


@pytest.mark.parametrize(
    "module_name",
    [
        "lmcache.integration.vllm.vllm_v1_adapter",
        "lmcache.inprocess.integration.vllm.vllm_v1_adapter",
        "lmcache.integration.vllm.lmcache_mp_metadata",
    ],
)
def test_lmcache_token_tensors_use_native_buffer_conversion(tmp_path, module_name):
    from types import SimpleNamespace

    import torch

    source = tmp_path / "tokens.py"
    source.write_text(
        '"""Installed adapter."""\n'
        "from __future__ import annotations\n"
        "import torch\n"
        "def convert(request):\n"
        "    return torch.tensor(request.prompt_token_ids)\n"
    )
    namespace: dict[str, Any] = {}
    exec(_LMCacheLoader(module_name, str(source)).get_code(module_name), namespace)
    converted = namespace["convert"](SimpleNamespace(prompt_token_ids=[2**33, 7]))
    assert converted.dtype == torch.int64
    assert converted.tolist() == [2**33, 7]
    assert namespace["convert"](SimpleNamespace(prompt_token_ids=None)).numel() == 0
