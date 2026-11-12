# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Sonar project

import importlib.util
import py_compile
import sys

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
