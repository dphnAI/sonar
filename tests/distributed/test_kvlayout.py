# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.config import (
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SonarConfig,
    set_current_sonar_config,
)
from sonar.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from sonar.logger import init_logger

logger = init_logger("test_expert_parallel")


def test_get_kv_connector_cache_layout_without_kv_connector():
    sonar_config = SonarConfig(device_config=DeviceConfig("cpu"))
    with set_current_sonar_config(sonar_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout is None


def test_get_kv_connector_cache_layout_with_lmcache_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    sonar_config = SonarConfig(device_config=DeviceConfig("cpu"), kv_transfer_config=kv_transfer_config)
    with set_current_sonar_config(sonar_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout is None


def test_get_kv_connector_cache_layout_with_nixl_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    model_config = ModelConfig()
    sonar_config = SonarConfig(
        device_config=DeviceConfig("cpu"),
        model_config=model_config,
        kv_transfer_config=kv_transfer_config,
    )
    with set_current_sonar_config(sonar_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout == "LBHNC"


def test_get_kv_connector_cache_layout_with_multi_connector():
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "ExampleConnector", "kv_role": "kv_both"},
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            ]
        },
    )
    model_config = ModelConfig()
    sonar_config = SonarConfig(
        device_config=DeviceConfig("cpu"),
        model_config=model_config,
        kv_transfer_config=kv_transfer_config,
    )
    with set_current_sonar_config(sonar_config):
        # Test with default settings
        layout = get_kv_connector_cache_layout()
        assert layout == "LBHNC"
