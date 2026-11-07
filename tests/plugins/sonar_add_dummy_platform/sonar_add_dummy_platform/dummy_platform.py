# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from sonar.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sonar.config import SonarConfig
    from sonar.v1.attention.backends.registry import AttentionBackendEnum
    from sonar.v1.attention.selector import AttentionSelectorConfig
else:
    SonarConfig = None


class DummyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "DummyDevice"
    device_type: str = "privateuseone"
    dispatch_key: str = "PrivateUse1"

    @classmethod
    def check_and_update_config(cls, sonar_config: SonarConfig) -> None:
        sonar_config.compilation_config.custom_ops = ["all"]

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        return "sonar_add_dummy_platform.dummy_attention_backend.DummyAttentionBackend"  # noqa E501
