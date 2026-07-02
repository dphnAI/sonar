# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from aphrodite.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from aphrodite.config import AphroditeConfig
else:
    AphroditeConfig = None


class DummyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "DummyDevice"
    device_type: str = "privateuseone"
    dispatch_key: str = "PrivateUse1"

    @classmethod
    def check_and_update_config(cls, aphrodite_config: AphroditeConfig) -> None:
        aphrodite_config.compilation_config.custom_ops = ["all"]

    def get_attn_backend_cls(
        self,
        backend_name,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
    ):
        return "aphrodite_add_dummy_platform.dummy_attention_backend.DummyAttentionBackend"  # noqa E501
