# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from sonar.config import KVTransferConfig
from sonar.config.kv_transfer import KVCacheAccess
from sonar.distributed.kv_transfer.kv_connector.v1.lmcache_integration import import_lmcache

if TYPE_CHECKING:
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector as _LMCacheMPConnector
else:
    # Keep the connector and its wire protocol from the same LMCache installation.
    _LMCacheMPConnector = import_lmcache("lmcache.integration.vllm.lmcache_mp_connector").LMCacheMPConnector


class LMCacheMPConnector(_LMCacheMPConnector):
    @classmethod
    def kv_cache_access(cls, kv_transfer_config: KVTransferConfig) -> KVCacheAccess:
        # The maintained MP adapter shares the cache through CUDA IPC handles.
        return KVCacheAccess.LEGACY_IPC
