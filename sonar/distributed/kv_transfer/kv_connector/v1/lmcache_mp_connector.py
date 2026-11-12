# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from sonar.distributed.kv_transfer.kv_connector.v1.lmcache_integration import import_lmcache

# Keep the connector and its wire protocol from the same LMCache installation.
LMCacheMPConnector = import_lmcache("lmcache.integration.vllm.lmcache_mp_connector").LMCacheMPConnector
