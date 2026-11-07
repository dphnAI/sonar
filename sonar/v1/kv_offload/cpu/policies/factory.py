# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable

from aphrodite.logger import init_logger
from aphrodite.v1.kv_offload.cpu.policies.base import CachePolicy

logger = init_logger(__name__)


class CachePolicyFactory:
    """Registry for CachePolicy implementations, resolved by name.

    Built-in policies are pre-registered below. External policies can either
    register a friendly short name up front or pass a module path at lookup
    time without modifying Aphrodite.
    """

    _registry: dict[str, Callable[[], type[CachePolicy]]] = {}

    @classmethod
    def register_cache_policy(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a cache policy with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Cache policy '{name}' is already registered.")

        def loader() -> type[CachePolicy]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_cache_policy_cls(cls, name: str, module_path: str | None = None) -> type[CachePolicy]:
        """Get a registered or out-of-tree cache policy class by name."""
        if name in cls._registry:
            return cls._registry[name]()
        if module_path is None:
            raise ValueError(
                f"Unknown cache policy: {name!r}. Supported: {list(cls._registry)}. "
                "For an out-of-tree policy, also set cache_policy_module_path."
            )
        logger.warning_once(
            "Loading out-of-tree cache policy. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        logger.info(
            "Loading out-of-tree cache policy '%s' from '%s'.",
            name,
            module_path,
        )
        module = importlib.import_module(module_path)
        policy_cls = getattr(module, name)
        assert issubclass(policy_cls, CachePolicy)
        return policy_cls


CachePolicyFactory.register_cache_policy("lru", "aphrodite.v1.kv_offload.cpu.policies.lru", "LRUCachePolicy")
CachePolicyFactory.register_cache_policy("arc", "aphrodite.v1.kv_offload.cpu.policies.arc", "ARCCachePolicy")
