# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import pytest

from sonar.v1.kv_offload.base import OffloadKey, ReqContext
from sonar.v1.kv_offload.cpu.manager import CPUOffloadingManager
from sonar.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from sonar.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from sonar.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from sonar.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class _DummyCachePolicy(CachePolicy):
    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)

    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return None

    def insert(self, key: OffloadKey, chunk: ChunkStatus, req_context: ReqContext) -> None:
        pass

    def remove(self, key: OffloadKey) -> None:
        pass

    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        pass

    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        return None

    def clear(self) -> None:
        pass


@pytest.fixture(autouse=True)
def restore_cache_policy_registry():
    original = dict(CachePolicyFactory._registry)
    yield
    CachePolicyFactory._registry = original


class TestCachePolicyFactory:
    def test_pre_registered_policies_can_be_imported(self):
        for name in CachePolicyFactory._registry:
            policy_cls = CachePolicyFactory._registry[name]()
            assert issubclass(policy_cls, CachePolicy)

    def test_lru_and_arc_registered(self):
        assert CachePolicyFactory.get_cache_policy_cls("lru") is LRUCachePolicy
        assert CachePolicyFactory.get_cache_policy_cls("arc") is ARCCachePolicy

    def test_register_and_resolve_custom_policy(self):
        CachePolicyFactory.register_cache_policy(
            "dummy",
            "tests.v1.kv_offload.cpu.policies.test_factory",
            "_DummyCachePolicy",
        )
        policy_cls = CachePolicyFactory.get_cache_policy_cls("dummy")
        assert policy_cls is _DummyCachePolicy

        manager = CPUOffloadingManager(num_chunks=4, cache_policy="dummy")
        assert isinstance(manager._policy, _DummyCachePolicy)

    def test_unregistered_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CachePolicyFactory.get_cache_policy_cls("nonexistent")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="is already registered"):
            CachePolicyFactory.register_cache_policy("lru", "some.module", "SomeClass")

    def test_dynamic_load_via_cache_policy_module_path(self):
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            "_DummyCachePolicy",
            "tests.v1.kv_offload.cpu.policies.test_factory",
        )
        assert policy_cls is _DummyCachePolicy

    def test_manager_resolves_policy_via_module_path(self):
        manager = CPUOffloadingManager(
            num_chunks=4,
            cache_policy="_DummyCachePolicy",
            cache_policy_module_path="tests.v1.kv_offload.cpu.policies.test_factory",
        )
        assert isinstance(manager._policy, _DummyCachePolicy)

    def test_unregistered_policy_without_module_path_raises(self):
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CachePolicyFactory.get_cache_policy_cls("nonexistent", None)
