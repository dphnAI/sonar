# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from heapq import heapify, heappop, heappush
from itertools import count, islice
from typing import TYPE_CHECKING

from sonar.utils.gc_utils import gc_untrack
from sonar.v1.core.kv_cache_utils import BlockHash, KVCacheBlock, get_block_hash

if TYPE_CHECKING:
    from sonar.v1.core.block_pool import BlockPool
    from sonar.v1.request import Request


class QueueGuard:
    """Keep indexed FCFS prefixes until other cached free blocks run out.

    Rank the existing pool's free blocks without acquiring cache references.
    Scan released candidates once and park protected ones until their owner
    leaves or allocation reaches them. Timestamps invalidate stale heap entries
    when a block is touched and released again.
    """

    def __init__(self, pool: "BlockPool", page_size: int, waiting: Callable[[], Iterable["Request"]]) -> None:
        self.pool = pool
        self.page_size = page_size
        self.hashes_per_page = page_size // pool.hash_block_size
        self.waiting = waiting
        self._seq = count()
        self._free_at: dict[int, int] = gc_untrack({})
        self._free: list[tuple[int, int]] = []
        self._parked: dict[str, list[tuple[int, int, int]]] = {}
        self._places: dict[str, int] = {}
        self._indexed: dict[str, tuple[int, list[BlockHash]]] = {}
        self._owners: dict[BlockHash, str | set[str]] = gc_untrack({})
        self._dirty = True

    def on_blocks_freed(self, blocks: list[KVCacheBlock]) -> None:
        free, free_at, seq = self._free, self._free_at, self._seq
        for block in blocks:
            block_id = block.block_id
            at = next(seq)
            free_at[block_id] = at
            heappush(free, (at, block_id))
        if len(free) > 2 * len(free_at) + 1024:
            self._free = [entry for entry in free if free_at.get(entry[1]) == entry[0]]
            heapify(self._free)

    def begin_step(self) -> None:
        self._dirty = True

    def reset(self) -> None:
        self._free.clear()
        self._free_at.clear()
        self._parked.clear()
        self._places.clear()
        self._indexed.clear()
        self._owners.clear()
        self._dirty = True

    def prepare_eviction(self, num_blocks: int) -> None:
        queue = self.pool.free_block_queue
        node = queue.fake_free_list_head.next_free_block
        while num_blocks and node is not None and node.block_hash is None:
            num_blocks -= 1
            node = node.next_free_block
        if not num_blocks:
            return
        if self._dirty:
            self._refresh()
        if not self._owners:
            return

        victims: list[KVCacheBlock] = []
        while len(victims) < num_blocks and self._free:
            at, block_id = heappop(self._free)
            if (block := self._candidate(at, block_id)) is None:
                continue
            owner = self._owner(block)
            if owner is None:
                del self._free_at[block_id]
                victims.append(block)
            else:
                # Give passed-over blocks a new place in the release order.
                at = next(self._seq)
                self._free_at[block_id] = at
                self._park(owner, block, at)

        while len(victims) < num_blocks and self._parked:
            owner = max(self._parked, key=self._places.__getitem__)
            bucket = self._parked[owner]
            while bucket and len(victims) < num_blocks:
                _, at, block_id = heappop(bucket)
                if (block := self._candidate(at, block_id)) is None:
                    continue
                current = self._owner(block)
                if current is not None and current != owner:
                    self._park(current, block, at)
                else:
                    del self._free_at[block_id]
                    victims.append(block)
            if not bucket:
                del self._parked[owner]

        if len(victims) == 1 and victims[0] is node:
            return
        for block in victims:
            queue.remove(block)
        queue.prepend_n(victims)

    def _candidate(self, at: int, block_id: int) -> KVCacheBlock | None:
        if self._free_at.get(block_id) != at:
            return None
        block = self.pool.blocks[block_id]
        if block.ref_cnt or block.prev_free_block is None or block.block_hash is None:
            del self._free_at[block_id]
            return None
        return block

    def _owner(self, block: KVCacheBlock) -> str | None:
        assert block.block_hash is not None
        owners = self._owners.get(get_block_hash(block.block_hash))
        if owners is None or isinstance(owners, str):
            return owners
        return min(owners, key=self._places.__getitem__)

    def _park(self, owner: str, block: KVCacheBlock, at: int) -> None:
        bucket = self._parked.get(owner)
        if bucket is None:
            bucket = self._parked[owner] = []
        heappush(bucket, (-(block.block_hash_num_tokens or 0), at, block.block_id))

    def _refresh(self) -> None:
        self._dirty = False
        places: dict[str, int] = {}
        entered: list[Request] = []
        indexed, owners = self._indexed, self._owners
        for place, request in enumerate(islice(self.waiting(), 256)):
            req_id = request.request_id
            places[req_id] = place
            entry = indexed.get(req_id)
            if len(entered) < 8 and (entry is None or entry[0] != request.num_tokens):
                entered.append(request)
        self._places = places

        changed = {request.request_id for request in entered}
        left = [req_id for req_id in indexed if req_id not in places or req_id in changed]
        for req_id in left:
            _, hashes = indexed.pop(req_id)
            for key in hashes:
                current = owners[key]
                if isinstance(current, str):
                    del owners[key]
                else:
                    current.remove(req_id)
                    if len(current) == 1:
                        owners[key] = next(iter(current))
        stride = self.hashes_per_page
        for request in entered:
            req_id = request.request_id
            pages = request.get_max_cache_hit_length() // self.page_size
            hashes = request.block_hashes[stride - 1 : max(0, pages) * stride : stride]
            indexed[req_id] = (request.num_tokens, hashes)
            for key in hashes:
                previous = owners.get(key)
                if previous is None:
                    owners[key] = req_id
                elif isinstance(previous, str):
                    owners[key] = gc_untrack({previous, req_id})
                else:
                    previous.add(req_id)

        for req_id in left:
            for _, at, block_id in self._parked.pop(req_id, ()):
                if (block := self._candidate(at, block_id)) is None:
                    continue
                owner = self._owner(block)
                if owner is None:
                    heappush(self._free, (at, block_id))
                else:
                    self._park(owner, block, at)

        if sum(map(len, self._parked.values())) > 2 * len(self._free_at) + 1024:
            for owner, bucket in list(self._parked.items()):
                live = [entry for entry in bucket if self._free_at.get(entry[2]) == entry[1]]
                if live:
                    heapify(live)
                    self._parked[owner] = live
                else:
                    del self._parked[owner]
