# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import numpy as np

from aphrodite.v1.outputs import LogprobsLists


def join_logprobs(rows: list[LogprobsLists]) -> LogprobsLists | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    return LogprobsLists(*(np.concatenate([row[i] for row in rows]) for i in range(3)))


class PendingLogprobs:
    def __init__(self):
        self.rows: deque[LogprobsLists] = deque()

    def append(self, packet: LogprobsLists, index: int):
        # A held token must not retain another request's whole batch allocation.
        self.rows.append(LogprobsLists(*(array[index : index + 1].copy() for array in packet[:3])))

    def take(self, count: int) -> list[LogprobsLists]:
        return [self.rows.popleft() for _ in range(count)]

    def clear(self):
        self.rows.clear()
