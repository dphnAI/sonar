# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

def compile(patterns: Sequence[bytes], /) -> object: ...
def scan(pattern: object, state: int, data: bytes, /) -> tuple[int, int | None, int]: ...
