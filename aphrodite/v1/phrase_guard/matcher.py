# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from threading import RLock

from cachetools import LRUCache, cached

import aphrodite._phrase_matcher as _phrase_matcher

MAX_PHRASES = 2048
MAX_PHRASE_CHARS = 256
MAX_ZERO_WIDTH_TOKENS = 256
MAX_RETRIES = 128
_pattern_cache: LRUCache[tuple, tuple[object, int]] = LRUCache(maxsize=65536, getsizeof=lambda entry: entry[1])


def validate_phrases(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_PHRASES:
        raise ValueError(f"banned_strings must be a nonempty list of at most {MAX_PHRASES} strings")
    for phrase in value:
        if not isinstance(phrase, str) or not 0 < len(phrase) <= MAX_PHRASE_CHARS:
            raise ValueError(f"Each banned string must contain 1..{MAX_PHRASE_CHARS} characters")
    if sum(max(len(phrase.encode("utf-8")), len(phrase.casefold().encode("utf-8"))) for phrase in value) > 262144:
        raise ValueError("banned_strings exceeds the 256 KiB phrase budget")
    return tuple(dict.fromkeys(value))


@cached(_pattern_cache, lock=RLock())
def _compile_phrases(phrases: tuple[str, ...], case_sensitive: bool) -> tuple[object, int]:
    encoded = tuple((phrase if case_sensitive else phrase.casefold()).encode("utf-8") for phrase in phrases)
    # Each encoded byte adds at most one trie node. Include cache-entry overhead;
    # entries larger than the cache budget are compiled without being retained.
    return _phrase_matcher.compile(encoded), sum(map(len, encoded)) + 64


def compile_phrases(phrases: tuple[str, ...], case_sensitive: bool):
    return _compile_phrases(phrases, case_sensitive)[0]


@dataclass(frozen=True)
class Decision:
    ready: list[int]
    rewind_to: int | None = None
    blocked_token: int | None = None


class PendingText:
    """Token-aligned output journal. Only ``ready`` tokens may leave the engine."""

    def __init__(self, phrases: tuple[str, ...], case_sensitive: bool = False):
        self.pattern = compile_phrases(phrases, case_sensitive)
        self.case_sensitive = case_sensitive
        self.max_pending_tokens = MAX_ZERO_WIDTH_TOKENS + max(
            max(len(phrase.encode("utf-8")), len(phrase.casefold().encode("utf-8"))) for phrase in phrases
        )
        self.zero_width_tokens = 0
        self.state = 0
        self.tokens: list[int] = []
        self.ends: list[int] = []
        self.states: list[int] = []
        self.published_state = 0
        self.num_bytes = 0
        self.published = 0

    def _token_at(self, byte_offset: int) -> int:
        # Include zero-width tokens which contributed to a later decoded
        # character. A UTF-8 character can span several byte-fallback tokens.
        start = 0
        for i, end in enumerate(self.ends):
            if end > byte_offset:
                return start
            if end > (self.ends[i - 1] if i else 0):
                start = i + 1
        return start

    def _release(self, count: int) -> list[int]:
        if not count:
            return []
        ready = self.tokens[:count]
        released_bytes = self.ends[count - 1]
        self.tokens = self.tokens[count:]
        self.ends = [end - released_bytes for end in self.ends[count:]]
        self.published_state = self.states[count - 1]
        self.states = self.states[count:]
        self.num_bytes -= released_bytes
        self.published += count
        return ready

    def feed(self, token: int, text: str) -> Decision:
        self.tokens.append(token)
        data = (text if self.case_sensitive else text.casefold()).encode("utf-8")
        offset = self.num_bytes
        self.num_bytes += len(data)
        self.ends.append(self.num_bytes)
        self.zero_width_tokens = 0 if data else self.zero_width_tokens + 1
        if len(self.tokens) > self.max_pending_tokens or self.zero_width_tokens > MAX_ZERO_WIDTH_TOKENS:
            raise ValueError("banned_strings pending-token limit exceeded")
        self.state, match, suffix = _phrase_matcher.scan(self.pattern, self.state, data)
        self.states.append(self.state)
        keep_from = self.num_bytes - suffix
        if match is not None:
            start = min(offset + match, keep_from)
            boundary = self._token_at(start)
            ready = self._release(boundary)
            blocked = self.tokens[0]
            self.tokens.clear()
            self.ends.clear()
            self.states.clear()
            self.num_bytes = 0
            self.zero_width_tokens = 0
            # Whole-token rollback can reactivate a previously released prefix.
            self.state = self.published_state
            return Decision(ready, self.published, blocked)
        # Retain zero-width tokens until their decode is resolved.
        return Decision(self._release(self._token_at(keep_from)))

    def finish(self) -> list[int]:
        """Release an incomplete prefix at EOS, after checking the final token."""
        return self._release(len(self.tokens))
