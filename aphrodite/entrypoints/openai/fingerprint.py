# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the ``system_fingerprint`` string returned by the OpenAI-compatible
server.
Four modes, configured via ``--fingerprint-mode``:
* ``full`` (default): ``aphrodite-<version>[-<parallelism>]-<hash8>`` — encodes
  server version, any non-trivial parallelism degree (tp/pp/dp/ep), and an
  8-char prefix of ``aphrodite_config.compute_hash()`` (covers model identity,
  quant config, speculative, attention backend, etc.).
* ``hash``: ``aphrodite-<version>-<hash8>`` — parallelism stripped.
* ``custom``: user-provided literal via ``--fingerprint-value``.
* ``none``: the field is omitted (serialized as ``null``).
``get_system_fingerprint`` is only called at serving-class init (a handful
of times per server); each subclass caches the returned string on
``self.system_fingerprint``, so per-request cost is one attribute read.
"""

from __future__ import annotations

from typing import Any, Literal

FingerprintMode = Literal["full", "hash", "custom", "none"]

_DEFAULT_MODE: FingerprintMode = "full"
_CUSTOM_VALUE: str | None = None


def set_default_fingerprint_mode(
    mode: FingerprintMode,
    custom_value: str | None = None,
) -> None:
    """Configure the fingerprint mode for subsequent ``get_system_fingerprint``
    calls. Called once at server startup."""
    global _DEFAULT_MODE, _CUSTOM_VALUE
    _DEFAULT_MODE = mode
    _CUSTOM_VALUE = custom_value


def get_system_fingerprint(aphrodite_config: Any) -> str | None:
    """Return the fingerprint for ``aphrodite_config`` using the mode configured by
    ``set_default_fingerprint_mode``."""
    return build_system_fingerprint(aphrodite_config, _DEFAULT_MODE, _CUSTOM_VALUE)


def build_system_fingerprint(
    aphrodite_config: Any,
    mode: FingerprintMode = "full",
    custom_value: str | None = None,
) -> str | None:
    if mode == "none":
        return None
    if mode == "custom":
        return custom_value

    from aphrodite import __version__ as aphrodite_version

    try:
        hash8 = aphrodite_config.compute_hash()[:8]
    except Exception:
        hash8 = "nohash"

    if mode == "hash":
        return f"aphrodite-{aphrodite_version}-{hash8}"

    # mode == "full"
    parts: list[str] = [f"aphrodite-{aphrodite_version}"]
    pc = getattr(aphrodite_config, "parallel_config", None)
    if pc is not None:
        tp = getattr(pc, "tensor_parallel_size", 1)
        if tp > 1:
            parts.append(f"tp{tp}")
        pp = getattr(pc, "pipeline_parallel_size", 1)
        if pp > 1:
            parts.append(f"pp{pp}")
        dp = getattr(pc, "data_parallel_size", 1)
        if dp > 1:
            parts.append(f"dp{dp}")
        if getattr(pc, "enable_expert_parallel", False):
            parts.append("ep")
    parts.append(hash8)
    return "-".join(parts)
