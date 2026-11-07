# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in configuration for sampled runtime-weight receipts."""

import os
from typing import Any, Self

from pydantic import Field, model_validator

from aphrodite.config.utils import config


@config
class DphnNetworkConfig:
    """Weight-only sampling configuration; does not attest inference execution."""

    public_key: str = ""
    """Verifier X25519 public key, encoded as base64/base64url or PEM."""
    worker_id: str = Field(default="", min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    """Worker identity prefix, qualified with data/model-parallel ranks at runtime."""
    model_manifest_id: str = Field(default="", min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    """Identifier of the trusted runtime-weight manifest."""
    key_id: str = Field(default="default", min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    """Verifier key identifier."""
    num_raw_terms: int = Field(default=8, ge=1, le=64, strict=True)
    """Number of sampled raw-weight terms per worker receipt."""
    raw_term_bytes: int = Field(default=256, ge=16, le=4096, strict=True)
    """Maximum bytes per sampled term."""

    @model_validator(mode="after")
    def _validate_sampling(self) -> Self:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

        from aphrodite.proof.verifier import _load_x25519_public_key

        missing = [name for name in ("public_key", "worker_id", "model_manifest_id") if not getattr(self, name)]
        if missing:
            raise ValueError(f"DPHN network configuration requires {', '.join(missing)}")
        public_key = _load_x25519_public_key(self.public_key)
        if not isinstance(public_key, X25519PublicKey):
            raise ValueError("DPHN public_key must be an X25519 public key")
        X25519PrivateKey.generate().exchange(public_key)
        return self

    @classmethod
    def resolve(cls, enabled: bool, value: "DphnNetworkConfig | dict[str, Any] | None") -> "DphnNetworkConfig | None":
        """Read environment defaults only after explicit opt-in; JSON wins."""
        if not enabled:
            if value:
                raise ValueError("dphn_network_config requires explicit --dphn-network")
            return None
        if isinstance(value, cls):
            return value
        if value is not None and not isinstance(value, dict):
            raise TypeError("dphn_network_config must be a configuration object or JSON object")
        defaults: dict[str, Any] = {
            field: os.environ[env]
            for field, env in (
                ("public_key", "DPHN_PUBLIC_KEY"),
                ("worker_id", "DPHN_WORKER_ID"),
                ("model_manifest_id", "DPHN_MODEL_MANIFEST_ID"),
                ("key_id", "DPHN_KEY_ID"),
            )
            if env in os.environ
        }
        defaults.update(value or {})
        return cls(**defaults)
