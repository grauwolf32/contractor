"""Keyed identity for idempotent allocation preparation."""

from __future__ import annotations

import hashlib
import hmac

from contractor_runtime.contracts import (
    AllocationSpec,
)


def _spec_fingerprint(spec: AllocationSpec, key: bytes) -> str:
    encoded = spec.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()
