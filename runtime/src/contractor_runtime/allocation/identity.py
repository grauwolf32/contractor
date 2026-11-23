"""Keyed identity for idempotent allocation preparation."""

from __future__ import annotations

import hashlib
import hmac

from contractor_runtime.contracts import (
    AllocationSpec,
)


def _spec_fingerprint(spec: AllocationSpec, key: bytes) -> str:
    # The confirmed control lease can advance while durable placement stays
    # pinned. A replay must still identify the already-prepared allocation.
    encoded = spec.model_dump_json(
        by_alias=True, exclude_none=True, exclude={"lease_expires_at"}
    ).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()
