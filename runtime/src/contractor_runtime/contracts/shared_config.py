"""Strict, versioned startup configuration returned by the Control Plane."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from contractor_runtime.contracts.base import (
    DIGEST_PATTERN,
    MAX_EXACT_INTEGER,
    VersionedWireModel,
    WireModel,
)
from contractor_runtime.shared_config import SharedConfigSnapshot


class SharedLimits(WireModel):
    model_config = ConfigDict(frozen=True)
    artifact_payload_max_bytes: int = Field(strict=True, gt=0, le=MAX_EXACT_INTEGER)
    public_page_max_items: int = Field(strict=True, gt=0, le=MAX_EXACT_INTEGER)
    artifact_list_max_items: int = Field(strict=True, gt=0, le=MAX_EXACT_INTEGER)


class SharedConfig(VersionedWireModel):
    model_config = ConfigDict(frozen=True)
    version: int = Field(strict=True, ge=1, le=1)
    limits: SharedLimits
    digest: str = Field(pattern=DIGEST_PATTERN.pattern)

    @model_validator(mode="after")
    def verify_digest(self) -> Self:
        projection = {"limits": self.limits.model_dump(by_alias=True), "version": self.version}
        canonical = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
        if self.digest != expected:
            raise ValueError("invalid shared config digest")
        return self

    def snapshot(self) -> SharedConfigSnapshot:
        return SharedConfigSnapshot(
            self.version,
            self.digest,
            self.limits.artifact_payload_max_bytes,
            self.limits.artifact_list_max_items,
        )
