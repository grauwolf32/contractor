"""Task-restricted engagement metadata on the private allocation boundary."""

from __future__ import annotations

import ipaddress
import json
from datetime import datetime
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from contractor_runtime.contracts.artifacts import ArtifactRef
from contractor_runtime.contracts.base import (
    StrictWireModel,
    _require_aware_datetime,
    _require_text,
)
from contractor_runtime.contracts.engagement import (
    HTTPLimits,
    Identity,
    Route,
    canonical_url,
)


class ProtectedDestination(StrictWireModel):
    model_config = ConfigDict(populate_by_name=False, validate_by_name=False)

    hostname: str
    addresses: list[str] = Field(min_length=1, max_length=32)
    port: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def validate_destination(self) -> Self:
        host = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        canonical = canonical_url(f"http://{host}:{self.port}/")
        if urlsplit(canonical).hostname != self.hostname:
            raise ValueError("engagement protected hostname is not canonical")
        if len(set(self.addresses)) != len(self.addresses):
            raise ValueError("engagement protected addresses are duplicated")
        for raw in self.addresses:
            ip = ipaddress.ip_address(raw)
            if str(ip) != raw or (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped):
                raise ValueError("engagement protected address is invalid")
        return self


class HTTPAssignment(StrictWireModel):
    model_config = ConfigDict(populate_by_name=False, validate_by_name=False)

    adapter: Literal["http-target@1"]
    limits: HTTPLimits | None = None
    routes: list[Route] = Field(min_length=1, max_length=128)
    exclusions: list[Route] = Field(max_length=128)
    identities: list[Identity] = Field(min_length=1, max_length=8)
    protected_destinations: list[ProtectedDestination] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        ids = {identity.id for identity in self.identities}
        credential_ids = [
            identity.credential.credential_id
            for identity in self.identities
            if identity.credential is not None
        ]
        if len(set(credential_ids)) != len(credential_ids):
            raise ValueError("engagement credential selection is ambiguous")
        if len(ids) != len(self.identities) or "anonymous" not in ids:
            raise ValueError("controlled HTTP identities are invalid")
        for identity in self.identities:
            if identity.kind not in {"anonymous", "basic", "bearer", "cookies"}:
                raise ValueError("engagement identity kind is unavailable")
        return self


class EngagementAllocationProjection(StrictWireModel):
    model_config = ConfigDict(populate_by_name=False, validate_by_name=False)

    schema_: Literal["contractor.engagement.allocation.v1"] = Field(alias="schema")
    engagement: ArtifactRef
    engagement_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    authorized_by: str
    authorized_at: datetime
    http: HTTPAssignment | None = None

    @field_validator("authorized_at", mode="before")
    @classmethod
    def decode_authorized_at(cls, value: object, info: ValidationInfo) -> object:
        # Pydantic's before-model member guard materializes JSON objects. Keep
        # the strict JSON timestamp decoder, which otherwise sees a Python str.
        if info.mode == "json" and isinstance(value, str):
            return TypeAdapter(datetime).validate_json(json.dumps(value), strict=True)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        _require_text("authorizedBy", self.authorized_by)
        _require_aware_datetime("authorizedAt", self.authorized_at)
        self.engagement.require_exact()
        return self
