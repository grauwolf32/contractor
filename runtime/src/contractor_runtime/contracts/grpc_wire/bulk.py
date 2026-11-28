"""Strict native payload metadata, shared by transport and trusted clients."""

import re
from datetime import UTC, datetime
from typing import Self

from pydantic import ConfigDict, field_validator, model_validator

from contractor_runtime.contracts.artifacts import ArtifactReadResult
from contractor_runtime.shared_config import artifact_payload_max_bytes

_UTC = re.compile(
    r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?Z$"
)


class ArtifactPayloadMetadata(ArtifactReadResult):
    model_config = ConfigDict(
        populate_by_name=False,
        validate_by_name=False,
        validate_by_alias=True,
        hide_input_in_errors=True,
    )
    binding_created_at: datetime
    revision_created_at: datetime

    @field_validator("binding_created_at", "revision_created_at", mode="before")
    @classmethod
    def canonical_utc(cls, value):
        if not isinstance(value, str) or (match := _UTC.fullmatch(value)) is None:
            raise ValueError("invalid Artifact timestamp")
        fraction = match.group("fraction") or ""
        suffix = "." + (fraction + "000000")[:6] if fraction else ""
        parsed = datetime.fromisoformat(f"{match.group('date')}{suffix}+00:00")
        if parsed.tzinfo != UTC:
            raise ValueError("invalid Artifact timestamp")
        return parsed

    @model_validator(mode="after")
    def bounded_payload(self) -> Self:
        if self.size > artifact_payload_max_bytes():
            raise ValueError("Artifact payload exceeds its limit")
        return self
