"""Private Runtime protocol artifacts models and validation."""

from __future__ import annotations

from typing import Self

from pydantic import (
    Field,
    model_validator,
)

from contractor_runtime.contracts.base import (
    ARTIFACT_NAME_PATTERN,
    VersionedWireModel,
    WireModel,
    _require_text,
)


class ArtifactRef(WireModel):
    namespace: str
    name: str
    revision: str | None = None

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        if ARTIFACT_NAME_PATTERN.fullmatch(self.namespace) is None:
            raise ValueError(
                "namespace must be a portable ASCII Artifact name of 1 through 128 characters"
            )
        if ARTIFACT_NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError(
                "name must be a portable ASCII Artifact name of 1 through 128 characters"
            )
        if self.revision is not None:
            _require_text("revision", self.revision)
        return self

    def require_exact(self) -> Self:
        if self.revision is None:
            raise ValueError("artifact revision is required")
        return self


def _validate_media_type(value: str) -> None:
    parts = value.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or value != value.lower()
        or any(char in value for char in "; ")
    ):
        raise ValueError("mediaType must be lowercase type/subtype without parameters")


class ArtifactListResult(VersionedWireModel):
    artifacts: list[ArtifactRef]

    @model_validator(mode="after")
    def validate_artifacts(self) -> Self:
        if any(artifact.revision is not None for artifact in self.artifacts):
            raise ValueError("listed artifact refs must be versionless")
        return self


class ArtifactReadResult(VersionedWireModel):
    artifact: ArtifactRef
    media_type: str
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        self.artifact.require_exact()
        _validate_media_type(self.media_type)
        return self


class ArtifactWriteResult(ArtifactReadResult):
    pass
