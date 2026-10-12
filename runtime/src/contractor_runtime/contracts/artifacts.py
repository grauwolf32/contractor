"""Private Runtime protocol artifacts models and validation."""

from __future__ import annotations

from typing import Self

from pydantic import (
    Field,
    model_validator,
)

from contractor_runtime.contracts.base import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    ARTIFACT_NAME_PATTERN,
    MAX_ARTIFACT_LIST_CURSOR_LENGTH,
    MEDIA_TYPE_PATTERN,
    VersionedWireModel,
    WireModel,
    _require_text,
)

# The Run Namespaces a trusted purpose-specific flow owns (ART-R114): none is a
# Stage Agent or MemoryTools Namespace, and no Stage or Worker result binds
# into one. The memory. name prefix belongs to MemoryTools in every other
# Namespace (ART-R19 to ART-R22). The Server's artifactpolicy applies the same
# rule; api/testdata/v1alpha1/memory-reservation-cases.json pins both.
PURPOSE_RESERVED_NAMESPACES = frozenset(
    {"contractor-system", "finding-proposals", "inputs", "outputs", "skills"}
)
MEMORY_ARTIFACT_PREFIX = "memory."


def is_reserved_memory_binding(namespace: object, name: object) -> bool:
    """Return whether a Run binding belongs exclusively to MemoryTools."""

    return (
        isinstance(namespace, str)
        and namespace not in PURPOSE_RESERVED_NAMESPACES
        and isinstance(name, str)
        and name.startswith(MEMORY_ARTIFACT_PREFIX)
    )


def is_reserved_result_binding(namespace: object, name: object) -> bool:
    """Return whether a Stage or Worker result may not bind this Run binding."""

    return (
        isinstance(namespace, str) and namespace in PURPOSE_RESERVED_NAMESPACES
    ) or is_reserved_memory_binding(namespace, name)


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
    if MEDIA_TYPE_PATTERN.fullmatch(value) is None:
        raise ValueError("mediaType must be lowercase type/subtype without parameters")


class ArtifactListResult(VersionedWireModel):
    """One page of versionless refs; next_cursor continues a paged listing."""

    artifacts: list[ArtifactRef]
    next_cursor: str | None = None

    @model_validator(mode="after")
    def validate_artifacts(self) -> Self:
        if any(artifact.revision is not None for artifact in self.artifacts):
            raise ValueError("listed artifact refs must be versionless")
        if self.next_cursor is not None:
            if ARTIFACT_LIST_CURSOR_PATTERN.fullmatch(self.next_cursor) is None:
                raise ValueError(
                    f"nextCursor must be 1 through {MAX_ARTIFACT_LIST_CURSOR_LENGTH} "
                    "base64url characters"
                )
            if not self.artifacts:
                raise ValueError("an empty artifact list page cannot continue")
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
