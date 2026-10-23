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
# FINDING_PROPOSAL_NAMESPACE (artifactpolicy.FindingProposalNamespace) holds the
# proposals and frozen HTTP exchanges Server finding intake writes; Runtime code
# outside this package names it only through the constant.
FINDING_PROPOSAL_NAMESPACE = "finding-proposals"
PURPOSE_RESERVED_NAMESPACES = frozenset(
    {"contractor-system", FINDING_PROPOSAL_NAMESPACE, "inputs", "outputs", "skills"}
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


def validate_artifact_name(field_name: str, value: object) -> str:
    """Return ``value`` when it is a portable Artifact name.

    The one Runtime check of the grammar that artifact-name-cases.json pins,
    for namespaces, names and name prefixes alike.
    """

    if type(value) is not str or ARTIFACT_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a portable ASCII Artifact name of 1 through 128 characters"
        )
    return value


def validate_media_type(value: object) -> str:
    """Return ``value`` when it is a media type that media-type-cases.json admits."""

    if type(value) is not str or MEDIA_TYPE_PATTERN.fullmatch(value) is None:
        raise ValueError("artifact media type is invalid: use lowercase type/subtype only")
    return value


class ArtifactRef(WireModel):
    namespace: str
    name: str
    revision: str | None = None

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        validate_artifact_name("namespace", self.namespace)
        validate_artifact_name("name", self.name)
        if self.revision is not None:
            _require_text("revision", self.revision)
        return self

    def require_exact(self) -> Self:
        if self.revision is None:
            raise ValueError("artifact revision is required")
        return self


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
        validate_media_type(self.media_type)
        return self


class ArtifactWriteResult(ArtifactReadResult):
    pass
