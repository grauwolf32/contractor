"""Pinned Project sitemap policy and allocation command responses."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from contractor_runtime.contracts.artifacts import (
    ArtifactRef,
    validate_media_type,
)
from contractor_runtime.contracts.base import (
    DIGEST_PATTERN,
    NAME_PATTERN,
    VersionedWireModel,
    WireModel,
    _require_resource_id,
    _require_sorted_unique,
)
from contractor_runtime.shared_config import artifact_payload_max_bytes

MAX_SITEMAP_SNAPSHOT_INPUTS = 8


class ProjectSitemapPolicy(WireModel):
    version: Literal["sitemap-access@1"]
    collect_observations: bool
    deployment: str | None = Field(default=None, exclude_if=lambda value: value is None)
    snapshot_inputs: list[str] = Field(
        default_factory=list,
        max_length=MAX_SITEMAP_SNAPSHOT_INPUTS,
        exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.deployment is not None:
            _require_resource_id("sitemap deployment", self.deployment)
        _require_sorted_unique(
            "sitemap snapshot inputs", self.snapshot_inputs, maximum=MAX_SITEMAP_SNAPSHOT_INPUTS
        )
        if any(NAME_PATTERN.fullmatch(name) is None for name in self.snapshot_inputs):
            raise ValueError("invalid sitemap snapshot input name")
        return self


class SitemapInputDescriptor(WireModel):
    ref: ArtifactRef
    digest: str = Field(pattern=DIGEST_PATTERN.pattern)
    media_type: str
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_payload_size(self) -> Self:
        if self.size_bytes > artifact_payload_max_bytes():
            raise ValueError("Artifact size exceeds the shared payload limit")
        return self

    @model_validator(mode="after")
    def validate_descriptor(self) -> Self:
        self.ref.require_exact()
        validate_media_type(self.media_type)
        return self


class SitemapMutationResult(WireModel):
    artifact: ArtifactRef
    replayed: bool
    changed: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        self.artifact.require_exact()
        return self


class SitemapMutationResponse(VersionedWireModel):
    result: SitemapMutationResult


class SitemapSnapshotResult(WireModel):
    artifact: SitemapInputDescriptor
    replayed: bool


class SitemapSnapshotResponse(VersionedWireModel):
    result: SitemapSnapshotResult
