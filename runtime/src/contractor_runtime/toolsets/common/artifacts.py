"""Artifact client construction helpers for toolsets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import RuntimeSettings

ArtifactClientFactory = Callable[[str, RuntimeSettings], ArtifactClient]


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    """Default factory whose client fails on its first request."""

    return ArtifactClient(allocation_id, _UnavailableTransport())


def _reject_unconfigured_client(
    allocation_id: str, runtime_settings: RuntimeSettings
) -> ArtifactClient:
    """Default factory that fails when the toolset creates its tools."""

    del allocation_id, runtime_settings
    raise RuntimeError(_UNCONFIGURED_MESSAGE)


_UNCONFIGURED_MESSAGE = "Artifact transport is not configured"


class _UnavailableTransport:
    async def _unavailable(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(_UNCONFIGURED_MESSAGE)

    list_artifacts = _unavailable

    read_artifact_metadata = _unavailable

    check_artifact_access = _unavailable

    read_artifact = _unavailable

    write_artifact = _unavailable

    submit_finding = _unavailable

    planner_operation = _unavailable

    gateway_recovery = _unavailable

    submit_audit_evidence = _unavailable

    submit_audit_judgments = _unavailable

    mutate_sitemap = _unavailable

    snapshot_sitemap = _unavailable

    collect_sitemap = _unavailable

    list_findings = _unavailable

    read_finding = _unavailable

    select_findings = _unavailable
