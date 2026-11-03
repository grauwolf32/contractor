"""Typed private resources shared by one allocation's tools and Worker.

These objects never enter revisioned Contractor State or model-visible JSON.
AllocationContext owns their lifetime; consumers receive this explicit bundle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.common.artifact_read_cache import ExactArtifactReadCache
from contractor_runtime.toolsets.common.execution import WorkerExecutionState
from contractor_runtime.toolsets.common.http_capture import RecentHTTPCaptures
from contractor_runtime.toolsets.common.http_observations import HTTPObservationSink
from contractor_runtime.toolsets.common.report_publications import ReportPublicationSink


class ProposalCollector(Protocol):
    def check_finding_item(self, invocation_id: str, item_key: str | None) -> None: ...

    def check_proposal_refs(
        self, invocation_id: str, standard_refs: list[dict[str, str]]
    ) -> frozenset[tuple[str, str, str]]: ...

    async def register_proposal(
        self, invocation_id: str, client_key: str, standard_refs: list[dict[str, str]]
    ) -> None: ...


@dataclass(slots=True)
class ToolResources:
    metrics: MetricsState = field(default_factory=MetricsState)
    execution: WorkerExecutionState = field(default_factory=WorkerExecutionState)
    artifact_read_cache: ExactArtifactReadCache = field(default_factory=ExactArtifactReadCache)
    http_captures: RecentHTTPCaptures = field(default_factory=RecentHTTPCaptures)
    http_observation_sink: HTTPObservationSink | None = None
    caido_observation_sink: HTTPObservationSink | None = None
    report_publication_sink: ReportPublicationSink | None = None
    sitemap_read_flush: Callable[[], Awaitable[None]] | None = None
    before_completion: Callable[[], Awaitable[None]] | None = None
    proposal_collector: Callable[[], ProposalCollector | None] | None = None
