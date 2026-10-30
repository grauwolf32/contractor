"""Optional notifications for ordinary tool reports, independent of their consumers."""

from __future__ import annotations

from typing import Protocol

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.subscribers import notify_subscriber


class ReportPublicationSink(Protocol):
    async def receive_report(self, tool: str, artifact: ArtifactRef) -> None: ...


async def notify_report_publication(
    sink: ReportPublicationSink | None, tool: str, artifact: ArtifactRef
) -> None:
    if sink is not None:
        await notify_subscriber(
            lambda: sink.receive_report(tool, artifact), name="tool-report-publication"
        )
