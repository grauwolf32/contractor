"""Final reports built from allocation-owned metrics and resource counters."""

from __future__ import annotations

from datetime import datetime

from contractor_runtime.allocation.context import _AllocationContext
from contractor_runtime.contracts import (
    AllocationFinalReport,
    RuntimeReport,
    TerminationError,
)
from contractor_runtime.telemetry.metrics import MetricsState


def _build_report(
    context: _AllocationContext,
    finished_at: datetime,
    reason: TerminationError | None,
) -> AllocationFinalReport:
    metrics = context.worker_state.metrics if context.worker_state is not None else MetricsState()
    duration_ms = max(0, int((finished_at - context.started_at).total_seconds() * 1000))
    errors = (reason,) if reason is not None else ()
    stop_reason = reason.code if reason is not None else "finalized"
    worker = metrics.build_report(
        report_id=f"worker-{context.allocation_id}",
        duration_ms=duration_ms,
        extra_errors=errors,
    )
    return AllocationFinalReport(
        reportId=f"allocation-final-{context.allocation_id}",
        allocationId=context.allocation_id,
        startedAt=context.started_at,
        finishedAt=finished_at,
        worker=worker,
        runtime=RuntimeReport(
            complete=True,
            durationMs=duration_ms,
            stopReason=stop_reason,
            adapters=context.adapter_host.report_metrics(),
            **(
                {"resources": context.resource_metrics.finish()}
                if context.resource_metrics is not None
                else {}
            ),
        ),
    )
