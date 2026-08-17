from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from contractor.runners.models import TaskRunnerEvent
from contractor.runners.task_runner import TaskRunnerEventHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowContext:
    project_path: Path
    folder_name: str
    model: str
    app_name: str
    user_id: str
    artifact_service: BaseArtifactService
    fs: AbstractFileSystem
    artifact: str | None = None
    prompt: str | None = None
    checkpoint_path: Path | None = None
    # Per-request model timeout (seconds). Mirrors Settings.default_model_timeout;
    # the CLI passes the configured/overridden value, so workflow-built LiteLlm
    # instances honour it instead of silently dropping the timeout.
    timeout: int = 300


class Workflow(ABC):
    """Base class for all workflows.

    Subclasses receive a ``WorkflowContext`` at construction time and
    implement ``_run_impl()``. The public ``run()`` wraps it with
    ``workflow_started`` / ``workflow_finished`` lifecycle events so
    the UI shows activity immediately, even when sub-agent setup is
    slow.
    """

    def __init__(self, ctx: WorkflowContext) -> None:
        self.ctx = ctx

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: TaskRunnerEventHandler | None = None,
    ) -> Any:
        await self._emit(on_event, "workflow_started", phase="initializing")
        ok = False
        try:
            result = await self._run_impl(user_id=user_id, on_event=on_event)
            ok = True
            return result
        finally:
            try:
                await self._cleanup(user_id=user_id)
            except Exception:
                logger.exception("_cleanup failed")
            await self._emit(
                on_event,
                "workflow_finished",
                ok=ok,
            )

    @abstractmethod
    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any: ...

    async def _cleanup(self, *, user_id: str) -> None:  # noqa: B027 — optional override hook, deliberately a concrete no-op (not abstract)
        """Hook for subclasses to persist supplementary state (overlay FS, etc.).

        Called in ``run()``'s finally block, before the ``workflow_finished``
        event. The default implementation is a no-op.
        """

    async def _emit(
        self,
        on_event: TaskRunnerEventHandler | None,
        event_type: str,
        **payload: Any,
    ) -> None:
        if on_event is None:
            return
        event = TaskRunnerEvent(
            type=event_type,
            task_name=self.name,
            task_id=-1,
            payload={"workflow": self.name, **payload},
        )
        try:
            await on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Lifecycle events are telemetry/UI notifications. A broken sink
            # must not prevent workflow setup, suppress cleanup, or replace the
            # workflow's real result/error.
            logger.exception(
                "event handler failed for %s event (workflow %s)",
                event_type,
                self.name,
            )

    async def artifact_exists(self, *, user_id: str, filename: str) -> bool:
        """True iff a non-empty artifact is already stored under ``filename``.

        Useful for workflows that want to skip an upstream analysis task when
        a previous run (or a sibling workflow) has already produced its
        output artifact.
        """
        part = await self.ctx.artifact_service.load_artifact(
            app_name=self.ctx.app_name,
            user_id=user_id,
            filename=filename,
        )
        if part is None:
            return False
        if part.inline_data is not None:
            return True
        return bool(part.text)

    async def emit_task_skipped(
        self,
        on_event: TaskRunnerEventHandler | None,
        task_name: str,
        *,
        reason: str = "artifact_already_exists",
    ) -> None:
        """Log + emit a ``task_skipped`` lifecycle event."""
        import logging

        logging.getLogger(self.__class__.__module__).info(
            "skipping %s — %s", task_name, reason
        )
        await self._emit(
            on_event,
            "task_skipped",
            task_name=task_name,
            reason=reason,
        )

    async def run_skippable_job(
        self,
        awaitable: Awaitable[Any],
        *,
        job_name: str,
        on_event: TaskRunnerEventHandler | None,
        default: Any = None,
        skip_on_failure: bool = True,
    ) -> Any:
        """Run one fan-out job, isolating its failure from the rest of the run.

        Trace / vuln workflows fan a freshly-built agent out over every
        OpenAPI path / operation / route group. Without isolation a single
        job that raises — exhausted retries (``TaskNotCompletedError``), a
        per-attempt wall-clock timeout, or a transient LLM/tooling error —
        aborts the enclosing loop and discards every *other* job's annotations.

        By default (``skip_on_failure=True``) such a failure is logged,
        surfaced as a ``task_skipped`` event, and swallowed so the remaining
        jobs still run and their artifacts are preserved. Pass
        ``skip_on_failure=False`` to restore fail-fast behaviour.

        Returns the awaitable's result on success, or ``default`` when skipped.
        ``asyncio.CancelledError`` / ``KeyboardInterrupt`` are always
        re-raised — cancellation is not a job failure.
        """
        import logging

        try:
            return await awaitable
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            if not skip_on_failure:
                raise
            logging.getLogger(self.__class__.__module__).warning(
                "skipping job %s — failed: %s", job_name, exc, exc_info=True
            )
            await self.emit_task_skipped(
                on_event, job_name, reason=f"job_failed: {exc}"
            )
            return default


async def persist_seed_artifact(ctx: WorkflowContext, filename: str) -> None:
    """Save ctx.artifact (text) under `filename` if provided."""
    if not ctx.artifact:
        return
    await ctx.artifact_service.save_artifact(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        filename=filename,
        artifact=types.Part.from_text(text=ctx.artifact),
    )


def get_workflows() -> dict[str, type[Workflow]]:
    from .exploitability import ExploitabilityWorkflow
    from .likec4_building import LikeC4BuildingWorkflow
    from .oas_building import OasBuildingWorkflow
    from .oas_enrichment import OasEnrichmentWorkflow
    from .router import RouterWorkflow
    from .trace_annotation import TraceAnnotationWorkflow
    from .trace_annotation_direct import TraceAnnotationDirectWorkflow
    from .trace_graph import TraceGraphWorkflow
    from .trace_graph_pathpar import TraceGraphPathParWorkflow
    from .trace_postdiff import TracePostDiffWorkflow
    from .trace_verify import TraceVerifyWorkflow
    from .vuln_assess import VulnAssessWorkflow
    from .vuln_scan import VulnScanWorkflow
    from .vuln_scan_fast import VulnScanFastWorkflow
    from .vuln_scan_trace import VulnScanTraceWorkflow
    from .vuln_sweep import VulnSweepWorkflow

    return {
        "oas_build": OasBuildingWorkflow,
        "oas_update": OasEnrichmentWorkflow,
        "exploit": ExploitabilityWorkflow,
        "likec4": LikeC4BuildingWorkflow,
        "trace": TraceAnnotationWorkflow,
        "trace-direct": TraceAnnotationDirectWorkflow,
        "trace-graph": TraceGraphWorkflow,
        "trace-graph-pathpar": TraceGraphPathParWorkflow,
        "trace-postdiff": TracePostDiffWorkflow,
        "trace-verify": TraceVerifyWorkflow,
        "vuln-assess": VulnAssessWorkflow,
        "vuln-scan": VulnScanWorkflow,
        "vuln-scan-fast": VulnScanFastWorkflow,
        "vuln-scan-trace": VulnScanTraceWorkflow,
        "vuln-sweep": VulnSweepWorkflow,
        "router": RouterWorkflow,
    }


__all__ = [
    "Workflow",
    "WorkflowContext",
    "get_workflows",
    "persist_seed_artifact",
]
