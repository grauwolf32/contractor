from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fsspec import AbstractFileSystem
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from contractor.runners.models import TaskRunnerEvent
from contractor.runners.task_runner import TaskRunnerEventHandler


@dataclass(frozen=True)
class PipelineContext:
    project_path: Path
    folder_name: str
    model: str
    app_name: str
    user_id: str
    artifact_service: BaseArtifactService
    fs: AbstractFileSystem
    artifact: Optional[str] = None
    prompt: Optional[str] = None
    checkpoint_path: Optional[Path] = None


class Pipeline(ABC):
    """Base class for all pipelines.

    Subclasses receive a ``PipelineContext`` at construction time and
    implement ``_run_impl()``. The public ``run()`` wraps it with
    ``pipeline_started`` / ``pipeline_finished`` lifecycle events so
    the UI shows activity immediately, even when sub-agent setup is
    slow.
    """

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> Any:
        await self._emit(on_event, "pipeline_started", phase="initializing")
        ok = False
        try:
            result = await self._run_impl(user_id=user_id, on_event=on_event)
            ok = True
            return result
        finally:
            try:
                await self._cleanup(user_id=user_id)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("_cleanup failed")
            await self._emit(
                on_event,
                "pipeline_finished",
                ok=ok,
            )

    @abstractmethod
    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> Any: ...

    async def _cleanup(self, *, user_id: str) -> None:
        """Hook for subclasses to persist supplementary state (overlay FS, etc.).

        Called in ``run()``'s finally block, before the ``pipeline_finished``
        event. The default implementation is a no-op.
        """

    async def _emit(
        self,
        on_event: Optional[TaskRunnerEventHandler],
        event_type: str,
        **payload: Any,
    ) -> None:
        if on_event is None:
            return
        await on_event(
            TaskRunnerEvent(
                type=event_type,
                task_name=self.name,
                task_id=-1,
                payload={"pipeline": self.name, **payload},
            )
        )

    async def artifact_exists(self, *, user_id: str, filename: str) -> bool:
        """True iff a non-empty artifact is already stored under ``filename``.

        Useful for pipelines that want to skip an upstream analysis task when
        a previous run (or a sibling pipeline) has already produced its
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
        on_event: Optional[TaskRunnerEventHandler],
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


async def persist_seed_artifact(ctx: PipelineContext, filename: str) -> None:
    """Save ctx.artifact (text) under `filename` if provided."""
    if not ctx.artifact:
        return
    await ctx.artifact_service.save_artifact(
        app_name=ctx.app_name,
        user_id=ctx.user_id,
        filename=filename,
        artifact=types.Part.from_text(text=ctx.artifact),
    )


def get_pipelines() -> dict[str, type[Pipeline]]:
    from .exploitability import ExploitabilityPipeline
    from .likec4_building import LikeC4BuildingPipeline
    from .oas_building import OasBuildingPipeline
    from .oas_enrichment import OasEnrichmentPipeline
    from .router import RouterPipeline
    from .trace_annotation import TraceAnnotationPipeline
    from .trace_annotation_direct import TraceAnnotationDirectPipeline
    from .trace_graph import TraceGraphPipeline
    from .trace_graph_pathpar import TraceGraphPathParPipeline
    from .trace_verify import TraceVerifyPipeline
    from .vuln_scan import VulnScanPipeline

    return {
        "build": OasBuildingPipeline,
        "enrich": OasEnrichmentPipeline,
        "exploit": ExploitabilityPipeline,
        "likec4": LikeC4BuildingPipeline,
        "trace": TraceAnnotationPipeline,
        "trace-direct": TraceAnnotationDirectPipeline,
        "trace-graph": TraceGraphPipeline,
        "trace-graph-pathpar": TraceGraphPathParPipeline,
        "trace-verify": TraceVerifyPipeline,
        "vuln-scan": VulnScanPipeline,
        "router": RouterPipeline,
    }


__all__ = [
    "Pipeline",
    "PipelineContext",
    "get_pipelines",
    "persist_seed_artifact",
]
