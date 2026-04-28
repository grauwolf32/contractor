from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    artifact: Optional[str] = None
    prompt: Optional[str] = None


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
    from .oas_building import OasBuildingPipeline
    from .oas_enrichment import OasEnrichmentPipeline
    from .router import RouterPipeline
    from .trace_annotation import TraceAnnotationPipeline

    return {
        "build": OasBuildingPipeline,
        "enrich": OasEnrichmentPipeline,
        "trace": TraceAnnotationPipeline,
        "router": RouterPipeline,
    }


__all__ = [
    "Pipeline",
    "PipelineContext",
    "get_pipelines",
    "persist_seed_artifact",
]
