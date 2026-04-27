from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol

from google.adk.artifacts import BaseArtifactService
from google.genai import types

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


class Pipeline(Protocol):
    async def run(
        self,
        *,
        user_id: str = ...,
        on_event: Optional[TaskRunnerEventHandler] = ...,
    ) -> Any: ...


PipelineBuilder = Callable[[PipelineContext], Awaitable[Pipeline]]


@dataclass(frozen=True)
class PipelineSpec:
    builder: PipelineBuilder


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


def get_pipelines() -> dict[str, PipelineSpec]:
    from .oas_building import oas_building_pipeline
    from .oas_enrichment import oas_enrichment_pipeline
    from .trace_annotation import trace_annotation_pipeline

    return {
        "build": PipelineSpec(builder=oas_building_pipeline),
        "enrich": PipelineSpec(builder=oas_enrichment_pipeline),
        "trace": PipelineSpec(builder=trace_annotation_pipeline),
    }


__all__ = [
    "PipelineContext",
    "Pipeline",
    "PipelineBuilder",
    "PipelineSpec",
    "get_pipelines",
    "persist_seed_artifact",
]
