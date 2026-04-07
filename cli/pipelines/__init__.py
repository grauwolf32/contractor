from dataclasses import dataclass, field
from typing import Callable, Awaitable
from contractor.runners.task_runner import TaskRunner

from .oas_building import oas_building_pipeline
from .oas_enrichment import oas_enrichment_pipeline
from .trace_annotation import trace_annotation_pipeline


@dataclass(frozen=True)
class PipelineSpec:
    builder: Callable[..., Awaitable[TaskRunner]]
    required: list[str] = field(default_factory=list)


def get_pipelines() -> dict[str, PipelineSpec]:
    return {
        "build": PipelineSpec(
            builder=oas_building_pipeline, required=["project_path", "folder_name"]
        ),
        "enrich": PipelineSpec(
            builder=oas_enrichment_pipeline, required=["project_path", "folder_name"]
        ),
        "trace": PipelineSpec(
            builder=trace_annotation_pipeline, required=["project_path", "folder_name"]
        ),
    }


__all__ = ["get_pipelines", "PipelineSpec"]
