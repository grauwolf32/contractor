import yaml
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from functools import partial
from google.genai import types
from contractor.tools.fs import RootedLocalFileSystem, MemoryOverlayFileSystem
from contractor.tools.openapi import resolve_refs
from google.adk.artifacts import BaseArtifactService
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.task_runner import (
    TaskRunner,
    TaskRunnerEventHandler,
)
from google.adk.models import LiteLlm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class OpenApiOperation:
    operation_id: str
    method: str
    path: str
    schema: dict[str, Any]


@dataclass
class OpenApiPath:
    path: str
    operations: list[OpenApiOperation] = field(default_factory=list)

    @property
    def path_key(self) -> str:
        """Normalized path key suitable for use in namespaces and refs."""
        return (
            self.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
            or "root"
        )


def extract_openapi_paths(
    openapi: dict[str, Any],
) -> list[OpenApiPath]:
    """Extract all paths and their operations from an OpenAPI schema."""
    paths: list[OpenApiPath] = []
    securitySchemes = openapi.get("components", {}).get("securitySchemes", {})

    for path, path_item in openapi["paths"].items():
        path_files: list[str] | None = path_item.pop("x-path-files", None)
        api_path = OpenApiPath(path=path)

        for method, operation in path_item.items():
            if method in {"get", "post", "put", "delete", "patch"}:
                try:
                    operation_schema = resolve_refs(
                        operation,
                        openapi,
                    )
                except Exception as exc:
                    logger.error(
                        f"Error resolving refs for operation {operation.get('operationId', '')}: {exc}"
                    )
                    continue

                if path_files:
                    operation_schema["x-path-files"] = path_files

                if securitySchemes:
                    operation_schema.setdefault("components", {})
                    operation_schema["components"].setdefault("securitySchemes", {})
                    operation_schema["components"]["securitySchemes"].update(
                        securitySchemes
                    )

                api_path.operations.append(
                    OpenApiOperation(
                        operation_id=operation.get("operationId", ""),
                        method=method,
                        path=path,
                        schema=operation_schema,
                    )
                )

        if api_path.operations:
            paths.append(api_path)

    total_ops = sum(len(p.operations) for p in paths)
    logger.info(f"Found {len(paths)} paths with {total_ops} operations")
    return paths


@dataclass
class AnnotationRunner:
    app_name: str
    llm: LiteLlm
    fs: MemoryOverlayFileSystem
    artifact_service: BaseArtifactService
    paths: list[OpenApiPath] = field(default_factory=list)
    namespace: str = "openapi"
    folder: str = "/"
    overlayfs: MemoryOverlayFileSystem = field(init=False)

    def __post_init__(self):
        self.overlayfs = MemoryOverlayFileSystem(fs=self.fs)

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ):
        raw = await self.artifact_service.load_artifact(
            app_name=self.app_name,
            user_id=user_id,
            filename=f"oas-{self.namespace}-building",
        )
        if not raw:
            raise ValueError("No OpenAPI artifact found")

        openapi = yaml.safe_load(raw.text)
        self.paths = extract_openapi_paths(openapi=openapi)

        for api_path in self.paths:
            fs_state_artifact = await self.artifact_service.load_artifact(
                app_name=self.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-fs",
            )
            if fs_state_artifact:
                self.overlayfs.load(json.loads(fs_state_artifact.text))

            await self.run_path_analysis(
                api_path,
                user_id=user_id,
                on_event=on_event,
            )

            artifact_text = types.Part.from_text(text=json.dumps(self.overlayfs.save()))
            await self.artifact_service.save_artifact(
                app_name=self.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-fs",
                artifact=artifact_text,
            )

            artifact_text = types.Part.from_text(
                text=self.overlayfs.diff(context_lines=4)
            )
            await self.artifact_service.save_artifact(
                app_name=self.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-diff",
                artifact=artifact_text,
            )

    async def run_path_analysis(
        self,
        api_path: OpenApiPath,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ):
        trace_builder = partial(
            build_trace_agent,
            name="trace_agent",
            fs=self.overlayfs,
            model=self.llm,
            max_tokens=160000,
            enable_vuln_reporting=True,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=self.artifact_service,
        )

        runner.add_variable(name="project_path", value=self.folder)

        path_namespace = f"trace-annotation:{self.namespace}:{api_path.path_key}"

        for operation in api_path.operations:
            operation_id = operation.operation_id
            operation_schema = yaml.safe_dump(
                {operation.path: {operation.method: operation.schema}}, sort_keys=False
            )

            runner.add_variable(name="operation_id", value=operation_id)
            runner.add_variable(name="operation_schema", value=operation_schema)

            runner.add_task(
                name="trace_annotation",
                ref=f"trace_annotation:{self.namespace}:{operation_id}",
                worker_builder=trace_builder,
                iterations=1,
                max_attempts=3,
                max_steps=20,
                artifacts=[],
                namespace=path_namespace,
                model=self.llm,
            )

        await runner.run(user_id=user_id, on_event=on_event)


async def trace_annotation_pipeline(
    project_path: Path,
    folder_name: str,
    model: str,
    app_name: str,
    user_id: str,
    artifact_service: BaseArtifactService,
    artifact: Optional[str] = None,
    **kwargs,
) -> AnnotationRunner:
    base_fs = RootedLocalFileSystem(root_path=project_path)
    llm = LiteLlm(model=model)

    if artifact:
        artifact_text = types.Part.from_text(text=artifact)
        await artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            filename="oas-openapi-building",
            artifact=artifact_text,
        )

    return AnnotationRunner(
        app_name=app_name,
        llm=llm,
        fs=base_fs,
        artifact_service=artifact_service,
        folder=folder_name,
    )
