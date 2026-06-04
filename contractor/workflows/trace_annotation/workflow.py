import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import yaml
from google.genai import types

from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.tools.openapi import resolve_refs
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext, persist_seed_artifact
from contractor.workflows.config import WorkflowConfig

CFG = WorkflowConfig.load(__file__)

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


class TraceAnnotationWorkflow(Workflow):
    namespace: str = "openapi"

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)
        self.fs = ctx.fs
        self.overlayfs = MemoryOverlayFileSystem(fs=self.fs)
        self.paths: list[OpenApiPath] = []
        self._overlay_seeded = False

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any:
        ctx = self.ctx
        await persist_seed_artifact(ctx, filename="oas-openapi-building")

        raw = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"oas-{self.namespace}-building",
        )
        if not raw:
            raise ValueError("No OpenAPI artifact found")

        openapi = yaml.safe_load(raw.text or "")
        self.paths = extract_openapi_paths(openapi=openapi)

        fs_state_artifact = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-fs",
        )
        if fs_state_artifact:
            self.overlayfs.load(json.loads(fs_state_artifact.text or "{}"))
        self._overlay_seeded = True

        for api_path in self.paths:
            await self._run_path_analysis(
                api_path,
                user_id=user_id,
                on_event=on_event,
            )

    async def _cleanup(self, *, user_id: str) -> None:
        if not self._overlay_seeded:
            return
        ctx = self.ctx
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-fs",
            artifact=types.Part.from_text(text=json.dumps(self.overlayfs.save())),
        )
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-diff",
            artifact=types.Part.from_text(
                text=self.overlayfs.diff(context_lines=4)
            ),
        )

    async def _run_path_analysis(
        self,
        api_path: OpenApiPath,
        *,
        user_id: str = "cli-user",
        on_event: TaskRunnerEventHandler | None = None,
    ) -> None:
        if not api_path.operations:
            return

        ctx = self.ctx
        trace_builder = partial(
            build_trace_agent,
            name="trace_agent",
            _format=CFG.agent("trace_agent").output_format,
            fs=self.overlayfs,
            model=self.llm,
            max_tokens=CFG.budgets.max_tokens,
            enable_vuln_reporting=True,
            with_graph_tools=CFG.agent("trace_agent").with_graph_tools,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=CFG.observations,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        operation_ids, operation_schema_yaml = self._build_path_task_payload(api_path)
        # Per-path memory namespace (matches trace_annotation_direct / trace_graph).
        # A shared namespace accumulated every path's notes + the injected skill
        # bodies into one ever-growing store (O(n²) context across paths), which
        # was the dominant cost on large specs; skills are re-injected per path
        # via add_task(skills=...).
        workflow_namespace = f"trace-annotation:{self.namespace}:{api_path.path_key}"

        runner.add_variable(name="operation_id", value=operation_ids)
        runner.add_variable(name="operation_schema", value=operation_schema_yaml)

        runner.add_task(
            name="trace_annotation",
            ref=f"trace_annotation:{self.namespace}:{api_path.path_key}",
            worker_builder=trace_builder,
            **CFG.tasks.annotate.as_kwargs(),
            artifacts=[],
            skills=["trace"],
            namespace=workflow_namespace,
            model=self.llm,
        )

        await runner.run(user_id=user_id, on_event=on_event)

    @staticmethod
    def _build_path_task_payload(api_path: OpenApiPath) -> tuple[str, str]:
        """Collapse all operations under ``api_path`` into a single task payload.

        Returns a (operation_ids, operation_schema_yaml) tuple where operation_ids
        is a comma-separated list and the schema YAML keeps shared
        ``x-path-files`` / ``components`` at the path-item / document level
        instead of duplicating them per method.
        """
        methods: dict[str, Any] = {}
        shared_path_files: list[str] | None = None
        shared_security_schemes: dict[str, Any] = {}

        for operation in api_path.operations:
            schema = dict(operation.schema)
            path_files = schema.pop("x-path-files", None)
            components = schema.pop("components", None)

            if path_files and shared_path_files is None:
                shared_path_files = path_files
            if components:
                schemes = components.get("securitySchemes") or {}
                if schemes:
                    shared_security_schemes.update(schemes)

            methods[operation.method] = schema

        if shared_path_files is not None:
            methods["x-path-files"] = shared_path_files

        doc: dict[str, Any] = {api_path.path: methods}
        if shared_security_schemes:
            doc["components"] = {"securitySchemes": shared_security_schemes}

        operation_ids = ", ".join(op.operation_id for op in api_path.operations)
        operation_schema_yaml = yaml.safe_dump(doc, sort_keys=False)
        return operation_ids, operation_schema_yaml
