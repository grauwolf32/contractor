import json
import logging
from typing import Any, Optional
from uuid import uuid4

import yaml
from google.adk.models import LiteLlm
from google.genai import types

from cli.pipelines import Pipeline, PipelineContext, persist_seed_artifact
from cli.pipelines.trace_annotation import (OpenApiOperation, OpenApiPath,
                                            extract_openapi_paths)
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import (RenderedTask, TaskRunnerEventHandler,
                                       TaskTemplate)
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.tools.fs import MemoryOverlayFileSystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


TRACE_TASK_TEMPLATE: str = "trace_annotation"
TRACE_MAX_TOKENS: int = 100_000


class TraceAnnotationDirectPipeline(Pipeline):
    """Variant of TraceAnnotationPipeline that skips the planning_agent layer.

    Each OpenAPI operation is rendered directly from its task template and
    handed to a freshly built ``trace_agent`` via ``AgentRunner``. The pipeline
    keeps the same overlay-FS / artifact contract as ``TraceAnnotationPipeline``
    so that produced annotations can be compared head-to-head.
    """

    namespace: str = "openapi"

    def __init__(self, ctx: PipelineContext) -> None:
        super().__init__(ctx)
        self.llm = LiteLlm(model=ctx.model, timeout=ctx.timeout)
        self.fs = ctx.fs
        self.overlayfs = MemoryOverlayFileSystem(fs=self.fs)
        self.paths: list[OpenApiPath] = []
        self._template = TaskTemplate.load(TRACE_TASK_TEMPLATE)
        self._runner = AgentRunner(
            name=ctx.app_name,
            artifact_service=ctx.artifact_service,
        )

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
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

        for api_path in self.paths:
            fs_state_artifact = await ctx.artifact_service.load_artifact(
                app_name=ctx.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-fs",
            )
            if fs_state_artifact:
                self.overlayfs.load(json.loads(fs_state_artifact.text or "{}"))

            await self._run_path_analysis(
                api_path,
                user_id=user_id,
                on_event=on_event,
            )

            artifact_text = types.Part.from_text(text=json.dumps(self.overlayfs.save()))
            await ctx.artifact_service.save_artifact(
                app_name=ctx.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-fs",
                artifact=artifact_text,
            )

            artifact_text = types.Part.from_text(
                text=self.overlayfs.diff(context_lines=4)
            )
            await ctx.artifact_service.save_artifact(
                app_name=ctx.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-diff",
                artifact=artifact_text,
            )

    async def _run_path_analysis(
        self,
        api_path: OpenApiPath,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> None:
        path_namespace = f"trace-annotation:{self.namespace}:{api_path.path_key}"
        base_variables: dict[str, Any] = {"project_path": self.ctx.folder_name}

        await inject_skills(
            ["trace"],
            namespace=path_namespace,
            artifact_service=self.ctx.artifact_service,
            app_name=self.ctx.app_name,
            user_id=user_id,
        )

        for idx, operation in enumerate(api_path.operations):
            await self._run_operation_trace(
                operation=operation,
                idx=idx,
                namespace=path_namespace,
                base_variables=base_variables,
                user_id=user_id,
                on_event=on_event,
            )

    async def _run_operation_trace(
        self,
        *,
        operation: OpenApiOperation,
        idx: int,
        namespace: str,
        base_variables: dict[str, Any],
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> None:
        operation_schema = yaml.safe_dump(
            {operation.path: {operation.method: operation.schema}},
            sort_keys=False,
        )

        rendered = RenderedTask.from_template(
            template=self._template,
            variables={
                **base_variables,
                "operation_id": operation.operation_id,
                "operation_schema": operation_schema,
            },
            params={},
            artifacts={},
        )

        agent = build_trace_agent(
            name="trace_agent",
            fs=self.overlayfs,
            namespace=namespace,
            _format=self._template.format,
            model=self.llm,
            max_tokens=TRACE_MAX_TOKENS,
            enable_vuln_reporting=True,
            with_graph_tools=True,
        )

        session_id = uuid4().hex
        event_name = f"trace_annotation:{self.namespace}:{operation.operation_id}"
        plugins = [
            AdkTracePlugin(
                task_name=event_name,
                task_id=idx,
                iteration=1,
                session_id=session_id,
                emit=self._runner._emit,
            ),
            AdkMetricsPlugin(
                task_name=event_name,
                task_id=idx,
                iteration=1,
                session_id=session_id,
                emit=self._runner._emit,
            ),
        ]

        await self._runner.run(
            agent=agent,
            message=rendered._format_task(),
            user_id=user_id,
            session_id=session_id,
            initial_state={},
            plugins=plugins,
            on_event=on_event,
            event_name=event_name,
        )
