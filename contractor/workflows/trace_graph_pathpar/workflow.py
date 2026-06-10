"""Path-level parallel trace-annotation workflow.

All API paths run concurrently (bounded by a semaphore), each in its
own forked ``MemoryOverlayFileSystem``.  Operations *within* a path
remain sequential — the overlay is shared across sibling operations so
later operations see earlier annotations, just like ``trace-graph``.

After every path completes, the per-path overlays are merged back into
the workflow's main overlay and persisted as a single diff artifact.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast
from uuid import uuid4

import yaml
from google.genai import types

from contractor.agents.trace_agent.agent import TraceFormat, build_trace_agent
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import RenderedTask, TaskRunnerEventHandler, TaskTemplate
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.tools.code import attach_graph_tools_if_local
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.tools.fs.merge import fork_overlay, merge_overlay_forks
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext, persist_seed_artifact
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.namespaces import TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX
from contractor.workflows.path_groups import PathGroup, group_paths_by_prefix
from contractor.workflows.trace_annotation import (
    OpenApiOperation,
    OpenApiPath,
    extract_openapi_paths,
)

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)

TRACE_TASK_TEMPLATE: str = "trace_annotation"

# Per-path namespace prefix used for this workflow's trace artifacts and
# vulnerability reports. Shared (via contractor.workflows.namespaces) with
# vuln_assess._collect_vuln_reports and trace_verify so the write key (here)
# and the read keys (there) cannot drift apart.
PATH_NAMESPACE_PREFIX: str = TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX


class TraceGraphPathParWorkflow(Workflow):
    """Path-level parallel variant of ``TraceGraphWorkflow``.

    Identical annotation semantics — every operation gets the same
    prompt, tools, and overlay-FS contract — but independent API paths
    execute concurrently (up to ``max_concurrency``).
    """

    namespace: str = "openapi"

    def __init__(
        self,
        ctx: WorkflowContext,
        *,
        max_concurrency: int = CFG.budgets.max_concurrency,
    ) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)
        self.fs = ctx.fs
        self.overlayfs = MemoryOverlayFileSystem(fs=self.fs)
        self.paths: list[OpenApiPath] = []
        self._template = TaskTemplate.load(TRACE_TASK_TEMPLATE)
        self.max_concurrency = max_concurrency

    # ── public workflow entry ─────────────────────────────────────────

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

        # Resume: load existing overlay state if present.
        fs_state_artifact = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-fs",
        )
        if fs_state_artifact:
            self.overlayfs.load(json.loads(fs_state_artifact.text or "{}"))

        # Build graph tools once — trailmark parses the base FS (read-only),
        # so the tool closures are safe to share across parallel forks.
        self._shared_graph_tools = attach_graph_tools_if_local(self.overlayfs)

        # Group by route prefix — the group is the fork/concurrency unit and
        # the memory-namespace unit, so sibling paths share context and a
        # budget instead of competing as N independent runs. group_depth=0
        # keeps the historical one-fork-per-path behavior.
        groups = group_paths_by_prefix(
            self.paths, depth=CFG.budgets.group_depth
        )

        # Snapshot pre-fork state for merge.
        pre_fork_patch = self.overlayfs.save()
        pre_fork_files = dict(self.overlayfs._files)

        forks = [fork_overlay(self.fs, pre_fork_patch) for _ in groups]
        sem = asyncio.Semaphore(self.max_concurrency)

        logger.info(
            "Running %d route group(s) covering %d path(s) in parallel "
            "(max_concurrency=%d, group_depth=%d)",
            len(groups),
            len(self.paths),
            self.max_concurrency,
            CFG.budgets.group_depth,
        )

        async def _run_group(group: PathGroup, overlay: MemoryOverlayFileSystem) -> None:
            async with sem:
                runner = AgentRunner(
                    name=ctx.app_name,
                    artifact_service=ctx.artifact_service,
                )
                await self._run_group_analysis(
                    group=group,
                    overlay=overlay,
                    runner=runner,
                    user_id=user_id,
                    on_event=on_event,
                )

        # Merge + persist in a `finally` (the trace_annotation `_cleanup`
        # precedent — done inline here because vuln_assess drives this
        # workflow via `_run_impl` directly, bypassing `run()`/`_cleanup`):
        # a single failed path makes the TaskGroup cancel its siblings and
        # re-raise, and without the `finally` every already-completed path's
        # annotations were lost. On the happy path this block is the one and
        # only merge/save, so nothing runs twice.
        try:
            async with asyncio.TaskGroup() as tg:
                for group, overlay in zip(groups, forks, strict=False):
                    tg.create_task(_run_group(group, overlay))
        finally:
            conflicts = merge_overlay_forks(self.overlayfs, forks, pre_fork_files)
            if conflicts:
                logger.warning(
                    "Overlay merge produced %d conflicting files: %s",
                    len(conflicts),
                    conflicts,
                )

            await self._save_overlay_artifacts(user_id)

    # ── per-group orchestration (operations sequential) ───────────────

    async def _run_group_analysis(
        self,
        *,
        group: PathGroup,
        overlay: MemoryOverlayFileSystem,
        runner: AgentRunner,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        group_namespace = f"{PATH_NAMESPACE_PREFIX}:{self.namespace}:{group.key}"
        base_variables: dict[str, Any] = {"project_path": self.ctx.folder_name}

        await inject_skills(
            ["trace"],
            namespace=group_namespace,
            artifact_service=self.ctx.artifact_service,
            app_name=self.ctx.app_name,
            user_id=user_id,
        )

        for idx, operation in enumerate(group.operations):
            await self._run_operation_trace(
                operation=operation,
                idx=idx,
                namespace=group_namespace,
                overlay=overlay,
                runner=runner,
                base_variables=base_variables,
                user_id=user_id,
                on_event=on_event,
            )

    # ── single operation ──────────────────────────────────────────────

    async def _run_operation_trace(
        self,
        *,
        operation: OpenApiOperation,
        idx: int,
        namespace: str,
        overlay: MemoryOverlayFileSystem,
        runner: AgentRunner,
        base_variables: dict[str, Any],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
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
            fs=overlay,
            namespace=namespace,
            _format=cast(TraceFormat, self._template.format),
            model=self.llm,
            max_tokens=CFG.budgets.max_tokens,
            enable_vuln_reporting=True,
            graph_tools=self._shared_graph_tools,
        )

        session_id = uuid4().hex
        event_name = (
            f"trace_graph_pathpar:{self.namespace}:{operation.operation_id}"
        )
        plugins = [
            AdkTracePlugin(
                task_name=event_name,
                task_id=idx,
                iteration=1,
                session_id=session_id,
                emit=runner._emit,
            ),
            AdkMetricsPlugin(
                task_name=event_name,
                task_id=idx,
                iteration=1,
                session_id=session_id,
                emit=runner._emit,
            ),
        ]

        await runner.run(
            agent=agent,
            message=rendered._format_task(),
            user_id=user_id,
            session_id=session_id,
            initial_state={},
            plugins=plugins,
            on_event=on_event,
            event_name=event_name,
        )

    # ── artifact persistence ──────────────────────────────────────────

    async def _save_overlay_artifacts(self, user_id: str) -> None:
        ctx = self.ctx
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-fs",
            artifact=types.Part.from_text(
                text=json.dumps(self.overlayfs.save()),
            ),
        )
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=f"trace-{self.namespace}-diff",
            artifact=types.Part.from_text(
                text=self.overlayfs.diff(context_lines=4),
            ),
        )
