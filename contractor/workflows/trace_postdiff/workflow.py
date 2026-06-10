"""Two-stage trace workflow: annotate-only trace, then post-diff analytics.

Stage A runs the per-operation ``trace_agent`` loop exactly like
``trace-graph``, but with vulnerability reporting disabled — the agent's
whole job is to drive ``@trace`` / ``@validate`` / ``@sink`` annotations
onto the execution paths (navigation).

Stage B runs once per path: ``vuln_analytics_agent`` receives the
annotation diff produced during that path's trace runs and judges the
annotated flows against the finding-shape taxonomy (judgement),
persisting supported findings via ``report_vulnerability`` under the
``trace-postdiff:{namespace}:{path_key}`` namespace — so ``trace-verify``
and ``vuln-assess`` pick them up through the shared prefix registry.

The split targets small models: a single agent asked to both navigate
*and* judge tends to do neither well; here each stage does one job.
A/B against the single-stage ``trace-graph`` on the same fixture.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast
from uuid import uuid4

import yaml
from google.genai import types

from contractor.agents.trace_agent.agent import TraceFormat, build_trace_agent
from contractor.agents.vuln_analytics_agent.agent import (
    AnalyticsFormat,
    build_vuln_analytics_agent,
)
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import RenderedTask, TaskRunnerEventHandler, TaskTemplate
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext, persist_seed_artifact
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.namespaces import TRACE_POSTDIFF_NAMESPACE_PREFIX
from contractor.workflows.path_groups import PathGroup, group_paths_by_prefix
from contractor.workflows.trace_annotation import (
    OpenApiOperation,
    OpenApiPath,
    extract_openapi_paths,
)

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)

TRACE_TASK_TEMPLATE: str = "trace_annotation"
ANALYTICS_TASK_TEMPLATE: str = "vuln_analytics"

_DIFF_HEADER_PREFIX = "diff --overlay a"
_DIFF_TRUNCATION_MARKER = (
    "\n... [diff truncated — read the annotated files directly for the rest]"
)


def _diff_header_path(line: str) -> str | None:
    """Parse the file path out of a ``diff --overlay a{path} b{path}`` header.

    The path appears twice, so its length is fixed by the header length —
    no delimiter guessing even for paths containing spaces.
    """
    if not line.startswith(_DIFF_HEADER_PREFIX):
        return None
    rest = line[len(_DIFF_HEADER_PREFIX) :]
    half = (len(rest) - 2) // 2
    if half > 0 and rest[half : half + 2] == " b" and rest[:half] == rest[half + 2 :]:
        return rest[:half]
    # Malformed / unexpected header — fall back to the first " b" split.
    return rest.split(" b", 1)[0] or None


def filter_diff_by_files(diff_text: str, files: set[str]) -> str:
    """Keep only the per-file chunks of an overlay diff whose path is in
    ``files``. Chunks are delimited by ``diff --overlay`` headers."""
    if not diff_text or not files:
        return ""
    keep = False
    out: list[str] = []
    for line in diff_text.splitlines():
        path = _diff_header_path(line)
        if path is not None:
            keep = path in files
        if keep:
            out.append(line)
    return "\n".join(out)


def truncate_diff(diff_text: str, max_chars: int) -> str:
    if len(diff_text) <= max_chars:
        return diff_text
    return diff_text[:max_chars] + _DIFF_TRUNCATION_MARKER


class TracePostDiffWorkflow(Workflow):
    """Annotate-only trace stage + per-path post-diff analytics stage."""

    namespace: str = "openapi"

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)
        self.fs = ctx.fs
        self.overlayfs = MemoryOverlayFileSystem(fs=self.fs)
        self.paths: list[OpenApiPath] = []
        self._template = TaskTemplate.load(TRACE_TASK_TEMPLATE)
        self._analytics_template = TaskTemplate.load(ANALYTICS_TASK_TEMPLATE)
        self._runner = AgentRunner(
            name=ctx.app_name,
            artifact_service=ctx.artifact_service,
        )

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

        # Group by route prefix: the group is the unit of memory namespace,
        # skill injection, and the analytics stage, so sibling paths share
        # discovered context instead of re-navigating from scratch.
        groups = group_paths_by_prefix(
            self.paths, depth=CFG.budgets.group_depth
        )
        logger.info(
            "trace-postdiff: %d paths in %d route group(s) (group_depth=%d)",
            len(self.paths),
            len(groups),
            CFG.budgets.group_depth,
        )

        for group in groups:
            fs_state_artifact = await ctx.artifact_service.load_artifact(
                app_name=ctx.app_name,
                user_id=user_id,
                filename=f"trace-{self.namespace}-fs",
            )
            if fs_state_artifact:
                self.overlayfs.load(json.loads(fs_state_artifact.text or "{}"))

            await self._run_group_analysis(
                group,
                user_id=user_id,
                on_event=on_event,
            )

            await self._save_overlay_artifacts(user_id)

    async def _save_overlay_artifacts(self, user_id: str) -> None:
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

    def _changed_since(self, before: dict[str, bytes]) -> set[str]:
        """Paths whose overlay content was added or modified after the
        ``before`` snapshot. Deletions are ignored — the annotate-only
        stage adds comments; a deletion carries no annotated flow to
        analyze. (``_files`` access mirrors trace_graph_pathpar's
        pre-fork snapshot.)"""
        return {
            path
            for path, content in self.overlayfs._files.items()  # noqa: SLF001
            if before.get(path) != content
        }

    async def _run_group_analysis(
        self,
        group: PathGroup,
        *,
        user_id: str = "cli-user",
        on_event: TaskRunnerEventHandler | None = None,
    ) -> None:
        group_namespace = (
            f"{TRACE_POSTDIFF_NAMESPACE_PREFIX}:{self.namespace}:{group.key}"
        )
        base_variables: dict[str, Any] = {"project_path": self.ctx.folder_name}

        await inject_skills(
            ["trace"],
            namespace=group_namespace,
            artifact_service=self.ctx.artifact_service,
            app_name=self.ctx.app_name,
            user_id=user_id,
        )

        before = dict(self.overlayfs._files)  # noqa: SLF001

        # ── Stage A: annotate-only trace, one run per operation ──────────
        for idx, operation in enumerate(group.operations):
            await self._run_operation_trace(
                operation=operation,
                idx=idx,
                namespace=group_namespace,
                base_variables=base_variables,
                user_id=user_id,
                on_event=on_event,
            )

        # ── Stage B: post-diff analytics over this group's annotations ───
        changed = self._changed_since(before)
        if not changed:
            logger.info(
                "trace-postdiff: no annotations produced for group %r — "
                "skipping analytics stage",
                group.key,
            )
            return

        await self._run_group_analytics(
            group,
            namespace=group_namespace,
            changed_files=changed,
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
            fs=self.overlayfs,
            namespace=namespace,
            _format=cast(TraceFormat, self._template.format),
            model=self.llm,
            max_tokens=CFG.budgets.max_tokens,
            # Annotate-only: judgement is the analytics stage's job.
            enable_vuln_reporting=False,
            with_graph_tools=CFG.agent("trace_agent").with_graph_tools,
        )

        session_id = uuid4().hex
        event_name = f"trace_postdiff:{self.namespace}:{operation.operation_id}"
        await self._runner.run(
            agent=agent,
            message=rendered._format_task(),
            user_id=user_id,
            session_id=session_id,
            initial_state={},
            plugins=self._plugins(event_name, idx, session_id),
            on_event=on_event,
            event_name=event_name,
        )

    async def _run_group_analytics(
        self,
        group: PathGroup,
        *,
        namespace: str,
        changed_files: set[str],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        group_diff = filter_diff_by_files(
            self.overlayfs.diff(context_lines=4), changed_files
        )
        group_diff = truncate_diff(
            group_diff, CFG.budgets.analytics_diff_max_chars
        )

        target_summary = yaml.safe_dump(
            {
                path.path: {op.method: op.schema for op in path.operations}
                for path in group.paths
            },
            sort_keys=False,
        )

        rendered = RenderedTask.from_template(
            template=self._analytics_template,
            variables={
                "target_summary": target_summary,
                "trace_diff": group_diff,
            },
            params={},
            artifacts={},
        )

        agent = build_vuln_analytics_agent(
            name="vuln_analytics_agent",
            fs=self.overlayfs,
            namespace=namespace,
            _format=cast(AnalyticsFormat, self._analytics_template.format),
            model=self.llm,
            max_tokens=CFG.budgets.analytics_max_tokens,
            with_graph_tools=CFG.agent("vuln_analytics_agent").with_graph_tools,
        )

        session_id = uuid4().hex
        event_name = (
            f"trace_postdiff:{self.namespace}:{group.key}:analytics"
        )
        await self._runner.run(
            agent=agent,
            message=rendered._format_task(),
            user_id=user_id,
            session_id=session_id,
            initial_state={},
            plugins=self._plugins(event_name, len(group.operations), session_id),
            on_event=on_event,
            event_name=event_name,
        )

    def _plugins(self, event_name: str, task_id: int, session_id: str) -> list:
        return [
            AdkTracePlugin(
                task_name=event_name,
                task_id=task_id,
                iteration=1,
                session_id=session_id,
                emit=self._runner._emit,
            ),
            AdkMetricsPlugin(
                task_name=event_name,
                task_id=task_id,
                iteration=1,
                session_id=session_id,
                emit=self._runner._emit,
            ),
        ]
