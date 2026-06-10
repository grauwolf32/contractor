"""Combined BFS→DFS vulnerability workflow.

Phase 1 (BFS): codereview_agent sweeps the codebase for findings.
Phase 2 (DFS): trace_agent deep-traces each reported finding to
produce annotated evidence chains.

The output is a vulnerability report enriched with per-finding
trace annotations and control checklists.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any

import yaml

from contractor.agents.codereview_agent.agent import build_codereview_agent
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.artifacts import artifact_key_slug
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext
from contractor.workflows.config import WorkflowConfig

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)


class VulnScanTraceWorkflow(Workflow):
    """BFS discovery → DFS confirmation workflow."""

    namespace: str = "vuln-scan-trace"

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any:
        ctx = self.ctx

        # ── Phase 1: BFS scan ───────────────────────────────────────────
        scan_namespace = f"{self.namespace}:scan"

        scan_builder = partial(
            build_codereview_agent,
            name="codereview_agent",
            _format=CFG.agent("codereview_agent").output_format,
            fs=ctx.fs,
            model=self.llm,
            max_tokens=CFG.budgets.scan_max_tokens,
            with_graph_tools=CFG.agent("codereview_agent").with_graph_tools,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=CFG.observations,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="vuln_scan",
            ref="vuln-scan-trace:scan",
            worker_builder=scan_builder,
            **CFG.tasks.scan.as_kwargs(),
            namespace=scan_namespace,
            skills=["vuln_scan"],
            model=self.llm,
            params={"project_path": ctx.folder_name},
        )

        await runner.run(user_id=user_id, on_event=on_event)

        # ── Load scan results ───────────────────────────────────────────
        findings = await self._load_findings(
            user_id=user_id,
            namespace=scan_namespace,
        )

        if not findings:
            logger.warning("vuln_scan found no findings — skipping trace phase")
            return

        logger.info("vuln_scan found %d findings — starting trace phase", len(findings))

        # ── Phase 2: DFS trace per finding ──────────────────────────────
        for finding in findings:
            await self._trace_finding(
                finding=finding,
                user_id=user_id,
                on_event=on_event,
            )

    async def _trace_finding(
        self,
        *,
        finding: dict[str, Any],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        name = finding.get("name", "")
        place = finding.get("place", "")
        title = finding.get("title", "")

        if not name or not place:
            return

        trace_namespace = f"{self.namespace}:trace:{name}"
        ctx = self.ctx

        trace_builder = partial(
            build_trace_agent,
            name="trace_agent",
            _format=CFG.agent("trace_agent").output_format,
            fs=ctx.fs,
            model=self.llm,
            max_tokens=CFG.budgets.trace_max_tokens,
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

        # Build trace task instructions from the finding
        details = finding.get("details", "")
        summary = finding.get("summary", "")
        severity = finding.get("severity", "medium")

        operation_schema = (
            f"Vulnerability: {title}\n"
            f"File: {place}\n"
            f"Severity: {severity}\n"
            f"Summary: {summary}\n"
            f"Details: {details}"
        )

        runner.add_task(
            name="trace_annotation",
            ref=f"vuln-scan-trace:trace:{name}",
            # Unique, stable per-finding publish key — one trace task is queued
            # per finding and the shared template key would make each finding
            # overwrite the previous one's artifacts.
            artifact_key=f"trace_annotation/{artifact_key_slug(name)}",
            worker_builder=trace_builder,
            **CFG.tasks.trace.as_kwargs(),
            namespace=trace_namespace,
            skills=["trace"],
            model=self.llm,
            params={
                "operation_id": name,
                "operation_schema": operation_schema,
            },
        )

        try:
            await runner.run(user_id=user_id, on_event=on_event)
        except Exception as exc:
            logger.warning("trace for %s failed: %s", name, exc)

    async def _load_findings(
        self,
        *,
        user_id: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """Load vulnerability reports from the scan phase artifacts."""
        artifact_key = f"user:vulnerability-reports/{namespace}"
        part = await self.ctx.artifact_service.load_artifact(
            app_name=self.ctx.app_name,
            user_id=user_id,
            filename=artifact_key,
        )
        if part is None or not getattr(part, "text", None):
            return []

        try:
            raw = yaml.safe_load(part.text or "") or {}
        except yaml.YAMLError as exc:
            logger.warning("could not parse scan results: %s", exc)
            return []

        if not isinstance(raw, dict):
            return []

        findings: list[dict[str, Any]] = []
        for name, item in raw.items():
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry.setdefault("name", name)
            findings.append(entry)

        # Sort by severity: critical first
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        findings.sort(key=lambda f: sev_order.get(f.get("severity", "low"), 4))

        return findings
