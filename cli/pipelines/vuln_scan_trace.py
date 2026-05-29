"""Combined BFS→DFS vulnerability pipeline.

Phase 1 (BFS): vuln_scan_agent sweeps the codebase for findings.
Phase 2 (DFS): trace_agent deep-traces each reported finding to
produce annotated evidence chains.

The output is a vulnerability report enriched with per-finding
trace annotations and control checklists.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

import yaml

from cli.pipelines import Pipeline, PipelineContext
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.agents.vuln_scan_agent.agent import build_vuln_scan_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model

logger = logging.getLogger(__name__)

SCAN_MAX_TOKENS: int = 80_000
TRACE_MAX_TOKENS: int = 80_000


class VulnScanTracePipeline(Pipeline):
    """BFS discovery → DFS confirmation pipeline."""

    namespace: str = "vuln-scan-trace"

    def __init__(self, ctx: PipelineContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> Any:
        ctx = self.ctx

        # ── Phase 1: BFS scan ───────────────────────────────────────────
        scan_namespace = f"{self.namespace}:scan"

        scan_builder = partial(
            build_vuln_scan_agent,
            name="vuln_scan_agent",
            fs=ctx.fs,
            model=self.llm,
            max_tokens=SCAN_MAX_TOKENS,
            with_graph_tools=True,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="vuln_scan",
            ref="vuln-scan-trace:scan",
            worker_builder=scan_builder,
            iterations=1,
            max_attempts=2,
            max_steps=75,
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
        on_event: Optional[TaskRunnerEventHandler],
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
            fs=ctx.fs,
            model=self.llm,
            max_tokens=TRACE_MAX_TOKENS,
            enable_vuln_reporting=True,
            with_graph_tools=True,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
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
            worker_builder=trace_builder,
            iterations=1,
            max_attempts=1,
            max_steps=30,
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
