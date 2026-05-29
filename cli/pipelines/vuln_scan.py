"""Rapid vulnerability discovery pipeline.

Scans a project codebase for security vulnerabilities using
pattern-based discovery. Runs a single vuln_scan_agent pass
with grep-driven breadth-first scanning.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

from cli.pipelines import Pipeline, PipelineContext
from cli.pipelines.config import VULN_SCAN as CFG
from contractor.agents.vuln_scan_agent.agent import build_vuln_scan_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model

logger = logging.getLogger(__name__)


class VulnScanPipeline(Pipeline):
    """Breadth-first vulnerability scanning against source code."""

    namespace: str = "vuln-scan"

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

        agent_builder = partial(
            build_vuln_scan_agent,
            name="vuln_scan_agent",
            fs=ctx.fs,
            model=self.llm,
            max_tokens=CFG.scan_max_tokens,
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
            ref="vuln-scan:full",
            worker_builder=agent_builder,
            **CFG.scan.as_kwargs(),
            namespace=self.namespace,
            skills=["vuln_scan"],
            model=self.llm,
            params={"project_path": ctx.folder_name},
        )

        await runner.run(user_id=user_id, on_event=on_event)
