"""Full vulnerability assessment workflow (Workflow A).

Five-step chain with verification at steps 3 and 5::

    Step 1: project_discovery  — SWE agent: deps + project structure
    Step 2: oas_build          — OAS builder: extract API surface
    Step 3: oas_validate       — [VERIFY] linter: verify spec, fix errors
    Step 4: trace_vuln         — trace agent per-operation (pathpar, vuln reporting)
    Step 5: exploit            — [VERIFY] exploitability agent per-finding

Steps 1-2 are skipped when their artifacts already exist from a prior
run.  Step 5 requires ``CONTRACTOR_TARGET_URL``; if unset it is skipped
with a warning.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any

import yaml

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext, persist_seed_artifact
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.trace_annotation import extract_openapi_paths
from contractor.workflows.trace_graph_pathpar import TraceGraphPathParWorkflow

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)

OAS_ARTIFACT = "user:oas-openapi-building"
TRACE_OAS_ARTIFACT = "oas-openapi-building"


class VulnAssessWorkflow(Workflow):
    """Full vulnerability assessment: discovery → OAS → trace → exploit."""

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any:
        # ── Steps 1-3: discovery + OAS build + validate ───────────
        await self._run_oas_stage(user_id=user_id, on_event=on_event)

        # ── Bridge: copy OAS artifact for trace stage ─────────────
        await self._bridge_oas_artifact(user_id=user_id)

        # ── Step 4: trace + vuln reporting (pathpar) ──────────────
        await self._run_trace_stage(user_id=user_id, on_event=on_event)

        # ── Step 5: exploitability verification ───────────────────
        await self._run_exploit_stage(user_id=user_id, on_event=on_event)

    # ── Stage 1-3: OAS workflow ──────────────────────────────────

    async def _run_oas_stage(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx
        fs = ctx.fs

        await persist_seed_artifact(ctx, filename=TRACE_OAS_ARTIFACT)

        if await self.artifact_exists(user_id=user_id, filename=OAS_ARTIFACT):
            await self.emit_task_skipped(
                on_event, "oas_stage", reason="OAS artifact already exists"
            )
            return

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=CFG.observations,
        )

        swe_builder = partial(
            build_swe_agent, name="swe_agent",
            _format=CFG.agent("swe_agent").output_format, fs=fs,
            model=self.llm, max_tokens=CFG.budgets.swe_max_tokens,
        )
        oas_builder = partial(
            build_oas_builder_agent, name="oas_builder",
            _format=CFG.agent("oas_builder").output_format, fs=fs,
            model=self.llm, max_tokens=CFG.budgets.builder_max_tokens,
        )
        oas_linter = partial(
            build_oas_linter_agent, name="oas_validator",
            _format=CFG.agent("oas_validator").output_format, fs=fs,
            model=self.llm, max_tokens=CFG.budgets.validator_max_tokens,
        )

        runner.add_variable(name="project_path", value=ctx.folder_name)

        if not await self.artifact_exists(
            user_id=user_id, filename="dependency_information/result"
        ):
            runner.add_task(
                name="dependency_information",
                worker_builder=swe_builder,
                **CFG.tasks.dependency_information.as_kwargs(),
                namespace="dependency_information", model=self.llm,
            )
        else:
            await self.emit_task_skipped(on_event, "dependency_information")

        if not await self.artifact_exists(
            user_id=user_id, filename="project_information/result"
        ):
            runner.add_task(
                name="project_information",
                worker_builder=swe_builder,
                **CFG.tasks.project_information.as_kwargs(),
                artifacts=["dependency_information/result"],
                namespace="project_information", model=self.llm,
            )
        else:
            await self.emit_task_skipped(on_event, "project_information")

        runner.add_task(
            name="oas_update",
            worker_builder=oas_builder,
            **CFG.tasks.oas_update.as_kwargs(),
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="openapi-building", model=self.llm,
        )

        # Step 3 [VERIFY]
        runner.add_task(
            name="oas_validate",
            worker_builder=oas_linter,
            **CFG.tasks.oas_validate.as_kwargs(),
            artifacts=[
                "dependency_information/result",
                "project_information/result",
                "oas_update/result",
            ],
            namespace="openapi-building", model=self.llm,
        )

        await runner.run(user_id=user_id, on_event=on_event)

    # ── Bridge ───────────────────────────────────────────────────

    async def _bridge_oas_artifact(self, *, user_id: str) -> None:
        """Copy ``user:oas-openapi-building`` → ``oas-openapi-building``.

        The OAS builder writes to the user-scoped key; the trace stage
        reads from the non-prefixed key.  This bridges the two.
        """
        ctx = self.ctx
        if await self.artifact_exists(user_id=user_id, filename=TRACE_OAS_ARTIFACT):
            return

        part = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=OAS_ARTIFACT,
        )
        if part is None:
            logger.warning("No OAS artifact found after build stage")
            return

        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename=TRACE_OAS_ARTIFACT,
            artifact=part,
        )

    # ── Stage 4: trace + vuln ────────────────────────────────────

    async def _run_trace_stage(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx

        has_oas = await self.artifact_exists(
            user_id=user_id, filename=TRACE_OAS_ARTIFACT,
        )
        if not has_oas:
            logger.error("Cannot run trace stage — no OAS artifact")
            return

        trace_workflow = TraceGraphPathParWorkflow(ctx)
        await trace_workflow._run_impl(user_id=user_id, on_event=on_event)

    # ── Stage 5: exploit [VERIFY] ────────────────────────────────

    async def _run_exploit_stage(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        target_url = os.environ.get("CONTRACTOR_TARGET_URL")
        if not target_url:
            logger.warning(
                "CONTRACTOR_TARGET_URL not set — skipping exploit stage"
            )
            await self.emit_task_skipped(
                on_event, "exploit",
                reason="CONTRACTOR_TARGET_URL not set",
            )
            return

        from contractor.workflows.exploitability import ExploitabilityWorkflow

        findings_yaml = await self._collect_vuln_reports(user_id=user_id)
        if not findings_yaml:
            logger.info("No vulnerability reports to verify — skipping exploit")
            await self.emit_task_skipped(
                on_event, "exploit", reason="no_findings",
            )
            return

        exploit_ctx = WorkflowContext(
            project_path=self.ctx.project_path,
            folder_name=self.ctx.folder_name,
            model=self.ctx.model,
            timeout=self.ctx.timeout,
            app_name=self.ctx.app_name,
            user_id=self.ctx.user_id,
            artifact_service=self.ctx.artifact_service,
            fs=self.ctx.fs,
            artifact=findings_yaml,
            checkpoint_path=self.ctx.checkpoint_path,
        )

        exploit_workflow = ExploitabilityWorkflow(exploit_ctx)
        await exploit_workflow._run_impl(user_id=user_id, on_event=on_event)

    async def _collect_vuln_reports(self, *, user_id: str) -> str:
        """Merge all ``user:vulnerability-reports/*`` artifacts into one YAML."""
        ctx = self.ctx
        merged: dict[str, Any] = {}

        part = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=user_id,
            filename="vulnerability-reports-seed",
        )
        if part and part.text:
            try:
                raw = yaml.safe_load(part.text) or {}
                if isinstance(raw, dict):
                    merged.update(raw)
            except yaml.YAMLError:
                pass

        for ns_suffix in ["openapi"]:
            oas_part = await ctx.artifact_service.load_artifact(
                app_name=ctx.app_name,
                user_id=user_id,
                filename=f"oas-{ns_suffix}-building",
            )
            if not oas_part:
                continue
            try:
                openapi = yaml.safe_load(oas_part.text or "") or {}
            except yaml.YAMLError:
                continue
            paths = extract_openapi_paths(openapi=openapi)
            for api_path in paths:
                ns = f"trace-annotation:{ns_suffix}:{api_path.path_key}"
                part = await ctx.artifact_service.load_artifact(
                    app_name=ctx.app_name,
                    user_id=user_id,
                    filename=f"user:vulnerability-reports/{ns}",
                )
                if part and part.text:
                    try:
                        reports = yaml.safe_load(part.text) or {}
                        if isinstance(reports, dict):
                            merged.update(reports)
                    except yaml.YAMLError:
                        continue

        if not merged:
            return ""
        return yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
