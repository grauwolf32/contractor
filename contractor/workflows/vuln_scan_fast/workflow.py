"""Fast vulnerability scan workflow (Workflow B).

High-recall breadth-first scan with downstream verification::

    Step 1: project_discovery  — SWE agent: deps + project structure
    Step 2: vuln_scan_fast     — vuln scan agent with over-report instructions
    Step 3: dedup              — [VERIFY] programmatic dedup by file+CWE
    Step 4: trace_confirm      — trace agent targeted per-finding
    Step 5: exploit            — [VERIFY] exploitability agent per-finding

Steps 1 is skipped when its artifact exists.  Step 5 requires
``CONTRACTOR_TARGET_URL``; if unset it is skipped with a warning.
"""
from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any

import yaml

from contractor.agents.codereview_agent.agent import build_codereview_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model, get_settings
from contractor.workflows import Workflow, WorkflowContext
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.findings import load_findings_artifact

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)

SCAN_TASK_TEMPLATE: str = "vuln_scan_fast"
VULN_REPORTS_KEY: str = "user:vulnerability-reports/vuln-scan-fast"


class VulnScanFastWorkflow(Workflow):
    """High-recall vulnerability scan → dedup → trace confirm → exploit."""

    namespace: str = "vuln-scan-fast"

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any:
        # ── Step 1: project discovery ─────────────────────────────
        await self._run_discovery(user_id=user_id, on_event=on_event)

        # ── Step 2: fast vuln scan (high recall) ──────────────────
        await self._run_fast_scan(user_id=user_id, on_event=on_event)

        # ── Step 3 [VERIFY]: dedup findings ───────────────────────
        findings = await self._load_and_dedup_findings(user_id=user_id)
        logger.info("After dedup: %d findings", len(findings))

        if not findings:
            logger.info("No findings to verify — done")
            return

        # ── Step 4: trace-confirm per finding ─────────────────────
        await self._run_trace_confirm(
            findings=findings, user_id=user_id, on_event=on_event,
        )

        # ── Step 5 [VERIFY]: exploit ──────────────────────────────
        await self._run_exploit_stage(
            findings=findings, user_id=user_id, on_event=on_event,
        )

    # ── Step 1 ───────────────────────────────────────────────────

    async def _run_discovery(
        self, *, user_id: str, on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=CFG.observations,
        )

        swe_builder = partial(
            build_swe_agent, name="swe_agent",
            _format=CFG.agent("swe_agent").output_format, fs=ctx.fs,
            model=self.llm, max_tokens=CFG.budgets.swe_max_tokens,
        )
        runner.add_variable(name="project_path", value=ctx.folder_name)

        if not await self.artifact_exists(
            user_id=user_id, filename="dependency_information/result"
        ):
            runner.add_task(
                name="dependency_information",
                # Stable explicit refs: the default positional ref
                # (`{name}:{len(queue)}`) shifts between runs when the sibling
                # task is conditionally skipped, breaking --resume checkpoints.
                ref="dependency_information",
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
                ref="project_information",
                worker_builder=swe_builder,
                **CFG.tasks.project_information.as_kwargs(),
                artifacts=["dependency_information/result"],
                namespace="project_information", model=self.llm,
            )
        else:
            await self.emit_task_skipped(on_event, "project_information")

        if runner.queue:
            await runner.run(user_id=user_id, on_event=on_event)

    # ── Step 2 ───────────────────────────────────────────────────

    async def _run_fast_scan(
        self, *, user_id: str, on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx

        agent_builder = partial(
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
            name=SCAN_TASK_TEMPLATE,
            ref="vuln-scan-fast:full",
            worker_builder=agent_builder,
            **CFG.tasks.scan.as_kwargs(),
            namespace=self.namespace,
            skills=["vuln_scan"],
            model=self.llm,
            params={"project_path": ctx.folder_name},
        )

        await runner.run(user_id=user_id, on_event=on_event)

    # ── Step 3 [VERIFY]: dedup ───────────────────────────────────

    async def _load_and_dedup_findings(
        self, *, user_id: str,
    ) -> list[dict[str, Any]]:
        findings = await load_findings_artifact(
            self.ctx.artifact_service,
            app_name=self.ctx.app_name,
            user_id=user_id,
            filename=VULN_REPORTS_KEY,
        )

        before = len(findings)
        findings = self._dedup(findings)
        if before != len(findings):
            logger.info(
                "Dedup: %d → %d findings (removed %d duplicates)",
                before, len(findings), before - len(findings),
            )

        return findings

    @staticmethod
    def _dedup(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge findings with same (file, CWE), keeping higher confidence."""
        conf_rank = {"high": 3, "medium": 2, "low": 1}
        buckets: dict[tuple[str, str], dict[str, Any]] = {}

        for f in findings:
            # Coerce defensively: explicit-null YAML fields (`details:`) come
            # back as None, and `str()` guards against non-string scalars.
            place = str(f.get("place") or "")
            details = str(f.get("details") or "")
            cwe = re.search(r"CWE-(\d+)", details)
            key = (
                place.strip("/").lower(),
                cwe.group(1) if cwe else "",
            )
            existing = buckets.get(key)
            if existing is None:
                buckets[key] = f
            else:
                cur_rank = conf_rank.get(f.get("confidence", ""), 0)
                old_rank = conf_rank.get(existing.get("confidence", ""), 0)
                if cur_rank > old_rank:
                    buckets[key] = f

        return list(buckets.values())

    # ── Step 4: trace confirm ────────────────────────────────────

    async def _run_trace_confirm(
        self,
        *,
        findings: list[dict[str, Any]],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        from contractor.agents.trace_agent.agent import build_trace_agent
        from contractor.runners.agent_runner import AgentRunner
        from contractor.tools.fs import MemoryOverlayFileSystem

        ctx = self.ctx
        overlay = MemoryOverlayFileSystem(fs=ctx.fs, skip_instance_cache=True)

        runner = AgentRunner(
            name=ctx.app_name,
            artifact_service=ctx.artifact_service,
        )

        for finding in findings:
            # `or ""` guards explicit-null YAML fields (None) before slicing.
            fname = str(finding.get("name") or "")
            place = str(finding.get("place") or "")
            title = str(finding.get("title") or "")
            details = str(finding.get("details") or "")

            ns = f"trace-confirm:{fname}"

            from contractor.runners.skills import inject_skills
            await inject_skills(
                ["trace"],
                namespace=ns,
                artifact_service=ctx.artifact_service,
                app_name=ctx.app_name,
                user_id=user_id,
            )

            agent = build_trace_agent(
                name="trace_agent",
                _format=CFG.agent("trace_agent").output_format,
                fs=overlay,
                namespace=ns,
                model=self.llm,
                max_tokens=CFG.budgets.scan_max_tokens,
                enable_vuln_reporting=True,
                with_graph_tools=CFG.agent("trace_agent").with_graph_tools,
            )

            message = (
                f"A vulnerability scan reported a potential finding:\n"
                f"  Title: {title}\n"
                f"  File: {place}\n"
                f"  Details: {details[:500]}\n\n"
                f"Trace the code path in `{place}` to CONFIRM or DENY this finding. "
                f"If confirmed, use `report_vulnerability` with full evidence. "
                f"If the finding is a false positive, explain why and do NOT report it."
            )

            try:
                await runner.run(
                    agent=agent,
                    message=message,
                    user_id=user_id,
                    initial_state={},
                    on_event=on_event,
                    event_name=f"trace_confirm:{fname}",
                )
            except Exception as exc:
                logger.warning("trace_confirm failed for %s: %s", fname, exc)

    # ── Step 5 [VERIFY]: exploit ─────────────────────────────────

    async def _run_exploit_stage(
        self,
        *,
        findings: list[dict[str, Any]],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        target_url = get_settings().target_url
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

        findings_yaml = yaml.safe_dump(
            {f["name"]: f for f in findings if f.get("name")},
            sort_keys=False,
            allow_unicode=True,
        )

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
