"""Recall-oriented two-pass vulnerability workflow.

Pass 1 (BFS sweep): one cheap nomination task *per vulnerability class*
runs in parallel, each sweeping the whole project for candidate sinks of
its single class and reporting them at low confidence. Splitting the
sweep by class — rather than one agent scanning for everything — keeps
each agent's attention narrow (the recall problem on large codebases is
an attention problem) and adds an explicit ABSENCE class so missing
controls, which a taint-only scan can never surface, get nominated.

Pass 2 (DFS trace): the nominations are merged, deduped, capped, and
each survivor is deep-traced by ``trace_agent`` to confirm or discard it
— reusing ``VulnScanTraceWorkflow._trace_finding`` verbatim.

This is the BFS/DFS duality made operational: a wide, blind sweep that
nominates, then a narrow, evidence-driven trace that judges.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

from contractor.agents.codereview_agent.agent import build_codereview_agent
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.findings import load_findings_artifact
from contractor.workflows.vuln_scan_trace.workflow import VulnScanTraceWorkflow

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SinkClass:
    """One vulnerability class swept by its own nomination agent."""

    key: str
    guidance: str


# The sweep surface, one agent per class. The last entry is the
# absence-of-control class — the structural answer to missing-auth /
# missing-ownership misses that a taint-following scan cannot find.
SINK_CLASSES: tuple[SinkClass, ...] = (
    SinkClass(
        "injection",
        "SQL/NoSQL/ORM-raw, OS command, template (SSTI), LDAP, and "
        "expression-language sinks: execute/cursor/query, subprocess/exec/"
        "system/popen, render_template_string, eval/compile.",
    ),
    SinkClass(
        "deserialization",
        "Unsafe deserialization and object construction from input: "
        "pickle, yaml.load (unsafe), marshal, jackson polymorphic typing, "
        "PHP unserialize, .NET BinaryFormatter.",
    ),
    SinkClass(
        "ssrf-fileio",
        "Server-side request forgery and unsafe file I/O: outbound "
        "requests built from input (requests/urllib/http clients), "
        "path joins / open / send_file with request-derived paths, "
        "redirects to input-derived URLs.",
    ),
    SinkClass(
        "secrets-crypto",
        "Hardcoded secrets/keys, debug flags, weak crypto (md5/sha1 for "
        "security, ECB, static IV/salt), and insecure randomness used "
        "for tokens/passwords.",
    ),
    SinkClass(
        "missing-access-control",
        "ABSENCE class — nominate per handler. Route handlers and "
        "sensitive operations (state change, data export, admin, "
        "object access by id) whose visible code lacks an "
        "authentication, authorization, or ownership check. Missing "
        "control is the signal; no taint flow is required.",
    ),
)


class VulnSweepWorkflow(VulnScanTraceWorkflow):
    """Per-class BFS nomination sweep → DFS trace of survivors.

    Reuses ``VulnScanTraceWorkflow._trace_finding`` / ``_load_findings``
    for the DFS pass; overrides ``_run_impl`` to fan the sweep out by
    vulnerability class. ``CFG`` is overridden so the inherited trace
    phase reads this workflow's sibling ``config.yaml``.
    """

    namespace: str = "vuln-sweep"
    CFG: ClassVar[WorkflowConfig] = CFG

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> Any:
        # ── Pass 1: per-class nomination sweep (parallel) ───────────────
        sem = asyncio.Semaphore(self.CFG.budgets.sweep_concurrency)

        async def _sweep(sink_class: SinkClass) -> None:
            async with sem:
                await self._sweep_class(
                    sink_class=sink_class, user_id=user_id, on_event=on_event
                )

        async with asyncio.TaskGroup() as tg:
            for sink_class in SINK_CLASSES:
                tg.create_task(_sweep(sink_class))

        # ── Merge, dedup, cap nominations ───────────────────────────────
        nominations = await self._collect_nominations(user_id=user_id)
        logger.info(
            "vuln-sweep: %d nominations after dedup across %d classes",
            len(nominations),
            len(SINK_CLASSES),
        )
        if not nominations:
            logger.warning("vuln-sweep: no nominations — skipping trace phase")
            return

        cap = self.CFG.budgets.max_trace_nominations
        if len(nominations) > cap:
            logger.info(
                "vuln-sweep: capping %d nominations to %d for the trace phase "
                "(highest severity/confidence first)",
                len(nominations),
                cap,
            )
            nominations = nominations[:cap]

        # ── Pass 2: DFS trace per surviving nomination ──────────────────
        for finding in nominations:
            await self._trace_finding(
                finding=finding, user_id=user_id, on_event=on_event
            )

    def _class_namespace(self, sink_class: SinkClass) -> str:
        return f"{self.namespace}:sweep:{sink_class.key}"

    async def _sweep_class(
        self,
        *,
        sink_class: SinkClass,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx
        class_namespace = self._class_namespace(sink_class)

        sweep_builder = partial(
            build_codereview_agent,
            name="codereview_agent",
            _format=self.CFG.agent("codereview_agent").output_format,
            fs=ctx.fs,
            model=self.llm,
            max_tokens=self.CFG.budgets.scan_max_tokens,
            with_graph_tools=self.CFG.agent("codereview_agent").with_graph_tools,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=self.CFG.observations,
        )
        runner.add_variable(name="project_path", value=ctx.folder_name)

        runner.add_task(
            name="sink_nomination",
            ref=f"{self.namespace}:sweep:{sink_class.key}",
            worker_builder=sweep_builder,
            **self.CFG.tasks.sweep.as_kwargs(),
            namespace=class_namespace,
            skills=["vuln_scan"],
            model=self.llm,
            params={
                "project_path": ctx.folder_name,
                "sink_class": sink_class.key,
                "class_guidance": sink_class.guidance,
            },
        )

        try:
            await runner.run(user_id=user_id, on_event=on_event)
        except Exception as exc:
            logger.warning("sweep for class %s failed: %s", sink_class.key, exc)

    async def _collect_nominations(
        self, *, user_id: str
    ) -> list[dict[str, Any]]:
        """Merge every class's nominations, dedup by (place, name), and
        sort highest severity/confidence first so the cap keeps the most
        promising candidates."""
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for sink_class in SINK_CLASSES:
            findings = await load_findings_artifact(
                self.ctx.artifact_service,
                app_name=self.ctx.app_name,
                user_id=user_id,
                filename=(
                    "user:vulnerability-reports/"
                    f"{self._class_namespace(sink_class)}"
                ),
            )
            for finding in findings:
                key = (
                    str(finding.get("place", "")),
                    str(finding.get("name", "")),
                )
                merged.setdefault(key, finding)

        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        conf_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            merged.values(),
            key=lambda f: (
                sev_order.get(f.get("severity", "low"), 4),
                conf_order.get(f.get("confidence", "low"), 3),
            ),
        )
