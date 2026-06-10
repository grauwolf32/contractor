"""Static verifier of upstream vulnerability findings (OpenAnt Stage-2 style).

For each path in the source OpenAPI schema, loads the per-path
:class:`VulnerabilityReport` artifacts written by a prior trace run — any of
``trace`` / ``trace-direct`` (``trace-annotation:`` prefix), ``trace-graph``
(``trace-graph:``), or ``trace-graph-pathpar`` (``trace-graph-pathpar:``) —
and queues one task per finding for ``trace_verifier_agent``.

The verifier is code-evidence-only — no HTTP probes — and persists verdicts
via ``verification_tools`` under the same namespace as the upstream findings,
so the two artifacts pair up:

    user:vulnerability-reports/{prefix}:openapi:{path_key}
    user:vulnerability-verifications/{prefix}:openapi:{path_key}

Paths with no findings are skipped (DEBUG log); if *no* path has findings
under *any* prefix the workflow logs a WARNING and completes as a no-op.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import yaml

from contractor.agents.trace_verifier_agent.agent import build_trace_verifier_agent
from contractor.runners.artifacts import artifact_key_slug
from contractor.runners.task_runner import TaskRunner, TaskRunnerEventHandler
from contractor.utils.settings import build_model
from contractor.workflows import Workflow, WorkflowContext, persist_seed_artifact
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.namespaces import TRACE_NAMESPACE_PREFIXES
from contractor.workflows.trace_annotation import OpenApiPath, extract_openapi_paths

CFG = WorkflowConfig.load(__file__)

logger = logging.getLogger(__name__)


class TraceVerifyWorkflow(Workflow):
    """OpenAnt Stage-2-style verifier for trace findings."""

    namespace: str = "openapi"

    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)
        self.paths: list[OpenApiPath] = []

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

        total_findings = 0
        for api_path in self.paths:
            total_findings += await self._verify_path_findings(
                api_path=api_path,
                user_id=user_id,
                on_event=on_event,
            )

        if not total_findings:
            logger.warning(
                "trace-verify found no vulnerability reports for any of the "
                "%d OpenAPI paths under any known trace namespace prefix "
                "(probed: %s) — nothing to verify. Run a trace workflow "
                "(trace / trace-direct / trace-graph / trace-graph-pathpar) "
                "against this project first.",
                len(self.paths),
                ", ".join(TRACE_NAMESPACE_PREFIXES),
            )

    def _candidate_namespaces(self, api_path: OpenApiPath) -> list[str]:
        """Every namespace a trace producer may have written findings under
        for ``api_path``, one per known prefix, in probe order."""
        return [
            f"{prefix}:{self.namespace}:{api_path.path_key}"
            for prefix in TRACE_NAMESPACE_PREFIXES
        ]

    async def _discover_findings(
        self,
        *,
        user_id: str,
        api_path: OpenApiPath,
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Probe every candidate namespace for ``api_path`` and return
        ``(source_namespace, findings)`` pairs for the non-empty ones."""
        discovered: list[tuple[str, list[dict[str, Any]]]] = []
        for source_namespace in self._candidate_namespaces(api_path):
            findings = await self._load_findings(
                user_id=user_id, source_namespace=source_namespace
            )
            if findings:
                discovered.append((source_namespace, findings))
        return discovered

    async def _verify_path_findings(
        self,
        *,
        api_path: OpenApiPath,
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> int:
        """Verify every finding recorded for ``api_path``; returns how many
        findings were queued (0 when the path has none under any prefix)."""
        discovered = await self._discover_findings(
            user_id=user_id, api_path=api_path
        )
        if not discovered:
            logger.debug(
                "no vulnerability reports for path %r under any trace "
                "namespace prefix (probed: %s) — skipping verify",
                api_path.path,
                ", ".join(self._candidate_namespaces(api_path)),
            )
            return 0

        total = 0
        for source_namespace, findings in discovered:
            total += len(findings)
            await self._verify_namespace_findings(
                api_path=api_path,
                source_namespace=source_namespace,
                findings=findings,
                user_id=user_id,
                on_event=on_event,
            )
        return total

    async def _verify_namespace_findings(
        self,
        *,
        api_path: OpenApiPath,
        source_namespace: str,
        findings: list[dict[str, Any]],
        user_id: str,
        on_event: TaskRunnerEventHandler | None,
    ) -> None:
        ctx = self.ctx
        verifier_builder = partial(
            build_trace_verifier_agent,
            name="trace_verifier",
            _format=CFG.agent("trace_verifier").output_format,
            fs=ctx.fs,
            source_namespace=source_namespace,
            model=self.llm,
            max_tokens=CFG.budgets.max_tokens,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
            observations=CFG.observations,
        )
        runner.add_variable(name="project_path", value=ctx.folder_name)

        for finding in findings:
            finding_name = finding.get("name", "")
            if not finding_name:
                continue
            runner.add_task(
                name="trace_verify",
                ref=(
                    f"trace_verify:{self.namespace}:"
                    f"{api_path.path_key}:{finding_name}"
                ),
                # Unique, stable per-finding publish key — every finding is a
                # separate `trace_verify` task and the shared template key
                # would make siblings overwrite each other's artifacts.
                artifact_key=(
                    f"trace_verify/{artifact_key_slug(source_namespace)}/"
                    f"{artifact_key_slug(finding_name)}"
                ),
                worker_builder=verifier_builder,
                **CFG.tasks.verify.as_kwargs(),
                namespace=source_namespace,
                model=self.llm,
                params={
                    "finding_name": finding_name,
                    "finding_title": finding.get("title", ""),
                    "finding_place_type": finding.get("place_type", "file"),
                    "finding_place": finding.get("place", ""),
                    "finding_severity": finding.get("severity", "medium"),
                    "finding_confidence": finding.get("confidence", "medium"),
                    "finding_summary": finding.get("summary", ""),
                    "source_namespace": source_namespace,
                },
            )

        await runner.run(user_id=user_id, on_event=on_event)

    async def _load_findings(
        self,
        *,
        user_id: str,
        source_namespace: str,
    ) -> list[dict[str, Any]]:
        """Read the per-namespace VulnerabilityReport artifact, if any.

        Mirrors :class:`VulnerabilityReportTools.load`: the YAML is a flat
        mapping of ``name → fields``. Returns each entry as a plain dict with
        ``name`` filled in from the key when absent. Empty / missing /
        malformed artifacts return an empty list (the path is then skipped).
        """
        artifact_key = f"user:vulnerability-reports/{source_namespace}"
        part = await self.ctx.artifact_service.load_artifact(
            app_name=self.ctx.app_name,
            user_id=user_id,
            filename=artifact_key,
        )
        if part is None or not part.text:
            return []
        try:
            raw = yaml.safe_load(part.text) or {}
        except yaml.YAMLError as exc:
            logger.warning(
                "could not parse %s as YAML: %s — skipping path",
                artifact_key,
                exc,
            )
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
        return findings
