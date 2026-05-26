"""Static verifier of upstream vulnerability findings (OpenAnt Stage-2 style).

For each path in the source OpenAPI schema, loads the per-path
:class:`VulnerabilityReport` artifact written by a prior ``trace-direct`` (or
``trace``) run and queues one task per finding for ``trace_verifier_agent``.

The verifier is code-evidence-only — no HTTP probes — and persists verdicts
via ``verification_tools`` under the same namespace as the upstream findings,
so the two artifacts pair up:

    user:vulnerability-reports/trace-annotation:openapi:{path_key}
    user:vulnerability-verifications/trace-annotation:openapi:{path_key}

Paths with no findings are silently skipped.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

import yaml
from google.adk.models import LiteLlm

from cli.pipelines import Pipeline, PipelineContext, persist_seed_artifact
from cli.pipelines.trace_annotation import (OpenApiPath,
                                            extract_openapi_paths)
from contractor.agents.trace_verifier_agent.agent import \
    build_trace_verifier_agent
from contractor.runners.task_runner import (TaskRunner,
                                            TaskRunnerEventHandler)

logger = logging.getLogger(__name__)

VERIFY_MAX_TOKENS: int = 80_000


class TraceVerifyPipeline(Pipeline):
    """OpenAnt Stage-2-style verifier for trace findings."""

    namespace: str = "openapi"

    def __init__(self, ctx: PipelineContext) -> None:
        super().__init__(ctx)
        self.llm = LiteLlm(model=ctx.model)
        self.paths: list[OpenApiPath] = []

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

        openapi = yaml.safe_load(raw.text)
        self.paths = extract_openapi_paths(openapi=openapi)

        for api_path in self.paths:
            await self._verify_path_findings(
                api_path=api_path,
                user_id=user_id,
                on_event=on_event,
            )

    async def _verify_path_findings(
        self,
        *,
        api_path: OpenApiPath,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> None:
        source_namespace = f"trace-annotation:{self.namespace}:{api_path.path_key}"
        findings = await self._load_findings(
            user_id=user_id, source_namespace=source_namespace
        )
        if not findings:
            logger.debug(
                "no findings under %r — skipping verify for path %r",
                source_namespace,
                api_path.path,
            )
            return

        ctx = self.ctx
        verifier_builder = partial(
            build_trace_verifier_agent,
            name="trace_verifier",
            fs=ctx.fs,
            source_namespace=source_namespace,
            model=self.llm,
            max_tokens=VERIFY_MAX_TOKENS,
        )

        runner = TaskRunner(
            name="contractor",
            artifact_service=ctx.artifact_service,
            checkpoint_path=ctx.checkpoint_path,
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
                worker_builder=verifier_builder,
                iterations=1,
                max_attempts=2,
                max_steps=20,
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
