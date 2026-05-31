"""Workflow-assembly smoke tests.

These tests exercise each workflow's ``_run_impl`` with the heavy I/O
side-effects stubbed out, then inspect the resulting ``TaskRunner.queue``.
They are the first line of defense against artifact-wiring regressions:
a typo in an upstream ``add_task(name=...)`` or a downstream
``artifacts=[...]`` reference is silently fatal at runtime, but caught
here by ``test_<workflow>_artifact_references_resolve``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from google.adk.artifacts import BaseArtifactService

from contractor.runners.agent_runner import AgentRunner
from contractor.runners.task_runner import TaskRunner
from contractor.workflows import Workflow, WorkflowContext, get_workflows
from contractor.workflows.likec4_building import LikeC4BuildingWorkflow
from contractor.workflows.oas_building import OasBuildingWorkflow
from contractor.workflows.oas_enrichment import OasEnrichmentWorkflow
from contractor.workflows.router import RouterWorkflow
from contractor.workflows.trace_annotation import TraceAnnotationWorkflow
from contractor.workflows.trace_annotation_direct import \
    TraceAnnotationDirectWorkflow
from contractor.workflows.trace_verify import TraceVerifyWorkflow

# ─── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_known_keys_present(self):
        registry = get_workflows()
        assert set(registry.keys()) == {
            "build",
            "enrich",
            "exploit",
            "likec4",
            "trace",
            "trace-direct",
            "trace-graph",
            "trace-graph-pathpar",
            "trace-verify",
            "vuln-assess",
            "vuln-scan",
            "vuln-scan-fast",
            "vuln-scan-trace",
            "router",
        }

    def test_all_classes_extend_workflow_base(self):
        for cls in get_workflows().values():
            assert issubclass(cls, Workflow), f"{cls.__name__} must extend Workflow"

    def test_registry_matches_explicit_imports(self):
        # Catch the case where get_workflows() drifts from the modules that
        # actually define the classes — e.g. someone deletes a workflow file
        # but forgets to remove it from the registry.
        registry = get_workflows()
        assert registry["build"] is OasBuildingWorkflow
        assert registry["enrich"] is OasEnrichmentWorkflow
        assert registry["likec4"] is LikeC4BuildingWorkflow
        assert registry["trace"] is TraceAnnotationWorkflow
        assert registry["trace-direct"] is TraceAnnotationDirectWorkflow
        assert registry["trace-verify"] is TraceVerifyWorkflow
        assert registry["router"] is RouterWorkflow


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_context(*, prompt: str | None = None) -> WorkflowContext:
    """A WorkflowContext with all I/O surfaces mocked.

    `load_artifact` returns None by default — workflows should treat that as
    "no prior run". Tests that need a non-empty artifact patch this further.
    """
    artifact_service = MagicMock(spec=BaseArtifactService)
    artifact_service.load_artifact = AsyncMock(return_value=None)
    artifact_service.save_artifact = AsyncMock()

    return WorkflowContext(
        project_path=Path("/tmp/proj"),
        folder_name="src",
        model="lm-studio-test",
        app_name="contractor-test",
        user_id="u",
        artifact_service=artifact_service,
        fs=MagicMock(),
        prompt=prompt,
    )


async def _capture_queue(workflow: Workflow, *, monkeypatch) -> list:
    """Invoke a workflow's ``_run_impl`` while suppressing the actual run.

    Returns the list of ``TaskInvocation`` objects added to the queue.
    """
    captured: dict = {}

    original_init = TaskRunner.__init__

    def capture_init(self, **kwargs):
        original_init(self, **kwargs)
        captured.setdefault("runners", []).append(self)

    async def fake_run(self, **kwargs):
        return []

    monkeypatch.setattr(TaskRunner, "__init__", capture_init)
    monkeypatch.setattr(TaskRunner, "run", fake_run)

    await workflow._run_impl(user_id="u", on_event=None)

    queues: list = []
    for r in captured.get("runners", []):
        queues.extend(r.queue)
    return queues


def _refs(queue) -> set[str]:
    """Set of ``ref``s actually queued."""
    return {item.ref for item in queue}


def _template_keys(queue) -> set[str]:
    return {item.template_key for item in queue}


def _all_artifact_refs(queue) -> set[str]:
    refs: set[str] = set()
    for item in queue:
        refs.update(item.artifacts)
    return refs


# Artifacts that workflows persist outside the runner before any task runs.
# Trace workflows seed ``oas-openapi-building`` via ``persist_seed_artifact``
# and depend on it being present — they don't go through the runner queue.
_KNOWN_SEED_ARTIFACTS: set[str] = set()


def _producing_task_key(artifact_ref: str) -> str:
    """For ``<template_key>/result``, the upstream task is ``<template_key>``."""
    return artifact_ref.rsplit("/", 1)[0]


# ─── OasBuildingWorkflow ──────────────────────────────────────────────────────


class TestOasBuildingWorkflow:
    @pytest.mark.asyncio
    async def test_assembles_four_tasks(self, monkeypatch):
        workflow = OasBuildingWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)

        # Two code-analysis tasks + oas_update + oas_validate, all queued
        # because load_artifact returns None (no prior run).
        assert _template_keys(queue) == {
            "dependency_information",
            "project_information",
            "oas_update",
            "oas_validate",
        }

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        workflow = OasBuildingWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | _KNOWN_SEED_ARTIFACTS
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced, (
                f"artifact {artifact_ref!r} is referenced but no upstream "
                f"task or known seed produces it"
            )

    @pytest.mark.asyncio
    async def test_skip_when_dependency_artifact_exists(self, monkeypatch):
        ctx = _make_context()

        # Pretend dependency_information already has a result on disk.
        async def fake_load(*, app_name, user_id, filename, **_):
            if filename == "dependency_information/result":
                return MagicMock(text="ok", inline_data=None)
            return None

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)

        workflow = OasBuildingWorkflow(ctx)
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)
        assert "dependency_information" not in _template_keys(queue)
        # Downstream tasks still queued — they will read the existing artifact.
        assert "oas_update" in _template_keys(queue)


# ─── OasEnrichmentWorkflow ────────────────────────────────────────────────────


class TestOasEnrichmentWorkflow:
    @pytest.mark.asyncio
    async def test_assembles_enrich_and_validate(self, monkeypatch):
        workflow = OasEnrichmentWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)
        assert _template_keys(queue) == {"oas_enrich", "oas_validate"}

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        # Enrich pulls upstream `dependency_information/result` and
        # `project_information/result`, which this workflow does NOT queue —
        # they must come from a prior `build` run. Treat them as accepted
        # external inputs.
        external_inputs = {"dependency_information", "project_information"}
        workflow = OasEnrichmentWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | external_inputs
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced


# ─── LikeC4BuildingWorkflow ───────────────────────────────────────────────────


class TestLikeC4BuildingWorkflow:
    @pytest.mark.asyncio
    async def test_assembles_four_tasks(self, monkeypatch):
        # Skip the overlay seed/persist (they hit ctx.fs which is a MagicMock).
        monkeypatch.setattr(
            LikeC4BuildingWorkflow,
            "_seed_overlay_from_artifact",
            AsyncMock(),
        )
        monkeypatch.setattr(
            LikeC4BuildingWorkflow,
            "_persist_overlay_to_artifact",
            AsyncMock(),
        )

        workflow = LikeC4BuildingWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)
        assert _template_keys(queue) == {
            "dependency_information",
            "project_information",
            "likec4_build",
            "likec4_validate",
        }

    @pytest.mark.asyncio
    async def test_artifact_references_resolve(self, monkeypatch):
        monkeypatch.setattr(
            LikeC4BuildingWorkflow,
            "_seed_overlay_from_artifact",
            AsyncMock(),
        )
        monkeypatch.setattr(
            LikeC4BuildingWorkflow,
            "_persist_overlay_to_artifact",
            AsyncMock(),
        )

        workflow = LikeC4BuildingWorkflow(_make_context())
        queue = await _capture_queue(workflow, monkeypatch=monkeypatch)

        produced = _template_keys(queue) | _KNOWN_SEED_ARTIFACTS
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced


# ─── TraceAnnotationWorkflow ──────────────────────────────────────────────────


class TestTraceAnnotationWorkflow:
    """Per-path trace workflow. Test the per-path task assembly directly
    using ``_run_path_analysis`` since the outer loop iterates over an
    OpenAPI artifact we'd otherwise need to fabricate."""

    @pytest.mark.asyncio
    async def test_per_path_task_assembles(self, monkeypatch):
        from contractor.workflows.trace_annotation import (OpenApiOperation,
                                                           OpenApiPath)

        api_path = OpenApiPath(
            path="/items/{id}",
            operations=[
                OpenApiOperation(
                    operation_id="getItem",
                    method="get",
                    path="/items/{id}",
                    schema={},
                )
            ],
        )

        workflow = TraceAnnotationWorkflow(_make_context())
        queue: list = []
        original_init = TaskRunner.__init__

        def capture_init(self, **kwargs):
            original_init(self, **kwargs)
            queue.append(self)

        monkeypatch.setattr(TaskRunner, "__init__", capture_init)
        monkeypatch.setattr(TaskRunner, "run", AsyncMock())

        await workflow._run_path_analysis(api_path, user_id="u")

        assert len(queue) == 1
        runner = queue[0]
        assert len(runner.queue) == 1
        item = runner.queue[0]
        assert item.template_key == "trace_annotation"
        assert "items_id" in item.ref
        assert item.skills == ["trace"]
        # No artifacts — the trace template reads from the overlay-FS instead.
        assert item.artifacts == []


# ─── TraceAnnotationDirectWorkflow ────────────────────────────────────────────


class TestTraceAnnotationDirectWorkflow:
    def test_template_loads_at_class_init(self):
        # The "trace_annotation" template MUST be loadable for this workflow
        # to import successfully. Construction triggers TaskTemplate.load —
        # a missing manifest or body file fails the test.
        workflow = TraceAnnotationDirectWorkflow(_make_context())
        assert workflow._template.key == "trace_annotation"
        assert workflow._template.version  # any non-empty version is fine


# ─── TraceVerifyWorkflow ──────────────────────────────────────────────────────


class TestTraceVerifyWorkflow:
    """Per-path verifier workflow. Tests use ``_verify_path_findings``
    directly with a hand-crafted ``OpenApiPath`` since the outer loop only
    reads the OpenAPI artifact to discover paths."""

    @staticmethod
    def _make_findings_yaml(*names: str) -> str:
        return yaml.safe_dump(
            {
                name: {
                    "name": name,
                    "place_type": "file",
                    "place": f"handler.py:{i+10}",
                    "title": f"finding {name}",
                    "summary": f"summary for {name}",
                    "severity": "high",
                    "confidence": "medium",
                    "details": "...",
                }
                for i, name in enumerate(names)
            }
        )

    @pytest.mark.asyncio
    async def test_one_task_per_finding(self, monkeypatch):
        from contractor.workflows.trace_annotation import OpenApiPath

        ctx = _make_context()
        findings_yaml = self._make_findings_yaml("sqli-list", "xss-list")

        async def fake_load(*, app_name, user_id, filename, **_):
            if filename == "user:vulnerability-reports/trace-annotation:openapi:items":
                return MagicMock(text=findings_yaml, inline_data=None)
            return None

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)
        workflow = TraceVerifyWorkflow(ctx)

        # Bypass the OpenAPI-loading outer loop — only this inner method
        # touches the findings artifact and the task queue.
        api_path = OpenApiPath(path="/items", operations=[])

        captured: list = []
        original_init = TaskRunner.__init__

        def capture_init(self, **kwargs):
            original_init(self, **kwargs)
            captured.append(self)

        monkeypatch.setattr(TaskRunner, "__init__", capture_init)
        monkeypatch.setattr(TaskRunner, "run", AsyncMock())

        await workflow._verify_path_findings(
            api_path=api_path,
            user_id="u",
            on_event=None,
        )

        assert len(captured) == 1
        runner = captured[0]
        assert len(runner.queue) == 2
        assert {item.template_key for item in runner.queue} == {"trace_verify"}
        assert {item.ref for item in runner.queue} == {
            "trace_verify:openapi:items:sqli-list",
            "trace_verify:openapi:items:xss-list",
        }
        for item in runner.queue:
            assert item.params["source_namespace"] == "trace-annotation:openapi:items"
            assert item.params["finding_name"] in {"sqli-list", "xss-list"}
            assert item.namespace == "trace-annotation:openapi:items"

    @pytest.mark.asyncio
    async def test_skips_path_with_no_findings(self, monkeypatch):
        from contractor.workflows.trace_annotation import OpenApiPath

        # Default _make_context's load_artifact returns None for everything.
        workflow = TraceVerifyWorkflow(_make_context())
        api_path = OpenApiPath(path="/items", operations=[])

        captured: list = []
        original_init = TaskRunner.__init__

        def capture_init(self, **kwargs):
            original_init(self, **kwargs)
            captured.append(self)

        monkeypatch.setattr(TaskRunner, "__init__", capture_init)
        monkeypatch.setattr(TaskRunner, "run", AsyncMock())

        await workflow._verify_path_findings(
            api_path=api_path,
            user_id="u",
            on_event=None,
        )

        # No findings → no TaskRunner created and no tasks queued.
        assert captured == []

    def test_template_loads(self):
        from contractor.runners.models import TaskTemplate

        t = TaskTemplate.load("trace_verify")
        assert t.key == "trace_verify"
        assert t.version


# ─── Vuln workflows: shared helpers ─────────────────────────────────────────────


def _patch_task_runners(monkeypatch) -> list:
    """Capture every TaskRunner created, stubbing .run(). Returns the runner list."""
    captured: list = []
    original_init = TaskRunner.__init__

    def capture_init(self, **kwargs):
        original_init(self, **kwargs)
        captured.append(self)

    monkeypatch.setattr(TaskRunner, "__init__", capture_init)
    monkeypatch.setattr(TaskRunner, "run", AsyncMock())
    return captured


def _flat_queue(runners: list) -> list:
    return [item for r in runners for item in r.queue]


def _findings_yaml(*specs: dict) -> str:
    return yaml.safe_dump({s["name"]: s for s in specs}, sort_keys=False)


# ─── VulnScanWorkflow ───────────────────────────────────────────────────────────


class TestVulnScanWorkflow:
    @pytest.mark.asyncio
    async def test_assembles_single_scan_task(self, monkeypatch):
        from contractor.workflows.vuln_scan import VulnScanWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanWorkflow(_make_context())
        await workflow._run_impl(user_id="u", on_event=None)

        queue = _flat_queue(captured)
        assert len(queue) == 1
        item = queue[0]
        assert item.template_key == "vuln_scan"
        assert item.ref == "vuln-scan:full"
        assert item.skills == ["vuln_scan"]


# ─── VulnScanTraceWorkflow ──────────────────────────────────────────────────────


class TestVulnScanTraceWorkflow:
    @pytest.mark.asyncio
    async def test_scan_only_when_no_findings(self, monkeypatch):
        from contractor.workflows.vuln_scan_trace import VulnScanTraceWorkflow

        # Default _make_context load_artifact → None → no findings → trace phase
        # skipped, only the BFS scan task is queued.
        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanTraceWorkflow(_make_context())
        await workflow._run_impl(user_id="u", on_event=None)

        queue = _flat_queue(captured)
        assert {item.template_key for item in queue} == {"vuln_scan"}
        assert queue[0].ref == "vuln-scan-trace:scan"

    @pytest.mark.asyncio
    async def test_trace_finding_assembles_task(self, monkeypatch):
        from contractor.workflows.vuln_scan_trace import VulnScanTraceWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanTraceWorkflow(_make_context())
        await workflow._trace_finding(
            finding={"name": "sqli-1", "place": "h.py:10", "title": "SQLi"},
            user_id="u",
            on_event=None,
        )

        queue = _flat_queue(captured)
        assert len(queue) == 1
        item = queue[0]
        assert item.template_key == "trace_annotation"
        assert item.ref == "vuln-scan-trace:trace:sqli-1"
        assert item.skills == ["trace"]
        assert item.params["operation_id"] == "sqli-1"
        assert "h.py:10" in item.params["operation_schema"]

    @pytest.mark.asyncio
    async def test_trace_finding_skips_without_name_or_place(self, monkeypatch):
        from contractor.workflows.vuln_scan_trace import VulnScanTraceWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanTraceWorkflow(_make_context())
        await workflow._trace_finding(
            finding={"name": "x", "place": ""}, user_id="u", on_event=None
        )
        assert _flat_queue(captured) == []

    @pytest.mark.asyncio
    async def test_load_findings_sorts_by_severity(self, monkeypatch):
        from contractor.workflows.vuln_scan_trace import VulnScanTraceWorkflow

        ctx = _make_context()
        yaml_text = _findings_yaml(
            {"name": "low1", "severity": "low", "place": "a"},
            {"name": "crit1", "severity": "critical", "place": "b"},
            {"name": "med1", "severity": "medium", "place": "c"},
        )
        ctx.artifact_service.load_artifact = AsyncMock(
            return_value=MagicMock(text=yaml_text, inline_data=None)
        )
        workflow = VulnScanTraceWorkflow(ctx)

        findings = await workflow._load_findings(user_id="u", namespace="ns")

        assert [f["name"] for f in findings] == ["crit1", "med1", "low1"]


# ─── VulnScanFastWorkflow ───────────────────────────────────────────────────────


class TestVulnScanFastWorkflow:
    def test_dedup_merges_by_file_and_cwe_keeping_higher_confidence(self):
        from contractor.workflows.vuln_scan_fast import VulnScanFastWorkflow

        findings = [
            {"name": "a", "place": "h.py", "details": "CWE-89 sqli", "confidence": "low"},
            {"name": "b", "place": "h.py", "details": "CWE-89 sqli", "confidence": "high"},
            {"name": "c", "place": "other.py", "details": "CWE-79 xss", "confidence": "medium"},
        ]
        deduped = VulnScanFastWorkflow._dedup(findings)

        # a and b collapse (same file+CWE) → the higher-confidence one wins.
        assert len(deduped) == 2
        h_py = next(f for f in deduped if f["place"] == "h.py")
        assert h_py["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_discovery_assembles_two_tasks(self, monkeypatch):
        from contractor.workflows.vuln_scan_fast import VulnScanFastWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanFastWorkflow(_make_context())
        await workflow._run_discovery(user_id="u", on_event=None)

        assert _template_keys(_flat_queue(captured)) == {
            "dependency_information",
            "project_information",
        }

    @pytest.mark.asyncio
    async def test_fast_scan_assembles_scan_task(self, monkeypatch):
        from contractor.workflows.vuln_scan_fast import VulnScanFastWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnScanFastWorkflow(_make_context())
        await workflow._run_fast_scan(user_id="u", on_event=None)

        queue = _flat_queue(captured)
        assert len(queue) == 1
        assert queue[0].template_key == "vuln_scan_fast"
        assert queue[0].ref == "vuln-scan-fast:full"


# ─── ExploitabilityWorkflow ─────────────────────────────────────────────────────


class TestExploitabilityWorkflow:
    def test_requires_target_url(self, monkeypatch):
        from contractor.workflows.exploitability import ExploitabilityWorkflow

        monkeypatch.delenv("CONTRACTOR_TARGET_URL", raising=False)
        with pytest.raises(ValueError, match="CONTRACTOR_TARGET_URL"):
            ExploitabilityWorkflow(_make_context())

    @pytest.mark.asyncio
    async def test_assess_finding_assembles_task(self, monkeypatch):
        from contractor.workflows.exploitability import ExploitabilityWorkflow

        monkeypatch.setenv("CONTRACTOR_TARGET_URL", "http://localhost:5002")
        captured = _patch_task_runners(monkeypatch)
        workflow = ExploitabilityWorkflow(_make_context())
        await workflow._assess_finding(
            finding={"name": "idor-1", "title": "IDOR", "place": "v.py:3"},
            user_id="u",
            on_event=None,
        )

        queue = _flat_queue(captured)
        assert len(queue) == 1
        item = queue[0]
        assert item.template_key == "exploitability_assessment"
        assert item.ref == "exploitability:idor-1"
        assert item.skills == ["exploit", "code-exec"]
        assert item.params["finding_name"] == "idor-1"
        assert item.params["source_namespace"] == "exploitability:idor-1"

    @pytest.mark.asyncio
    async def test_assess_finding_skips_unnamed(self, monkeypatch):
        from contractor.workflows.exploitability import ExploitabilityWorkflow

        monkeypatch.setenv("CONTRACTOR_TARGET_URL", "http://localhost:5002")
        captured = _patch_task_runners(monkeypatch)
        workflow = ExploitabilityWorkflow(_make_context())
        await workflow._assess_finding(finding={"name": ""}, user_id="u", on_event=None)
        assert _flat_queue(captured) == []

    @pytest.mark.asyncio
    async def test_load_findings_parses_seed(self, monkeypatch):
        from contractor.workflows.exploitability import ExploitabilityWorkflow

        monkeypatch.setenv("CONTRACTOR_TARGET_URL", "http://localhost:5002")
        ctx = _make_context()
        ctx.artifact_service.load_artifact = AsyncMock(
            return_value=MagicMock(
                text=_findings_yaml({"name": "f1", "severity": "high"}),
                inline_data=None,
            )
        )
        workflow = ExploitabilityWorkflow(ctx)
        findings = await workflow._load_findings(user_id="u")
        assert [f["name"] for f in findings] == ["f1"]


# ─── VulnAssessWorkflow ─────────────────────────────────────────────────────────


class TestVulnAssessWorkflow:
    @pytest.mark.asyncio
    async def test_oas_stage_assembles_four_tasks(self, monkeypatch):
        from contractor.workflows.vuln_assess import VulnAssessWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnAssessWorkflow(_make_context())
        await workflow._run_oas_stage(user_id="u", on_event=None)

        assert _template_keys(_flat_queue(captured)) == {
            "dependency_information",
            "project_information",
            "oas_update",
            "oas_validate",
        }

    @pytest.mark.asyncio
    async def test_oas_stage_artifact_references_resolve(self, monkeypatch):
        from contractor.workflows.vuln_assess import VulnAssessWorkflow

        captured = _patch_task_runners(monkeypatch)
        workflow = VulnAssessWorkflow(_make_context())
        await workflow._run_oas_stage(user_id="u", on_event=None)

        queue = _flat_queue(captured)
        produced = _template_keys(queue)
        for artifact_ref in _all_artifact_refs(queue):
            assert _producing_task_key(artifact_ref) in produced, (
                f"artifact {artifact_ref!r} referenced but not produced upstream"
            )

    @pytest.mark.asyncio
    async def test_oas_stage_skipped_when_artifact_exists(self, monkeypatch):
        from contractor.workflows.vuln_assess.workflow import (
            OAS_ARTIFACT, VulnAssessWorkflow)

        ctx = _make_context()

        async def fake_load(*, app_name, user_id, filename, **_):
            if filename == OAS_ARTIFACT:
                return MagicMock(text="exists", inline_data=None)
            return None

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)
        captured = _patch_task_runners(monkeypatch)
        workflow = VulnAssessWorkflow(ctx)
        await workflow._run_oas_stage(user_id="u", on_event=None)

        # Whole OAS stage short-circuits — no tasks queued.
        assert _flat_queue(captured) == []
