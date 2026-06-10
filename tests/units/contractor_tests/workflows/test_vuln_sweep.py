"""Unit tests for the vuln-sweep two-pass workflow (per-class BFS
nomination sweep → DFS trace of survivors)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from google.adk.artifacts import BaseArtifactService

from contractor.runners.task_runner import TaskRunner
from contractor.workflows import WorkflowContext, get_workflows
from contractor.workflows.vuln_sweep import VulnSweepWorkflow
from contractor.workflows.vuln_sweep.workflow import SINK_CLASSES


def _make_context() -> WorkflowContext:
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
    )


def _findings_yaml(*names: str, severity="high", confidence="low", place="app.py"):
    return yaml.safe_dump(
        {
            n: {
                "title": n,
                "place": place,
                "place_type": "file",
                "severity": severity,
                "confidence": confidence,
                "summary": "s",
                "details": "d",
            }
            for n in names
        }
    )


class TestSurfaces:
    def test_registry_exposes_vuln_sweep(self):
        assert get_workflows()["vuln-sweep"] is VulnSweepWorkflow

    def test_has_absence_class(self):
        keys = {c.key for c in SINK_CLASSES}
        assert "missing-access-control" in keys
        # Several distinct classes, all with guidance text.
        assert len(keys) >= 4
        assert all(c.guidance for c in SINK_CLASSES)

    def test_inherits_trace_phase(self):
        from contractor.workflows.vuln_scan_trace import VulnScanTraceWorkflow

        assert issubclass(VulnSweepWorkflow, VulnScanTraceWorkflow)
        # CFG override points the inherited phase at the sweep config.
        assert VulnSweepWorkflow.CFG is not VulnScanTraceWorkflow.CFG

    def test_nomination_template_renders(self):
        from contractor.runners.models import RenderedTask, TaskTemplate

        template = TaskTemplate.load("sink_nomination")
        rendered = RenderedTask.from_template(
            template=template,
            variables={
                "project_path": "src",
                "sink_class": "injection",
                "class_guidance": "GUIDANCE-SENTINEL",
            },
            params={},
            artifacts={},
        )
        text = rendered._format_task()
        assert "injection" in text
        assert "GUIDANCE-SENTINEL" in text


@pytest.mark.asyncio
class TestSweepRun:
    async def _capture(self, ctx, monkeypatch):
        """Run _run_impl with TaskRunner.run faked; return the queued tasks."""
        runners: list = []
        original_init = TaskRunner.__init__

        def capture_init(self, **kwargs):
            original_init(self, **kwargs)
            runners.append(self)

        monkeypatch.setattr(TaskRunner, "__init__", capture_init)
        monkeypatch.setattr(TaskRunner, "run", AsyncMock())

        workflow = VulnSweepWorkflow(ctx)
        await workflow._run_impl(user_id="u", on_event=None)
        return [item for r in runners for item in r.queue]

    async def test_one_nomination_task_per_class(self, monkeypatch):
        ctx = _make_context()  # load_artifact → None: no nominations, no trace
        queue = await self._capture(ctx, monkeypatch)

        sweep_tasks = [t for t in queue if t.template_key == "sink_nomination"]
        assert len(sweep_tasks) == len(SINK_CLASSES)
        # Each class gets its own namespace and stable ref.
        namespaces = {t.namespace for t in sweep_tasks}
        assert namespaces == {
            f"vuln-sweep:sweep:{c.key}" for c in SINK_CLASSES
        }
        # No trace tasks queued when there are no nominations.
        assert not [t for t in queue if t.template_key == "trace_annotation"]

    async def test_nominations_deduped_and_traced(self, monkeypatch):
        ctx = _make_context()

        # Two classes nominate; one slug is shared (same place+name) and
        # must dedup to a single trace task.
        per_ns = {
            "user:vulnerability-reports/vuln-sweep:sweep:injection": _findings_yaml(
                "injection-sqli", "shared-dup"
            ),
            "user:vulnerability-reports/vuln-sweep:sweep:deserialization": (
                _findings_yaml("shared-dup")
            ),
        }

        async def fake_load(*, app_name, user_id, filename, **_):
            text = per_ns.get(filename)
            if text is None:
                return None
            return MagicMock(text=text, inline_data=None)

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)
        queue = await self._capture(ctx, monkeypatch)

        trace_tasks = [t for t in queue if t.template_key == "trace_annotation"]
        traced_names = {t.params["operation_id"] for t in trace_tasks}
        assert traced_names == {"injection-sqli", "shared-dup"}

    async def test_cap_limits_trace_phase(self, monkeypatch):
        ctx = _make_context()
        many = _findings_yaml(*[f"inj-{i}" for i in range(100)])

        async def fake_load(*, app_name, user_id, filename, **_):
            if filename.endswith("vuln-sweep:sweep:injection"):
                return MagicMock(text=many, inline_data=None)
            return None

        ctx.artifact_service.load_artifact = AsyncMock(side_effect=fake_load)
        queue = await self._capture(ctx, monkeypatch)

        trace_tasks = [t for t in queue if t.template_key == "trace_annotation"]
        assert len(trace_tasks) == VulnSweepWorkflow.CFG.budgets.max_trace_nominations
