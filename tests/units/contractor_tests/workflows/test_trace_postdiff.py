"""Unit tests for the trace-postdiff workflow (annotate-only trace stage +
post-diff vuln-analytics stage) and its diff helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from google.adk.artifacts import BaseArtifactService
from google.genai import types

from cli.fs import RootedLocalFileSystem
from contractor.runners.models import RenderedTask, TaskTemplate
from contractor.utils import load_prompt
from contractor.workflows import WorkflowContext, get_workflows
from contractor.workflows.namespaces import (
    TRACE_NAMESPACE_PREFIXES,
    TRACE_POSTDIFF_NAMESPACE_PREFIX,
)
from contractor.workflows.trace_postdiff import TracePostDiffWorkflow
from contractor.workflows.trace_postdiff.workflow import (
    _diff_header_path,
    filter_diff_by_files,
    truncate_diff,
)

OPENAPI_DOC = {
    "openapi": "3.0.0",
    "info": {"title": "t", "version": "1"},
    "paths": {
        "/users/{user-id}": {
            "get": {"operationId": "getUser", "responses": {"200": {}}},
            "delete": {"operationId": "deleteUser", "responses": {"204": {}}},
        },
    },
}


def _make_context(tmp_path: Path, doc: dict = OPENAPI_DOC) -> WorkflowContext:
    (tmp_path / "app.py").write_text("def handler():\n    pass\n")

    artifact_service = MagicMock(spec=BaseArtifactService)

    async def load_artifact(*, app_name, user_id, filename):
        if filename == "oas-openapi-building":
            return types.Part.from_text(text=yaml.safe_dump(doc))
        return None

    artifact_service.load_artifact = AsyncMock(side_effect=load_artifact)
    artifact_service.save_artifact = AsyncMock()

    return WorkflowContext(
        project_path=tmp_path,
        folder_name="/",
        model="lm-studio-test",
        app_name="contractor-test",
        user_id="u",
        artifact_service=artifact_service,
        fs=RootedLocalFileSystem(str(tmp_path)),
    )


class TestDiffHelpers:
    def test_header_path_simple(self):
        line = "diff --overlay a/src/app.py b/src/app.py"
        assert _diff_header_path(line) == "/src/app.py"

    def test_header_path_with_space(self):
        line = "diff --overlay a/my dir/f.py b/my dir/f.py"
        assert _diff_header_path(line) == "/my dir/f.py"

    def test_header_path_non_header(self):
        assert _diff_header_path("+++ b/src/app.py") is None

    def test_filter_keeps_only_named_files(self):
        diff = (
            "diff --overlay a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "+# @trace target=x args= calls=\n"
            "diff --overlay a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "+irrelevant\n"
        )
        kept = filter_diff_by_files(diff, {"/a.py"})
        assert "a/a.py" in kept
        assert "@trace" in kept
        assert "b.py" not in kept
        assert "irrelevant" not in kept

    def test_filter_empty_inputs(self):
        assert filter_diff_by_files("", {"/a.py"}) == ""
        assert filter_diff_by_files("diff --overlay a/a.py b/a.py", set()) == ""

    def test_truncate(self):
        assert truncate_diff("short", 100) == "short"
        out = truncate_diff("x" * 200, 100)
        assert out.startswith("x" * 100)
        assert "truncated" in out


class TestSurfaces:
    def test_registry_exposes_trace_postdiff(self):
        assert get_workflows()["trace-postdiff"] is TracePostDiffWorkflow

    def test_namespace_prefix_registered(self):
        assert TRACE_POSTDIFF_NAMESPACE_PREFIX in TRACE_NAMESPACE_PREFIXES

    def test_analytics_prompt_loads(self):
        prompt = load_prompt("vuln_analytics_agent")
        assert "report_vulnerability" in prompt
        assert "@trace" in prompt

    def test_analytics_template_renders(self):
        template = TaskTemplate.load("vuln_analytics")
        rendered = RenderedTask.from_template(
            template=template,
            variables={
                "target_summary": "TARGET-SUMMARY-SENTINEL",
                "trace_diff": "TRACE-DIFF-SENTINEL",
            },
            params={},
            artifacts={},
        )
        text = rendered._format_task()
        assert "TARGET-SUMMARY-SENTINEL" in text
        assert "TRACE-DIFF-SENTINEL" in text


@pytest.mark.asyncio
class TestTwoStageRun:
    async def _run(
        self, tmp_path, monkeypatch, *, annotate: bool, doc: dict = OPENAPI_DOC
    ):
        """Run the workflow with both agent builders and the runner faked.

        When ``annotate`` is set, the fake trace stage writes an annotation
        into the overlay (as the real trace_agent would via its tools).
        """
        import contractor.workflows.trace_postdiff.workflow as wf_mod

        ctx = _make_context(tmp_path, doc)
        workflow = TracePostDiffWorkflow(ctx)

        trace_builds: list[dict] = []
        analytics_builds: list[dict] = []
        runs: list[dict] = []

        def fake_trace_agent(name, fs, **kwargs):
            trace_builds.append(kwargs)
            agent = MagicMock()
            agent._is_trace = True
            return agent

        def fake_analytics_agent(name, fs, **kwargs):
            analytics_builds.append(kwargs)
            agent = MagicMock()
            agent._is_trace = False
            return agent

        async def fake_run(self, *, agent, message, event_name, **kwargs):
            runs.append({"agent": agent, "message": message, "event": event_name})
            if agent._is_trace and annotate:
                # Unique content per run so every group sees fresh changes.
                workflow.overlayfs.pipe_file(
                    "/app.py",
                    b"# @trace target=getUser args=user_id:tainted calls=\n"
                    b"def handler():\n    pass\n"
                    + f"# run {len(runs)}\n".encode(),
                )

        monkeypatch.setattr(wf_mod, "build_trace_agent", fake_trace_agent)
        monkeypatch.setattr(wf_mod, "build_vuln_analytics_agent", fake_analytics_agent)
        monkeypatch.setattr(wf_mod, "inject_skills", AsyncMock())
        # AgentRunner is a pydantic model — patch the method on the class.
        monkeypatch.setattr(wf_mod.AgentRunner, "run", fake_run)

        await workflow._run_impl(user_id="u", on_event=None)
        return workflow, trace_builds, analytics_builds, runs

    async def test_annotate_stage_disables_vuln_reporting(
        self, tmp_path, monkeypatch
    ):
        _, trace_builds, _, _ = await self._run(
            tmp_path, monkeypatch, annotate=True
        )
        # One trace run per operation (get + delete).
        assert len(trace_builds) == 2
        for build in trace_builds:
            assert build["enable_vuln_reporting"] is False
            # group_depth=1 → namespace keyed by the route prefix.
            assert build["namespace"] == "trace-postdiff:openapi:users"

    async def test_analytics_stage_receives_annotation_diff(
        self, tmp_path, monkeypatch
    ):
        _, _, analytics_builds, runs = await self._run(
            tmp_path, monkeypatch, annotate=True
        )
        assert len(analytics_builds) == 1
        assert (
            analytics_builds[0]["namespace"] == "trace-postdiff:openapi:users"
        )

        analytics_runs = [r for r in runs if r["event"].endswith(":analytics")]
        assert len(analytics_runs) == 1
        message = analytics_runs[0]["message"]
        assert "diff --overlay a/app.py b/app.py" in message
        assert "@trace target=getUser" in message
        assert "/users/{user-id}" in message  # target summary present

    async def test_analytics_skipped_without_annotations(
        self, tmp_path, monkeypatch
    ):
        _, trace_builds, analytics_builds, runs = await self._run(
            tmp_path, monkeypatch, annotate=False
        )
        assert len(trace_builds) == 2
        assert analytics_builds == []
        assert all(not r["event"].endswith(":analytics") for r in runs)

    async def test_sibling_paths_share_group_and_analytics_run(
        self, tmp_path, monkeypatch
    ):
        doc = {
            "openapi": "3.0.0",
            "info": {"title": "t", "version": "1"},
            "paths": {
                "/users/{user-id}": {
                    "get": {"operationId": "getUser", "responses": {"200": {}}},
                },
                "/users/export": {
                    "get": {"operationId": "exportUsers", "responses": {"200": {}}},
                },
                "/admin/stats": {
                    "get": {"operationId": "adminStats", "responses": {"200": {}}},
                },
            },
        }
        _, trace_builds, analytics_builds, runs = await self._run(
            tmp_path, monkeypatch, annotate=True, doc=doc
        )
        # Three operations traced, but only two route groups analyzed.
        assert len(trace_builds) == 3
        assert {b["namespace"] for b in trace_builds} == {
            "trace-postdiff:openapi:users",
            "trace-postdiff:openapi:admin",
        }
        assert {b["namespace"] for b in analytics_builds} == {
            "trace-postdiff:openapi:users",
            "trace-postdiff:openapi:admin",
        }
        assert len(analytics_builds) == 2


@pytest.mark.asyncio
class TestPluginEmitWiring:
    async def test_plugin_emit_reaches_event_handler(self, tmp_path):
        """The plugins' ``emit(event_type, **payload)`` calls must thread
        through ``AgentRunner._emit`` to the run's event handler.

        Regression guard: ``AgentRunner._emit`` takes the handler as its
        first positional arg, and passing the unbound method to a plugin
        crashed every run in ``before_run_callback``.
        """
        ctx = _make_context(tmp_path)
        workflow = TracePostDiffWorkflow(ctx)

        events = []

        async def on_event(event):
            events.append(event)

        plugins = workflow._plugins("ev-name", 1, "sid", on_event)
        invocation_context = MagicMock(invocation_id="inv-1", agent_name="a")
        for plugin in plugins:
            await plugin.before_run_callback(invocation_context=invocation_context)

        # AdkTracePlugin emits agent_run_start; AdkMetricsPlugin's hook may
        # be a no-op — the wiring is shared, one delivered event proves it.
        assert events, "plugin emit never reached the event handler"
        assert any(e.type == "trace_agent_run_start" for e in events) or any(
            "agent_run_start" in e.type for e in events
        )
