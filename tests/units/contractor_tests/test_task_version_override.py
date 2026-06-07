"""Task-version env override (CONTRACTOR_TASK_VERSION_<NAME>) for A/B eval-gating
a task body without flipping the manifest's `active:`."""
from __future__ import annotations

from contractor.runners.models import TaskTemplate


def test_active_version_default():
    t = TaskTemplate.load("trace_annotation")
    assert t.version == "v1"  # manifest active


def test_explicit_version_arg_wins():
    t = TaskTemplate.load("trace_annotation", "v3")
    assert t.version == "v3"


def test_env_override_selects_version(monkeypatch):
    monkeypatch.setenv("CONTRACTOR_TASK_VERSION_TRACE_ANNOTATION", "v3")
    t = TaskTemplate.load("trace_annotation")
    assert t.version == "v3"


def test_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("CONTRACTOR_TASK_VERSION_TRACE_ANNOTATION", "v3")
    t = TaskTemplate.load("trace_annotation", "v1")
    assert t.version == "v1"


def test_v3_renders_without_brace_keyerror():
    """v3 must format cleanly with the workflow's scope vars — no stray
    identifier-shaped braces (the ADK/str.format pitfall)."""
    t = TaskTemplate.load("trace_annotation", "v3")
    scope = {"operation_id": "getAccount", "operation_schema": "openapi: 3.0.0"}
    rendered = (t.objective + t.instructions + t.output_format).format(**scope)
    assert "getAccount" in rendered
    # delegation + no tool/mechanics leakage into the planner surface
    assert "trace` skill" in t.instructions
    assert "annotate_trace" not in t.instructions and "insert_line" not in t.instructions
    # output aligned to the agent §OUTPUT headers
    assert "## Annotations Inserted" in t.output_format and "## Findings" in t.output_format
