"""Tests for deterministic worker-usage observations.

Covers the pure projector (every knob combination), the formatter embedding,
the workflow-config loader, and the end-to-end success / malformed projection
through ``execute_current_subtask``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

import contractor.tools.tasks as m
from contractor.tools.observations import (
    SKILLS_READ_STATE_KEY,
    WORKER_USAGE_STATE_KEY,
    ObservationConfig,
    has_observations,
    project_usage,
)
from contractor.tools.tasks import SubtaskExecutionResult, SubtaskFormatter
from contractor.tools.tasks.models import Subtask
from contractor.workflows.config import WorkflowConfig
from tests.units.contractor_tests.helpers import MockAgentTool, mk_tool_context

# ---------------------------------------------------------------------------
# project_usage — the pure projector
# ---------------------------------------------------------------------------

_RAW_SNAPSHOT = {
    "tools": {
        "read_file": {"calls": 3, "errors": 1},
        "search_code": {"calls": 2, "errors": 0},
    },
    "fs_coverage": {"a.py": 2, "b.py": 1},
}


def _state(snapshot=_RAW_SNAPSHOT, skills=("trace/references/sinks",)):
    return {
        WORKER_USAGE_STATE_KEY: snapshot,
        SKILLS_READ_STATE_KEY: list(skills),
    }


def test_project_usage_disabled_returns_none():
    assert project_usage(_state(), ObservationConfig(enabled=False)) is None


def test_project_usage_default_counts_only():
    out = project_usage(_state(), ObservationConfig(enabled=True))
    assert out == {
        "tools": {"read_file": 3, "search_code": 2},  # counts, no error breakdown
        "files": {"a.py": 2, "b.py": 1},
        "skills_read": ["trace/references/sinks"],
    }


def test_project_usage_include_tool_errors():
    out = project_usage(
        _state(), ObservationConfig(enabled=True, include_tool_errors=True)
    )
    assert out["tools"] == {
        "read_file": {"calls": 3, "errors": 1},
        "search_code": {"calls": 2, "errors": 0},
    }


def test_project_usage_tracked_tools_allowlist():
    out = project_usage(
        _state(),
        ObservationConfig(enabled=True, tracked_tools=("read_file",)),
    )
    assert out["tools"] == {"read_file": 3}


def test_project_usage_section_toggles():
    out = project_usage(
        _state(),
        ObservationConfig(
            enabled=True, track_tools=False, track_files=False, track_skills=True
        ),
    )
    assert out == {"skills_read": ["trace/references/sinks"]}


def test_project_usage_empty_state_when_enabled():
    # No worker activity at all — sections present but empty (signal for the
    # malformed path).
    out = project_usage({}, ObservationConfig(enabled=True))
    assert out == {"tools": {}, "files": None, "skills_read": []}
    assert has_observations(out) is False


def test_has_observations():
    assert has_observations(None) is False
    assert has_observations({}) is False
    assert has_observations({"tools": {}, "files": None, "skills_read": []}) is False
    assert has_observations({"tools": {"x": 1}}) is True


# ---------------------------------------------------------------------------
# format_task_record — embedding
# ---------------------------------------------------------------------------

_SUBTASK = Subtask(task_id="1", title="t", description="d", status="done")
_RESULT = SubtaskExecutionResult(
    task_id="1", status="done", output="o", summary="s"
)
_USAGE = {"tools": {"read_file": 2}, "files": {"a.py": 1}, "skills_read": ["sk"]}


def test_format_task_record_json_embeds_usage_as_field():
    rec = SubtaskFormatter(_format="json").format_task_record(
        _SUBTASK, _RESULT, usage=_USAGE
    )
    assert rec["usage"] == _USAGE
    # kept distinct from worker-reported fields
    assert rec["output"] == "o"


def test_format_task_record_json_no_usage_when_none():
    rec = SubtaskFormatter(_format="json").format_task_record(_SUBTASK, _RESULT)
    assert "usage" not in rec


def test_format_task_record_xml_appends_block():
    rec = SubtaskFormatter(_format="xml").format_task_record(
        _SUBTASK, _RESULT, usage=_USAGE
    )
    assert isinstance(rec, str)
    assert "[observed_usage]" in rec
    assert "read_file" in rec
    assert "skills_read: sk" in rec


# ---------------------------------------------------------------------------
# WorkflowConfig loader
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_config_observations_absent_is_disabled(tmp_path):
    cfg = WorkflowConfig.load(_write_cfg(tmp_path, "budgets: {}\n"))
    assert cfg.observations == ObservationConfig()
    assert cfg.observations.enabled is False


def test_config_observations_parsed(tmp_path):
    body = (
        "observations:\n"
        "  enabled: true\n"
        "  malformed_only: true\n"
        "  include_tool_errors: true\n"
        "  tracked_tools: ['skills_read', 'read_file']\n"
    )
    cfg = WorkflowConfig.load(_write_cfg(tmp_path, body))
    obs = cfg.observations
    assert obs.enabled is True
    assert obs.malformed_only is True
    assert obs.include_tool_errors is True
    assert obs.tracked_tools == ("skills_read", "read_file")  # normalised to tuple


def test_config_observations_invalid_tracked_tools(tmp_path):
    body = "observations:\n  enabled: true\n  tracked_tools: 'read_file'\n"
    with pytest.raises(ValueError, match="tracked_tools"):
        WorkflowConfig.load(_write_cfg(tmp_path, body))


def test_config_observations_unknown_key(tmp_path):
    body = "observations:\n  enabled: true\n  bogus: 1\n"
    with pytest.raises(ValueError, match="invalid observations config"):
        WorkflowConfig.load(_write_cfg(tmp_path, body))


# ---------------------------------------------------------------------------
# execute_current_subtask — end-to-end projection
# ---------------------------------------------------------------------------


def _mk_tools(monkeypatch, *, worker, observations):
    from contractor.tools.tasks import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "AgentTool", MockAgentTool)
    monkeypatch.setattr(_tools_mod, "instrument_worker", lambda w, *a, **k: w)
    tools = m.task_tools(
        name="tm",
        max_tasks=100,
        worker=worker,
        fmt=SubtaskFormatter(_format="json"),
        worker_instrumentation=False,
        use_input_schema=True,
        use_output_schema=False,
        use_skip=False,
        use_summarization=False,
        observations=observations,
    )
    return {fn.__name__: fn for fn in tools}


def _mk_worker():
    worker = type("Worker", (), {})()
    worker.run_async = AsyncMock()
    worker.tools = []
    worker.model = "gpt-3.5-turbo"
    worker.instruction = ""
    return worker


def _writes_usage(status, output, summary):
    """Worker side-effect that mimics the plugin writing usage into state."""

    async def _run(*, args, tool_context):
        tool_context.state[WORKER_USAGE_STATE_KEY] = {
            "tools": {"read_file": {"calls": 2, "errors": 0}},
            "fs_coverage": {"a.py": 1},
        }
        tool_context.state[SKILLS_READ_STATE_KEY] = ["trace/references/sinks"]
        if output is None:
            return "this is not parseable json"
        return json.dumps(
            {
                "task_id": args["task_id"],
                "status": status,
                "output": output,
                "summary": summary,
            }
        )

    return _run


@pytest.mark.anyio
async def test_execute_success_injects_observed_usage(monkeypatch):
    worker = _mk_worker()
    worker.run_async.side_effect = _writes_usage("done", "did it", "ok")
    tool = _mk_tools(monkeypatch, worker=worker, observations=ObservationConfig(enabled=True))
    ctx = mk_tool_context()
    tool["add_subtask"](title="t", description="d", tool_context=ctx)

    res = await tool["execute_current_subtask"](tool_context=ctx)

    assert res["observed_usage"]["tools"] == {"read_file": 2}
    assert res["observed_usage"]["skills_read"] == ["trace/references/sinks"]
    assert res["record"]["usage"]["files"] == {"a.py": 1}


@pytest.mark.anyio
async def test_execute_disabled_has_no_usage(monkeypatch):
    worker = _mk_worker()
    worker.run_async.side_effect = _writes_usage("done", "did it", "ok")
    tool = _mk_tools(monkeypatch, worker=worker, observations=ObservationConfig(enabled=False))
    ctx = mk_tool_context()
    tool["add_subtask"](title="t", description="d", tool_context=ctx)

    res = await tool["execute_current_subtask"](tool_context=ctx)

    assert "observed_usage" not in res
    assert "usage" not in res["record"]


@pytest.mark.anyio
async def test_malformed_only_suppresses_success_but_keeps_malformed(monkeypatch):
    obs = ObservationConfig(enabled=True, malformed_only=True)

    # success path: suppressed
    worker = _mk_worker()
    worker.run_async.side_effect = _writes_usage("done", "did it", "ok")
    tool = _mk_tools(monkeypatch, worker=worker, observations=obs)
    ctx = mk_tool_context()
    tool["add_subtask"](title="t", description="d", tool_context=ctx)
    res_ok = await tool["execute_current_subtask"](tool_context=ctx)
    assert "observed_usage" not in res_ok

    # malformed path: still injected
    worker2 = _mk_worker()
    worker2.run_async.side_effect = _writes_usage("done", None, None)
    tool2 = _mk_tools(monkeypatch, worker=worker2, observations=obs)
    ctx2 = mk_tool_context()
    tool2["add_subtask"](title="t", description="d", tool_context=ctx2)
    res_bad = await tool2["execute_current_subtask"](tool_context=ctx2)
    assert res_bad["record"]["status"] == "malformed"
    assert res_bad["observed_usage"]["tools"] == {"read_file": 2}


@pytest.mark.anyio
async def test_per_subtask_reset_isolates_usage(monkeypatch):
    # First subtask uses tools; second worker run writes nothing -> usage must
    # reset to empty rather than carry the first subtask's facts forward.
    worker = _mk_worker()
    calls = {"n": 0}

    async def _run(*, args, tool_context):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_context.state[WORKER_USAGE_STATE_KEY] = {
                "tools": {"read_file": {"calls": 5, "errors": 0}},
                "fs_coverage": None,
            }
        # second run writes nothing (simulates a zero-tool worker run)
        return json.dumps(
            {"task_id": args["task_id"], "status": "done", "output": "x", "summary": "s"}
        )

    worker.run_async.side_effect = _run
    tool = _mk_tools(monkeypatch, worker=worker, observations=ObservationConfig(enabled=True))
    ctx = mk_tool_context()
    tool["add_subtask"](title="t0", description="d", tool_context=ctx)
    tool["add_subtask"](title="t1", description="d", tool_context=ctx)

    r0 = await tool["execute_current_subtask"](tool_context=ctx)
    assert r0["observed_usage"]["tools"] == {"read_file": 5}

    r1 = await tool["execute_current_subtask"](tool_context=ctx)
    # reset cleared the carried-over snapshot; second run did no work -> pruned
    assert "observed_usage" not in r1
