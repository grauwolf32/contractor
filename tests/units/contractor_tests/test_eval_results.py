"""Unit tests for the standardized eval envelope (``tests/eval/results.py``)."""
from __future__ import annotations

import asyncio
import json

from tests.eval.results import (
    SCHEMA,
    AttemptOutcome,
    CaseResult,
    EvalRun,
    EvalSink,
    FixtureResult,
    derive_headline,
    derive_totals,
    metrics_from_task,
    pass_at,
    write_eval_results,
)


def _outcome(passed, **detail):
    return AttemptOutcome(passed=passed, metrics={"total_tokens": 10}, detail=detail)


# ───────────────────────── pass_at ─────────────────────────


def test_pass_at_single_attempt_no_runs():
    async def attempt(i):
        return _outcome(True, verdict="exploitable")

    case = asyncio.run(pass_at("c1", attempt, 1))
    assert isinstance(case, CaseResult)
    assert case.passed is True
    assert case.pass_count == 1
    assert case.attempts == 1
    assert case.runs is None  # no per-attempt breakdown for n=1
    assert case.detail == {"verdict": "exploitable"}


def test_pass_at_any_attempt_passes():
    outcomes = [False, False, True]

    async def attempt(i):
        return _outcome(outcomes[i], idx=i)

    case = asyncio.run(pass_at("c1", attempt, 3))
    assert case.passed is True          # pass@3: at least one passed
    assert case.pass_count == 1
    assert case.attempts == 3
    assert case.runs is not None and len(case.runs) == 3
    # representative attempt is the first PASSING one (idx 2)
    assert case.detail == {"idx": 2}


def test_pass_at_all_fail_uses_first_as_representative():
    async def attempt(i):
        return _outcome(False, idx=i)

    case = asyncio.run(pass_at("c1", attempt, 2))
    assert case.passed is False
    assert case.pass_count == 0
    assert case.detail == {"idx": 0}    # falls back to first attempt


# ───────────────────────── derivation ─────────────────────────


def _fixtures(*cases):
    return [FixtureResult(slug="fx", cases=list(cases)).to_dict()]


def test_derive_headline_generic_is_just_pass_rate():
    fx = _fixtures(
        CaseResult("a", True, 1, 1),
        CaseResult("b", False, 0, 1),
    )
    h = derive_headline("generic", fx)
    assert h == {"pass_rate": 0.5, "passed": 1, "total": 2}


def test_derive_headline_detection_micro_prf():
    fx = _fixtures(
        CaseResult("a", True, 1, 1, detail={"tp": 3, "fp": 1, "fn": 1}),
        CaseResult("b", True, 1, 1, detail={"tp": 1, "fp": 0, "fn": 0}),
    )
    h = derive_headline("detection", fx)
    # tp=4 fp=1 fn=1 → P=4/5=.8 R=4/5=.8 F1=.8
    assert h["precision"] == 0.8 and h["recall"] == 0.8 and h["f1"] == 0.8
    assert h["pass_rate"] == 1.0


def test_derive_headline_verdict_evidence_rate():
    fx = _fixtures(
        CaseResult("a", True, 1, 1, detail={"has_evidence": True}),
        CaseResult("b", False, 0, 1, detail={"has_evidence": False}),
    )
    assert derive_headline("verdict", fx)["evidence_rate"] == 0.5


def test_derive_headline_capture_chain_rate():
    fx = _fixtures(
        CaseResult("a", True, 1, 1, detail={"chain": True}),
        CaseResult("b", True, 1, 1, detail={"chain": False}),
    )
    assert derive_headline("capture", fx)["chain_rate"] == 0.5


def test_derive_totals_aggregates_tokens_and_tools():
    fx = _fixtures(
        CaseResult("a", True, 1, 1, metrics={
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "total_tool_calls": 5, "llm_calls": 3,
            "tool_counts": {"http_request": 4, "read_file": 1},
        }),
        CaseResult("b", True, 1, 1, metrics={
            "input_tokens": 50, "output_tokens": 10, "total_tokens": 60,
            "total_tool_calls": 2, "llm_calls": 1,
            "tool_counts": {"http_request": 2},
        }),
    )
    t = derive_totals(fx)
    assert t["total_tokens"] == 180
    assert t["total_tool_calls"] == 7
    assert t["llm_calls"] == 4
    assert t["tool_counts"]["http_request"] == 6
    assert t["cases"] == 2 and t["fixtures"] == 1


def test_metrics_from_events_counts_errors_and_tokens():
    from tests.eval.results import metrics_from_events
    events = [
        {"event_type": "tool_call", "tool_name": "http_request"},
        {"event_type": "tool_result", "result_error": False, "execution_time_ms": 5},
        {"event_type": "tool_call", "tool_name": "http_request"},
        {"event_type": "tool_result", "result_error": True},      # a result error
        {"event_type": "tool_exception"},                          # an exception
        {"event_type": "tool_call", "tool_name": "skills_read"},
        {"event_type": "llm_usage", "usage": {"input": 100, "output": 20, "total": 120}},
    ]
    m = metrics_from_events(events)
    assert m["total_tool_calls"] == 3
    assert m["tool_errors"] == 2          # result_error + tool_exception
    assert m["llm_calls"] == 1 and m["total_tokens"] == 120
    assert m["tool_counts"]["skills_read"] == 1


def test_derive_totals_includes_errors_and_skill_reads():
    fx = _fixtures(
        CaseResult("a", True, 1, 1, metrics={
            "tool_errors": 2, "tool_counts": {"skills_read": 3, "http_request": 4}}),
        CaseResult("b", True, 1, 1, metrics={
            "tool_errors": 1, "tool_counts": {"skills_read": 1}}),
    )
    t = derive_totals(fx)
    assert t["tool_errors"] == 3
    assert t["skill_reads"] == 4


def test_metrics_from_task_folds_taskmetrics():
    from tests.eval.task_harness import TaskMetrics

    m = TaskMetrics(task_ref="t:0")
    m.tool_counts.update({"http_request": 3})
    m.total_tool_calls = 3
    m.llm_calls = 2
    m.input_tokens = 80
    m.output_tokens = 20
    m.total_tokens = 100
    folded = metrics_from_task({"t:0": m})
    assert folded["total_tokens"] == 100
    assert folded["total_tool_calls"] == 3
    assert folded["tool_counts"] == {"http_request": 3}


def test_pass_artifact_root_isolates_passes():
    """pass@M isolation: each pass gets a distinct artifact-tree root, while
    no run_id leaves the base dir unchanged (within-pass reuse)."""
    from pathlib import Path

    from tests.eval.task_harness import _pass_artifact_root

    base = Path("/eval/case/artifacts")
    # No run_id → base unchanged (one pass: retries/iterations share the tree).
    assert _pass_artifact_root(base, None) == base
    # Distinct run_id → distinct, non-overlapping roots (memory can't leak).
    p0 = _pass_artifact_root(base, 0)
    p1 = _pass_artifact_root(base, 1)
    assert p0 == base / "pass-0" and p1 == base / "pass-1"
    assert p0 != p1 and base not in (p0, p1)
    # String run_id works too.
    assert _pass_artifact_root(base, "a2") == base / "pass-a2"


# ───────────────────────── envelope + write ─────────────────────────


def test_envelope_shape_and_embedded_snapshot():
    run = EvalRun(
        scenario="task",
        unit="exploitability_assessment",
        pass_at=2,
        metric_kind="capture",
        model="lm-studio-qwen3.6",
        prompt_version="v4",
        timestamp="2026-06-01T00:00:00Z",
        fixtures=[FixtureResult("XBEN-032-24", [
            CaseResult("XBEN-032-24", True, 2, 2, detail={"chain": True}),
        ])],
        meta={"agent_variant": "web_exploit"},
    )
    env = run.to_envelope()
    assert env["schema"] == SCHEMA
    assert env["scenario"] == "task"
    assert env["unit"] == "exploitability_assessment"
    assert env["metric_kind"] == "capture"
    assert env["pass_at"] == 2
    assert env["meta"]["agent_variant"] == "web_exploit"
    # embedded derived snapshot
    assert env["headline"]["pass_rate"] == 1.0
    assert env["headline"]["chain_rate"] == 1.0
    assert env["totals"]["cases"] == 1
    # fixtures carry per-case records
    assert env["fixtures"][0]["cases"][0]["id"] == "XBEN-032-24"


def test_eval_sink_groups_by_scenario_unit(tmp_path, monkeypatch):
    import tests.eval.results as R
    monkeypatch.setattr(R, "EVAL_ROOT", tmp_path)

    sink = EvalSink()
    sink.record(scenario="task", unit="oas_build", metric_kind="diff",
                fixture="vulnyapi", case=CaseResult("vulnyapi", True, 1, 1, detail={"f1": 1.0}),
                model="m", pass_at=2, artifacts={"oas_build/result": "openapi: 3.0.0"})
    # per-case persistence happens immediately on record()
    case_dir = tmp_path / "oas_build" / "cases" / "vulnyapi__vulnyapi"
    assert (case_dir / "metrics.json").is_file()
    assert (case_dir / "oas_build_result").read_text() == "openapi: 3.0.0"
    assert json.loads((case_dir / "metrics.json").read_text())["passed"] is True
    sink.record(scenario="task", unit="oas_build", metric_kind="diff",
                fixture="petstore", case=CaseResult("petstore", False, 0, 1, detail={"f1": 0.0}))
    sink.record(scenario="agent", unit="swe_edit_agent", metric_kind="generic",
                fixture="fx", case=CaseResult("fx", True, 1, 1))
    paths = sink.flush()
    assert len(paths) == 2   # two (scenario, unit) groups
    names = sorted(p.parent.name for p in paths)
    assert names == ["oas_build", "swe_edit_agent"]
    oas = json.loads((tmp_path / "oas_build" / "eval_results.json").read_text())
    assert oas["scenario"] == "task" and oas["pass_at"] == 2   # max pass_at wins
    assert len(oas["fixtures"]) == 2 and oas["headline"]["pass_rate"] == 0.5


def test_case_artifact_dir_colocated_with_eval_sink(tmp_path, monkeypatch):
    import tests.eval.results as R
    monkeypatch.setattr(R, "EVAL_ROOT", tmp_path)
    from tests.eval.results import case_artifact_dir
    d = case_artifact_dir("trace_agent", "vulnyapi", "c1")
    assert d == tmp_path / "trace_agent" / "cases" / "vulnyapi__c1" / "artifacts"


def test_write_eval_results_roundtrip(tmp_path):
    run = EvalRun(
        scenario="agent", unit="trace_agent", pass_at=1, metric_kind="diff",
        fixtures=[FixtureResult("vulnyapi", [
            CaseResult("vulnyapi", True, 1, 1, detail={"f1": 0.9}),
        ])],
    )
    path = write_eval_results(run, tmp_path / "trace_run")
    assert path.is_file() and path.name == "eval_results.json"
    data = json.loads(path.read_text())
    assert data["schema"] == SCHEMA and data["scenario"] == "agent"
    assert data["headline"]["mean_f1"] == 0.9
    # timestamp auto-stamped when omitted
    assert data["timestamp"]
