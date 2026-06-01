"""Unit tests for the analytics-ui eval reader (``analytics_ui/evals.py``).

Verifies it ingests the ``eval/v1`` envelope for every scenario/metric_kind and
produces the summary the frontend consumes (headline, domain panels, fixture
rows, detail).
"""
from __future__ import annotations

import pytest

from analytics_ui import evals
from tests.eval.results import (CaseResult, EvalRun, FixtureResult,
                                write_eval_results)


@pytest.fixture
def eval_root(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "EVAL_ROOT", tmp_path)
    return tmp_path


def _write(eval_root, name, run):
    write_eval_results(run, eval_root / name)


def test_detection_run_summary_and_panel(eval_root):
    _write(eval_root, "vuln", EvalRun(
        scenario="agent", unit="codereview_agent", pass_at=1, metric_kind="detection",
        fixtures=[FixtureResult("crapi", [CaseResult(
            "crapi", True, 1, 1,
            metrics={"total_tokens": 100, "total_tool_calls": 5, "tool_counts": {"grep": 5}},
            detail={"tp": 3, "fp": 1, "fn": 1, "tn": 0, "f1": 0.75,
                    "precision": 0.75, "recall": 0.75,
                    "per_cwe": {"CWE-89": {"tp": 2, "fp": 0, "fn": 0}},
                    "reported_findings": [{"file": "a.py"}]})])]))
    run = evals.get_eval_run("vuln")
    assert run["scenario"] == "agent" and run["metric_kind"] == "detection"
    assert run["summary"]["headline"]["f1"] == 0.75
    assert run["summary"]["per_cwe"][0]["cwe"] == "CWE-89"
    row = run["summary"]["fixtures"][0]
    assert row["tp"] == 3 and row["findings"] == 1
    assert "grep" in {t["name"] for t in run["summary"]["tools"]}


def test_verdict_run_matrix_and_cases(eval_root):
    _write(eval_root, "exploit", EvalRun(
        scenario="agent", unit="exploitability_agent", pass_at=3, metric_kind="verdict",
        fixtures=[FixtureResult("vulnyapi", [
            CaseResult("f1", True, 2, 3, metrics={"http_requests": 4, "total_tokens": 50},
                       detail={"expected_verdict": "exploitable", "actual_verdict": "exploitable",
                               "has_evidence": True}),
            CaseResult("f2", False, 0, 3, metrics={"http_requests": 1, "total_tokens": 20},
                       detail={"expected_verdict": "not_exploitable", "actual_verdict": "exploitable",
                               "has_evidence": False}),
        ])]))
    run = evals.get_eval_run("exploit")
    assert run["pass_at"] == 3
    m = run["summary"]["verdict_matrix"]
    assert m["exploitable"]["exploitable"] == 1
    assert m["not_exploitable"]["exploitable"] == 1   # mis-verdict off-diagonal
    assert run["summary"]["headline"]["evidence_rate"] == 0.5
    cases = run["detail"]["vulnyapi"]["cases"]
    assert cases[0]["expected_verdict"] == "exploitable" and cases[0]["pass_count"] == 2


def test_capture_run_chain_rate(eval_root):
    _write(eval_root, "xbow", EvalRun(
        scenario="task", unit="xbow:web_exploit", pass_at=1, metric_kind="capture",
        fixtures=[
            FixtureResult("XBEN-032", [CaseResult("XBEN-032", True, 1, 1,
                metrics={"total_tokens": 2_000_000, "total_tool_calls": 150},
                detail={"tags": "xxe", "captured": True, "chain": True})]),
            FixtureResult("XBEN-006", [CaseResult("XBEN-006", False, 0, 1,
                metrics={"total_tokens": 500_000},
                detail={"tags": "sqli", "captured": False, "chain": False})]),
        ]))
    run = evals.get_eval_run("xbow")
    assert run["scenario"] == "task" and run["metric_kind"] == "capture"
    assert run["summary"]["headline"]["pass_rate"] == 0.5
    assert run["summary"]["headline"]["chain_rate"] == 0.5
    case = run["detail"]["XBEN-032"]["cases"][0]
    assert case["captured"] is True and case["chain"] is True and case["tags"] == "xxe"


def test_list_sorted_by_scenario(eval_root):
    _write(eval_root, "p", EvalRun(scenario="pipeline", unit="trace", pass_at=1,
                                   metric_kind="diff", fixtures=[]))
    _write(eval_root, "a", EvalRun(scenario="agent", unit="x", pass_at=1,
                                   metric_kind="verdict", fixtures=[]))
    _write(eval_root, "t", EvalRun(scenario="task", unit="y", pass_at=1,
                                   metric_kind="capture", fixtures=[]))
    rows = evals.list_eval_runs()
    assert [r["scenario"] for r in rows] == ["agent", "task", "pipeline"]


def test_non_envelope_files_are_ignored(eval_root):
    (eval_root / "legacy").mkdir()
    (eval_root / "legacy" / "eval_results.json").write_text('{"fixtures": [{"slug": "x"}]}')
    assert evals.list_eval_runs() == []   # no schema → skipped, no crash
