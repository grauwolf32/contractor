"""Dated, never-overwritten eval archive (the data-loss fix).

Each run lands in its own ``eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/``
folder so results are never overwritten; the flat
``eval_runs/<scenario>-<unit>[-<metric_kind>]/`` path is kept as a "latest"
pointer for analytics-ui.
"""
from __future__ import annotations

import json

import tests.eval.results as results
from tests.eval.results import CaseResult, EvalSink


def test_run_slug_and_archive_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(results, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(results, "RUN_STAMP", "0607-120000")
    assert results._run_slug("agent", "trace_agent", "crapi-workshop") == \
        "agent-trace_agent-eval-crapi-workshop"
    assert results._run_slug("task", "oas_build") == "task-oas_build"
    assert results.run_archive_dir("agent", "trace_agent", "crapi-workshop") == \
        tmp_path / "0607-120000" / "agent-trace_agent-eval-crapi-workshop"


def test_case_artifact_dir_under_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(results, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(results, "RUN_STAMP", "0607-120000")
    assert results.case_artifact_dir("trace_agent", "vampi", "login") == \
        tmp_path / "0607-120000" / "agent-trace_agent-eval-vampi" / "cases" / "login" / "artifacts"
    # scenario tagging (task/pipeline)
    assert "task-oas_build-eval-vampi" in str(
        results.case_artifact_dir("oas_build", "vampi", "c1", scenario="task"))


def test_run_stamp_env_override(monkeypatch):
    monkeypatch.setenv("CONTRACTOR_EVAL_RUN_STAMP", "qw3-off run!")
    assert results._compute_run_stamp() == "qw3-off_run_"  # sanitized to [alnum-_]
    monkeypatch.delenv("CONTRACTOR_EVAL_RUN_STAMP", raising=False)
    s = results._compute_run_stamp()  # default mmdd-HHMMSS
    assert len(s) == 11 and s[4] == "-"


def test_flush_writes_latest_and_dated_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(results, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(results, "RUN_STAMP", "0607-A")
    sink = EvalSink()
    sink.record(scenario="agent", unit="trace_agent", metric_kind="diff",
                fixture="crapi-workshop",
                case=CaseResult(id="c1", passed=True, pass_count=1, attempts=1),
                model="m", pass_at=3)
    sink.record(scenario="agent", unit="trace_agent", metric_kind="diff",
                fixture="vulnyapi",
                case=CaseResult(id="c2", passed=False, pass_count=0, attempts=3),
                model="m", pass_at=3)
    sink.flush()

    # latest pointer — combined, both fixtures, at <scenario>-<unit>-<metric_kind>
    latest = tmp_path / "agent-trace_agent-diff" / "eval_results.json"
    assert latest.exists()
    assert {f["slug"] for f in json.loads(latest.read_text())["fixtures"]} == \
        {"crapi-workshop", "vulnyapi"}

    # dated per-fixture archives — one folder each, single-fixture envelopes
    a1 = tmp_path / "0607-A" / "agent-trace_agent-eval-crapi-workshop" / "eval_results.json"
    a2 = tmp_path / "0607-A" / "agent-trace_agent-eval-vulnyapi" / "eval_results.json"
    assert a1.exists() and a2.exists()
    assert [f["slug"] for f in json.loads(a1.read_text())["fixtures"]] == ["crapi-workshop"]
    # per-case metrics persisted under the archive (crash-safe)
    assert (tmp_path / "0607-A" / "agent-trace_agent-eval-crapi-workshop"
            / "cases" / "c1" / "metrics.json").exists()


def test_second_run_does_not_overwrite_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(results, "EVAL_ROOT", tmp_path)

    monkeypatch.setattr(results, "RUN_STAMP", "0607-A")
    s1 = EvalSink()
    s1.record(scenario="agent", unit="trace_agent", metric_kind="diff", fixture="crapi-workshop",
              case=CaseResult(id="c1", passed=True, pass_count=1, attempts=1))
    s1.flush()

    monkeypatch.setattr(results, "RUN_STAMP", "0607-B")
    s2 = EvalSink()
    s2.record(scenario="agent", unit="trace_agent", metric_kind="diff", fixture="crapi-workshop",
              case=CaseResult(id="c1", passed=False, pass_count=0, attempts=1))
    s2.flush()

    a = tmp_path / "0607-A" / "agent-trace_agent-eval-crapi-workshop" / "eval_results.json"
    b = tmp_path / "0607-B" / "agent-trace_agent-eval-crapi-workshop" / "eval_results.json"
    assert a.exists() and b.exists()  # both runs preserved — no overwrite
    assert json.loads(a.read_text())["fixtures"][0]["cases"][0]["passed"] is True
    assert json.loads(b.read_text())["fixtures"][0]["cases"][0]["passed"] is False
