"""Unit tests for ``scripts/rebuild_eval_envelope.py`` — re-aggregating a
unit's envelope from persisted per-case ``metrics.json`` files, across BOTH
on-disk layouts:

* legacy flat:   ``eval_runs/<unit>/cases/<case>/metrics.json``
* dated archive: ``eval_runs/<stamp>/<scenario>-<unit>-eval-<fixture>/cases/<case>/metrics.json``
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

import tests.eval.results as results

_SPEC = importlib.util.spec_from_file_location(
    "ree",
    Path(__file__).resolve().parents[3] / "scripts" / "rebuild_eval_envelope.py",
)
ree = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ree)


@pytest.fixture
def eval_root(tmp_path, monkeypatch):
    """Redirect EVAL_ROOT in BOTH modules: the script's copy (discovery) and
    ``tests.eval.results`` (``write_eval_results`` resolves bare names there)."""
    monkeypatch.setattr(ree, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(results, "EVAL_ROOT", tmp_path)
    return tmp_path


def _write_case(case_dir: Path, fixture: str, case_id: str, *, passed: bool,
                attempts: int = 1) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "metrics.json"
    path.write_text(json.dumps({
        "fixture": fixture, "id": case_id, "passed": passed,
        "pass_count": int(passed), "attempts": attempts,
        "metrics": {"total_tokens": 10}, "detail": {"f1": 1.0 if passed else 0.0},
    }))
    return path


def _flat_case(root: Path, unit: str, fixture: str, case_id: str, **kw) -> Path:
    return _write_case(root / unit / "cases" / case_id, fixture, case_id, **kw)


def _archive_case(root: Path, stamp: str, scenario: str, unit: str,
                  fixture: str, case_id: str, **kw) -> Path:
    run_dir = root / stamp / f"{scenario}-{unit}-eval-{fixture}"
    return _write_case(run_dir / "cases" / case_id, fixture, case_id, **kw)


def test_rebuild_legacy_flat_layout(eval_root):
    _flat_case(eval_root, "oas_build", "vulnyapi", "c1", passed=True)
    _flat_case(eval_root, "oas_build", "petstore", "c2", passed=False)

    path = ree.rebuild_unit("oas_build")
    assert path == eval_root / "oas_build" / "eval_results.json"
    env = json.loads(path.read_text())
    assert {f["slug"] for f in env["fixtures"]} == {"petstore", "vulnyapi"}
    assert env["headline"]["total"] == 2 and env["headline"]["passed"] == 1


def test_rebuild_dated_archive_layout(eval_root):
    _archive_case(eval_root, "0607-converge", "agent", "trace_agent",
                  "vulnyapi", "login", passed=True, attempts=3)
    _archive_case(eval_root, "0607-converge", "agent", "trace_agent",
                  "crapi-workshop", "bola", passed=False, attempts=3)

    path = ree.rebuild_unit("trace_agent")
    assert path == eval_root / "trace_agent" / "eval_results.json"
    env = json.loads(path.read_text())
    assert env["scenario"] == "agent"            # recovered from the dir name
    assert env["pass_at"] == 3
    assert {f["slug"] for f in env["fixtures"]} == {"crapi-workshop", "vulnyapi"}


def test_rebuild_merges_both_layouts_latest_wins(eval_root):
    old = _flat_case(eval_root, "trace_agent", "vulnyapi", "login", passed=False)
    new = _archive_case(eval_root, "0608-rerun", "agent", "trace_agent",
                        "vulnyapi", "login", passed=True)
    _flat_case(eval_root, "trace_agent", "spring", "send-message", passed=True)
    # make the archive copy strictly newer than the flat one
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    env = json.loads(ree.rebuild_unit("trace_agent").read_text())
    by_slug = {f["slug"]: f for f in env["fixtures"]}
    assert set(by_slug) == {"spring", "vulnyapi"}
    # the duplicated (fixture, case) was deduped; the newer archive copy won
    assert len(by_slug["vulnyapi"]["cases"]) == 1
    assert by_slug["vulnyapi"]["cases"][0]["passed"] is True


def test_rebuild_reads_run_meta_from_archive_envelope(eval_root):
    _archive_case(eval_root, "0607-A", "task", "oas_build", "vulnyapi", "c1", passed=True)
    run_dir = eval_root / "0607-A" / "task-oas_build-eval-vulnyapi"
    (run_dir / "eval_results.json").write_text(json.dumps({
        "schema": "eval/v1", "scenario": "task", "unit": "oas_build",
        "metric_kind": "detection", "model": "lm-studio-qwen3.6",
        "prompt_version": "v7", "pass_at": 1, "fixtures": [],
    }))

    env = json.loads(ree.rebuild_unit("oas_build").read_text())
    assert env["metric_kind"] == "detection"
    assert env["model"] == "lm-studio-qwen3.6"
    assert env["prompt_version"] == "v7"


def test_rebuild_unit_no_cases_returns_none(eval_root, capsys):
    assert ree.rebuild_unit("nonexistent") is None
    assert "skipped" in capsys.readouterr().out


def test_discover_units_finds_both_layouts(eval_root):
    _flat_case(eval_root, "oas_build", "vulnyapi", "c1", passed=True)
    _archive_case(eval_root, "0607-converge", "agent", "trace_agent",
                  "vulnyapi", "login", passed=True)
    _archive_case(eval_root, "0608-x", "pipeline", "vuln_scan", "vampi", "c9", passed=False)
    # noise: a log file and an archive dir without cases/
    (eval_root / "run.log").write_text("noise")
    (eval_root / "0609-y" / "agent-empty_unit-eval-fx").mkdir(parents=True)

    assert ree.discover_units() == ["oas_build", "trace_agent", "vuln_scan"]
