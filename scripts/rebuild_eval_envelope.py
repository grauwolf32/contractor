#!/usr/bin/env python3
"""Rebuild a unit's combined eval/v1 envelope from persisted per-case metrics.

When fixtures of one eval unit are run in *separate* pytest sessions, each
session's ``EvalSink`` flush writes a single-fixture ``eval_results.json`` and
overwrites the previous one — so analytics-ui only sees the last fixture, even
though every case's ``eval_runs/<unit>/cases/<fixture>__<case>/metrics.json``
survives. This re-aggregates all of them into one envelope (no re-run).

Usage:
    python scripts/rebuild_eval_envelope.py <unit> [<unit> ...]
    python scripts/rebuild_eval_envelope.py --all     # every unit under eval_runs/

Prefer running a unit's fixtures in ONE session (one combined envelope, no
rebuild needed). Use this when that wasn't possible (e.g. per-fixture reruns).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from contractor.utils.settings import DEFAULT_MODEL
from tests.eval.results import (
    EVAL_ROOT,
    CaseResult,
    EvalRun,
    FixtureResult,
    write_eval_results,
)


def _case_from_metrics(m: dict) -> CaseResult:
    return CaseResult(
        id=m.get("id", "?"),
        passed=bool(m.get("passed")),
        pass_count=int(m.get("pass_count", int(bool(m.get("passed"))))),
        attempts=int(m.get("attempts", 1)),
        metrics=m.get("metrics", {}),
        detail=m.get("detail", {}),
    )


def rebuild_unit(unit: str) -> Path | None:
    unit_dir = EVAL_ROOT / unit
    case_files = sorted(unit_dir.glob("cases/*/metrics.json"))
    if not case_files:
        print(f"[{unit}] no cases/*/metrics.json under {unit_dir} — skipped")
        return None

    by_fixture: dict[str, list[CaseResult]] = {}
    for cf in case_files:
        m = json.loads(cf.read_text())
        by_fixture.setdefault(m.get("fixture", "?"), []).append(_case_from_metrics(m))

    fixtures = [FixtureResult(slug=s, cases=cs) for s, cs in sorted(by_fixture.items())]
    run = EvalRun(
        scenario="task",          # task-level eval; adjust if reused for agent/pipeline
        unit=unit,
        pass_at=max((c.attempts for f in fixtures for c in f.cases), default=1),
        metric_kind="diff",
        model=str(getattr(DEFAULT_MODEL, "model", DEFAULT_MODEL)),
        prompt_version=None,
        fixtures=fixtures,
        meta={"rebuilt_from": "per-case metrics.json"},
    )
    path = write_eval_results(run, unit)
    n_cases = sum(len(f.cases) for f in fixtures)
    print(f"[{unit}] envelope -> {path}  ({len(fixtures)} fixtures, {n_cases} cases)")
    return path


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args == ["--all"]:
        units = [d.name for d in sorted(EVAL_ROOT.iterdir())
                 if d.is_dir() and (d / "cases").is_dir()]
    else:
        units = args
    for unit in units:
        rebuild_unit(unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
