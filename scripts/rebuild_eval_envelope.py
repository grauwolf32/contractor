#!/usr/bin/env python3
"""Rebuild a unit's combined eval/v1 envelope from persisted per-case metrics.

When fixtures of one eval unit are run in *separate* pytest sessions, each
session's ``EvalSink`` flush writes a single-fixture ``eval_results.json`` and
overwrites the previous one — so analytics-ui only sees the last fixture, even
though every case's ``metrics.json`` survives. This re-aggregates all of them
into one per-unit envelope (no re-run).

Two on-disk layouts are scanned:

* legacy flat:    ``eval_runs/<unit>/cases/<case>/metrics.json``
* dated archive:  ``eval_runs/<RUN_STAMP>/<scenario>-<unit>-eval-<fixture>/
  cases/<case>/metrics.json`` (what ``EvalSink._persist_case`` writes today)

When the same ``(fixture, case)`` appears in several runs, the most recently
written ``metrics.json`` wins.

Usage:
    python scripts/rebuild_eval_envelope.py <unit> [<unit> ...]
    python scripts/rebuild_eval_envelope.py --all     # every unit under eval_runs/

Prefer running a unit's fixtures in ONE session (one combined envelope, no
rebuild needed). Use this when that wasn't possible (e.g. per-fixture reruns).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from contractor.utils.settings import DEFAULT_MODEL
from tests.eval.results import (
    EVAL_ROOT,
    CaseResult,
    EvalRun,
    FixtureResult,
    _safe_name,
    write_eval_results,
)

_SCENARIOS = ("agent", "task", "pipeline")
# Dated-archive dir name: <scenario>-<unit>-eval-<fixture> (see _run_slug in
# tests/eval/results.py). Non-greedy unit so a fixture containing "-eval-"
# can't swallow part of the unit name.
_ARCHIVE_DIR_RE = re.compile(r"^(agent|task|pipeline)-(?P<unit>.+?)-eval-.+$")


def _case_from_metrics(m: dict) -> CaseResult:
    return CaseResult(
        id=m.get("id", "?"),
        passed=bool(m.get("passed")),
        pass_count=int(m.get("pass_count", int(bool(m.get("passed"))))),
        attempts=int(m.get("attempts", 1)),
        metrics=m.get("metrics", {}),
        detail=m.get("detail", {}),
    )


def _iter_case_files(unit: str) -> Iterator[tuple[str | None, Path | None, Path]]:
    """Yield ``(scenario, run_dir, metrics_path)`` for both layouts.

    ``scenario`` / ``run_dir`` are ``None`` for the legacy flat layout (which
    carries no scenario tag and no sibling envelope).
    """
    # Legacy flat layout: eval_runs/<unit>/cases/<case>/metrics.json
    for cf in sorted((EVAL_ROOT / unit).glob("cases/*/metrics.json")):
        yield None, None, cf
    # Dated archive layout:
    # eval_runs/<stamp>/<scenario>-<unit>-eval-<fixture>/cases/<case>/metrics.json
    safe = _safe_name(unit)
    prefixes = {f"{s}-{safe}-eval-": s for s in _SCENARIOS}
    for stamp_dir in sorted(p for p in EVAL_ROOT.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in stamp_dir.iterdir() if p.is_dir()):
            scenario = next(
                (s for pre, s in prefixes.items() if run_dir.name.startswith(pre)),
                None,
            )
            if scenario is None:
                continue
            for cf in sorted(run_dir.glob("cases/*/metrics.json")):
                yield scenario, run_dir, cf


def _read_run_meta(run_dir: Path) -> dict:
    """Best-effort read of the per-fixture envelope sitting next to ``cases/``
    (carries the run's true metric_kind / model / prompt_version)."""
    env_path = run_dir / "eval_results.json"
    if not env_path.is_file():
        return {}
    try:
        return json.loads(env_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def rebuild_unit(unit: str) -> Path | None:
    # Latest metrics.json wins per (fixture, case_id) — the same case can
    # appear in several dated runs (and in the legacy flat dir).
    latest: dict[tuple[str, str], tuple[float, CaseResult]] = {}
    scenarios: Counter[str] = Counter()
    run_meta: tuple[float, dict] | None = None
    for scenario, run_dir, cf in _iter_case_files(unit):
        m = json.loads(cf.read_text())
        mtime = cf.stat().st_mtime
        key = (m.get("fixture", "?"), m.get("id", "?"))
        if key not in latest or mtime >= latest[key][0]:
            latest[key] = (mtime, _case_from_metrics(m))
        if scenario:
            scenarios[scenario] += 1
        if run_dir is not None and (run_meta is None or mtime >= run_meta[0]):
            meta = _read_run_meta(run_dir)
            if meta:
                run_meta = (mtime, meta)

    if not latest:
        print(f"[{unit}] no cases/*/metrics.json under {EVAL_ROOT} "
              "(flat or dated layout) — skipped")
        return None

    by_fixture: dict[str, list[CaseResult]] = {}
    for (fixture_slug, _case_id), (_mtime, case) in sorted(latest.items()):
        by_fixture.setdefault(fixture_slug, []).append(case)

    fixtures = [FixtureResult(slug=s, cases=cs) for s, cs in sorted(by_fixture.items())]
    meta = run_meta[1] if run_meta else {}
    run = EvalRun(
        scenario=(scenarios.most_common(1)[0][0] if scenarios else "task"),
        unit=unit,
        pass_at=max((c.attempts for f in fixtures for c in f.cases), default=1),
        metric_kind=meta.get("metric_kind") or "diff",
        model=meta.get("model") or str(getattr(DEFAULT_MODEL, "model", DEFAULT_MODEL)),
        prompt_version=meta.get("prompt_version"),
        fixtures=fixtures,
        meta={"rebuilt_from": "per-case metrics.json"},
    )
    path = write_eval_results(run, unit)
    n_cases = sum(len(f.cases) for f in fixtures)
    print(f"[{unit}] envelope -> {path}  ({len(fixtures)} fixtures, {n_cases} cases)")
    return path


def discover_units() -> list[str]:
    """Every unit that has per-case metrics in either layout."""
    units: set[str] = set()
    for d in sorted(p for p in EVAL_ROOT.iterdir() if p.is_dir()):
        if (d / "cases").is_dir():
            units.add(d.name)  # legacy flat: eval_runs/<unit>/cases/
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            m = _ARCHIVE_DIR_RE.match(sub.name)
            if m and (sub / "cases").is_dir():
                units.add(m.group("unit"))
    return sorted(units)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    units = discover_units() if args == ["--all"] else args
    for unit in units:
        rebuild_unit(unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
