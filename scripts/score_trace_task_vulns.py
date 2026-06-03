#!/usr/bin/env python3
"""Score the vulns reported by a finished ``trace_annotation`` task run.

The task eval (``run_trace_task_eval.py``) scores annotation placement; this
scores the *vulnerabilities* the task surfaced — unioning ``report_vulnerability``
artifacts and the Shape A/B/C blocks in each path's result text, normalizing to
general AppSec families, and matching against each fixture's
``vulnerabilities.expected.json``. Writes a detection ``eval/v1`` envelope so the
A/B (v7 vs shannon) can be compared on *vulns found*, not just annotations.

Usage::

    poetry run python scripts/score_trace_task_vulns.py --run eval_runs/trace-task-v7
    poetry run python scripts/score_trace_task_vulns.py --run eval_runs/trace-task-shannon \
        --output eval_runs/trace-task-shannon/vuln_detection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_ROOT = REPO_ROOT / "tests" / "eval" / "fixtures"


def score_run(run_dir: Path, output_dir: Path) -> None:
    from tests.eval.results import (
        CaseResult,
        EvalRun,
        FixtureResult,
        write_eval_results,
    )
    from tests.eval.trace_vuln_scoring import (
        extract_from_run_dir,
        load_expected,
        score_vulns,
    )

    # Reuse the annotation envelope's metadata (model / prompt) when present.
    model = prompt_version = None
    ann_env = run_dir / "eval_results.json"
    if ann_env.is_file():
        d = json.loads(ann_env.read_text())
        model, prompt_version = d.get("model"), d.get("prompt_version")

    fixtures = []
    print(f"\n{'='*64}\n  vuln-detection scoring — {run_dir.name}\n{'='*64}")
    for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        slug = case_dir.name
        art = case_dir / "artifacts"
        gt = FIXTURES_ROOT / slug / "vulnerabilities.expected.json"
        if not art.is_dir() or not gt.is_file():
            continue
        reported = extract_from_run_dir(art)
        expected = load_expected(gt)
        s = score_vulns(reported, expected)

        # Per-fixture metrics (token/tool aggregates) from the run's metrics.json.
        metrics: dict = {}
        mjson = case_dir / "metrics.json"
        if mjson.is_file():
            m = json.loads(mjson.read_text())
            metrics = {k: m.get(k) for k in (
                "total_tokens", "input_tokens", "output_tokens",
                "total_tool_calls", "llm_calls", "tool_errors", "duration_s",
                "tool_counts") if k in m}

        passed = s.tp > 0 and s.f1 >= 0.3
        case = CaseResult(
            id=slug, passed=passed, pass_count=int(passed), attempts=1,
            metrics=metrics,
            detail={"tp": s.tp, "fp": s.fp, "fn": s.fn,
                    "precision": s.precision, "recall": s.recall, "f1": s.f1,
                    "per_cwe": s.per_family,            # analytics-ui per-class panel
                    "per_family": s.per_family,
                    "reported_findings": [{"family": r.family, "path": r.path,
                                           "title": r.title, "source": r.source}
                                          for r in reported],
                    "matches": s.matched, "missed": s.missed, "extra": s.extra,
                    "prompt_version": prompt_version})
        fixtures.append(FixtureResult(slug=slug, cases=[case]))
        print(f"  {slug:12s} reported={len(reported):3d} expected={len(expected):3d} | {s.explain()}")

    run = EvalRun(
        scenario="task", unit="trace_annotation", pass_at=1, metric_kind="detection",
        model=model, prompt_version=prompt_version, fixtures=fixtures,
        meta={"scored_from": str(run_dir), "scoring": "vuln-family-vs-expected"},
    )
    path = write_eval_results(run, output_dir)
    micro_tp = sum(f.cases[0].detail["tp"] for f in fixtures)
    micro_fp = sum(f.cases[0].detail["fp"] for f in fixtures)
    micro_fn = sum(f.cases[0].detail["fn"] for f in fixtures)
    P = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    R = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    print(f"\n  MICRO: TP={micro_tp} FP={micro_fp} FN={micro_fn} "
          f"P={P:.3f} R={R:.3f} F1={F1:.3f}")
    print(f"  envelope → {path}")


def main():
    ap = argparse.ArgumentParser(description="Score vulns reported by a trace_annotation task run")
    ap.add_argument("--run", required=True, help="run dir (e.g. eval_runs/trace-task-v7)")
    ap.add_argument("--output", default=None,
                    help="envelope dir (default: <run>/vuln_detection)")
    args = ap.parse_args()
    run_dir = Path(args.run)
    output_dir = Path(args.output) if args.output else run_dir / "vuln_detection"
    score_run(run_dir, output_dir)


if __name__ == "__main__":
    main()
