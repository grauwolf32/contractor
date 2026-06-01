#!/usr/bin/env python3
"""One-shot migration of pre-``eval/v1`` eval results into the standard envelope.

Converts the historical on-disk shapes into ``eval_results.json`` envelopes so
``analytics-ui`` (which reads only ``eval/v1``) can display past runs:

* legacy ``run_vuln_eval`` detection  → scenario=agent, metric_kind=detection
* legacy ``run_exploit_eval`` verdict → scenario=agent, metric_kind=verdict
* ``xbow_<agent>_metrics.jsonl``      → scenario=task,  metric_kind=capture

Legacy ``eval_results.json`` files are rewritten in place. Each XBOW JSONL is
converted into ``eval_runs/<stem>/eval_results.json`` (the JSONL is left alone;
the reader ignores it). Idempotent: files already in ``eval/v1`` are skipped.

    python scripts/migrate_eval_results.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json

from tests.eval.results import (EVAL_ROOT, CaseResult, EvalRun, FixtureResult,
                                write_eval_results)


def _metrics(src: dict) -> dict:
    in_tok = int(src.get("input_tokens", 0) or 0)
    out_tok = int(src.get("output_tokens", 0) or 0)
    tool_counts = dict(src.get("tool_counts") or {})
    # ``total_tool_calls`` may be absent; legacy exploit cases instead carry a
    # ``tool_calls`` LIST of names. Fall back to that, then to tool_counts.
    calls = src.get("total_tool_calls")
    if calls is None:
        tcalls = src.get("tool_calls")
        calls = len(tcalls) if isinstance(tcalls, list) else sum(
            int(v) for v in tool_counts.values() if isinstance(v, (int, float)))
    return {
        "input_tokens": in_tok, "output_tokens": out_tok,
        "total_tokens": int(src.get("total_tokens", 0) or 0) or (in_tok + out_tok),
        "total_tool_calls": int(calls or 0),
        "llm_calls": int(src.get("llm_calls", 0) or 0),
        "http_requests": int(src.get("http_requests", src.get("total_http_requests", 0)) or 0),
        "duration_s": float(src.get("duration_s", 0) or 0),
        "tool_counts": tool_counts,
    }


def _per_cwe_to_tpfpfn(per_cwe: dict) -> dict:
    out = {}
    for cwe, agg in (per_cwe or {}).items():
        if "tp" in agg or "fp" in agg or "fn" in agg:
            out[cwe] = {k: int(agg.get(k, 0) or 0) for k in ("tp", "fp", "fn")}
        else:
            found, expected = int(agg.get("found", 0) or 0), int(agg.get("expected", 0) or 0)
            out[cwe] = {"tp": min(found, expected), "fp": max(found - expected, 0),
                        "fn": max(expected - found, 0)}
    return out


def _from_legacy_vuln(data: dict) -> EvalRun:
    fixtures = []
    for f in data.get("fixtures") or []:
        tp, fp, fn, tn = (int(f.get(k, 0) or 0) for k in ("tp", "fp", "fn", "tn"))
        case = CaseResult(
            id=f.get("slug"), passed=tp > 0, pass_count=int(tp > 0), attempts=1,
            metrics=_metrics(f),
            detail={"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": f.get("precision"), "recall": f.get("recall"), "f1": f.get("f1"),
                    "prompt_version": f.get("prompt_version"),
                    "per_cwe": _per_cwe_to_tpfpfn(f.get("per_cwe") or {}),
                    "reported_findings": f.get("reported_findings") or [],
                    "matches": f.get("matches") or [], "skills_loaded": f.get("skills_loaded") or [],
                    "files_read": f.get("files_read") or [], "gt_cwes": f.get("gt_cwes") or []},
        )
        fixtures.append(FixtureResult(slug=f.get("slug"), cases=[case]))
    return EvalRun(scenario="agent", unit=data.get("unit") or "codereview_agent",
                   pass_at=1, metric_kind="detection", model=data.get("model"),
                   prompt_version=data.get("prompt_version"), timestamp=data.get("timestamp"),
                   fixtures=fixtures)


def _from_legacy_exploit(data: dict) -> EvalRun:
    n = int(data.get("n_runs", 1) or 1)
    fixtures = []
    for f in data.get("fixtures") or []:
        cases = []
        for c in f.get("cases") or []:
            runs = c.get("runs") or []
            pc = sum(1 for r in runs if r.get("passed")) if runs else int(bool(c.get("passed")))
            cases.append(CaseResult(
                id=c.get("id") or c.get("finding_name"), passed=bool(c.get("passed")),
                pass_count=pc, attempts=n, metrics=_metrics(c),
                detail={"finding_name": c.get("finding_name"), "vuln_class": c.get("vuln_class"),
                        "expected_verdict": c.get("expected_verdict"),
                        "actual_verdict": c.get("actual_verdict"),
                        "has_evidence": bool(c.get("has_evidence"))},
            ))
        fixtures.append(FixtureResult(slug=f.get("slug"), cases=cases))
    return EvalRun(scenario="agent", unit=data.get("agent_variant") or "exploitability_agent",
                   pass_at=n, metric_kind="verdict", model=data.get("model"),
                   prompt_version=data.get("prompt_version"), timestamp=data.get("timestamp"),
                   fixtures=fixtures, meta={"agent_variant": data.get("agent_variant")})


def _from_xbow(rows: list[dict], stem: str) -> EvalRun:
    agent = stem[len("xbow_"):-len("_metrics")] if stem.startswith("xbow_") else stem
    fixtures = [
        FixtureResult(slug=r.get("id"), cases=[CaseResult(
            id=r.get("id"), passed=bool(r.get("captured")),
            pass_count=int(bool(r.get("captured"))), attempts=1, metrics=_metrics(r),
            detail={"tags": r.get("tags"), "captured": bool(r.get("captured")),
                    "chain": bool(r.get("chain"))})])
        for r in rows
    ]
    return EvalRun(scenario="task", unit=f"xbow:{agent}", pass_at=1,
                   metric_kind="capture", fixtures=fixtures,
                   meta={"benchmark": "xbow"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    converted = skipped = 0
    for path in sorted(EVAL_ROOT.rglob("eval_results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("schema") == "eval/v1":
            skipped += 1
            continue
        fx = data.get("fixtures") or []
        is_exploit = bool(fx) and isinstance(fx[0], dict) and "cases" in fx[0]
        run = _from_legacy_exploit(data) if is_exploit else _from_legacy_vuln(data)
        print(f"  {'[dry] ' if args.dry_run else ''}{path.relative_to(EVAL_ROOT)} "
              f"→ {run.scenario}/{run.metric_kind}")
        if not args.dry_run:
            # Preserve the original (eval_runs/ is not git-tracked) before
            # overwriting eval_results.json with the envelope.
            backup = path.with_suffix(".legacy.json")
            if not backup.exists():
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            write_eval_results(run, path.parent)
        converted += 1

    for path in sorted(EVAL_ROOT.glob("xbow_*_metrics.jsonl")):
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        run = _from_xbow(rows, path.stem)
        out_dir = EVAL_ROOT / path.stem
        print(f"  {'[dry] ' if args.dry_run else ''}{path.name} → {out_dir.name}/ "
              f"({run.scenario}/{run.metric_kind})")
        if not args.dry_run:
            write_eval_results(run, out_dir)
        converted += 1

    print(f"\nconverted {converted}, skipped {skipped} (already eval/v1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
