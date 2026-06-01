"""Read + analyze persisted eval-run results for the explorer UI.

Eval runs drop an ``eval_results.json`` under ``eval_runs/<run>/`` (or the
``eval_runs/`` root). Two shapes exist, distinguished by their fixture records:

* **vuln detection** — fixtures carry ``tp/fp/fn/tn`` + ``precision/recall/f1``
  and ``reported_findings`` / ``matches``.
* **exploitability** — fixtures carry ``cases_total/cases_passed`` and a
  ``cases`` list of per-finding verdicts (expected vs actual + evidence).

This module discovers those files live, infers the kind, and derives the
aggregate analytics the UI charts (micro precision/recall, pass rate, evidence
rate, a verdict matrix, tool-usage and token totals) — so the frontend only
renders, mirroring how the matplotlib report scripts crunch the same JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = _REPO_ROOT / "eval_runs"
_RESULTS_NAME = "eval_results.json"


def _safe_load(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _id_for(path: Path) -> str:
    """Stable, single-segment id from a result file's directory."""
    rel = path.parent.relative_to(EVAL_ROOT).as_posix()
    return "root" if rel == "." else rel.replace("/", "~")


def _path_for(run_id: str) -> Optional[Path]:
    rel = "" if run_id == "root" else run_id.replace("~", "/")
    candidate = (EVAL_ROOT / rel / _RESULTS_NAME) if rel else EVAL_ROOT / _RESULTS_NAME
    candidate = candidate.resolve()
    if EVAL_ROOT.resolve() not in candidate.parents and candidate.parent != EVAL_ROOT.resolve():
        return None
    return candidate if candidate.is_file() else None


def _kind(data: dict[str, Any]) -> str:
    fixtures = data.get("fixtures") or []
    if fixtures and isinstance(fixtures[0], dict) and "cases" in fixtures[0]:
        return "exploit"
    return "vuln"


def _round(x: float, n: int = 3) -> float:
    return round(x, n)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": _round(p), "recall": _round(r), "f1": _round(f1)}


def _add_tools(acc: dict[str, int], counts: Any) -> None:
    if isinstance(counts, dict):
        for k, v in counts.items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + int(v)


# ───────────────────────── vuln analytics ─────────────────────────


def _vuln_summary(data: dict[str, Any]) -> dict[str, Any]:
    fixtures = data.get("fixtures") or []
    tp = fp = fn = tn = 0
    in_tok = out_tok = 0
    dur = 0.0
    tools: dict[str, int] = {}
    rows = []
    cwe: dict[str, dict[str, int]] = {}
    for f in fixtures:
        ftp, ffp, ffn, ftn = (int(f.get(k, 0) or 0) for k in ("tp", "fp", "fn", "tn"))
        tp, fp, fn, tn = tp + ftp, fp + ffp, fn + ffn, tn + ftn
        in_tok += int(f.get("input_tokens", 0) or 0)
        out_tok += int(f.get("output_tokens", 0) or 0)
        dur += float(f.get("duration_s", 0) or 0)
        _add_tools(tools, f.get("tool_counts"))
        for c, agg in (f.get("per_cwe") or {}).items():
            slot = cwe.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                slot[k] += int(agg.get(k, 0) or 0)
        rows.append({
            "slug": f.get("slug"),
            "prompt_version": f.get("prompt_version"),
            "tp": ftp, "fp": ffp, "fn": ffn, "tn": ftn,
            "precision": f.get("precision"), "recall": f.get("recall"), "f1": f.get("f1"),
            "findings": len(f.get("reported_findings") or []),
            "gt_vuln_count": f.get("gt_vuln_count"),
            "total_tokens": int(f.get("total_tokens", 0) or 0),
            "tool_calls": int(f.get("total_tool_calls", 0) or 0),
            "tool_errors": int(f.get("tool_errors", 0) or 0),
            "duration_s": f.get("duration_s"),
        })
    micro = _prf(tp, fp, fn)
    cwe_rows = [{"cwe": c, **v, **_prf(v["tp"], v["fp"], v["fn"])} for c, v in cwe.items()]
    cwe_rows.sort(key=lambda r: (-(r["tp"] + r["fn"]), r["cwe"]))
    return {
        "headline": {"precision": micro["precision"], "recall": micro["recall"], "f1": micro["f1"]},
        "totals": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "input_tokens": in_tok, "output_tokens": out_tok, "duration_s": _round(dur, 1),
                   "fixtures": len(fixtures)},
        "micro": micro,
        "fixtures": rows,
        "per_cwe": cwe_rows,
        "tools": _top_tools(tools),
    }


def _vuln_fixture_detail(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "reported_findings": f.get("reported_findings") or [],
        "matches": f.get("matches") or [],
        "tool_counts": f.get("tool_counts") or {},
        "skills_loaded": f.get("skills_loaded") or [],
        "files_read": f.get("files_read") or [],
        "gt_cwes": f.get("gt_cwes") or [],
    }


# ───────────────────────── exploit analytics ─────────────────────────

_VERDICTS = ["exploitable", "not_exploitable", "inconclusive"]


def _exploit_summary(data: dict[str, Any]) -> dict[str, Any]:
    fixtures = data.get("fixtures") or []
    total = passed = evidence = 0
    in_tok = out_tok = http = 0
    dur = 0.0
    tools: dict[str, int] = {}
    matrix: dict[str, dict[str, int]] = {e: {a: 0 for a in _VERDICTS + ["other"]} for e in _VERDICTS}
    rows = []
    for f in fixtures:
        cases = f.get("cases") or []
        ct = int(f.get("cases_total", len(cases)) or 0)
        cp = int(f.get("cases_passed", sum(1 for c in cases if c.get("passed"))) or 0)
        total, passed = total + ct, passed + cp
        in_tok += int(f.get("input_tokens", 0) or 0)
        out_tok += int(f.get("output_tokens", 0) or 0)
        http += int(f.get("total_http_requests", 0) or 0)
        dur += float(f.get("duration_s", 0) or 0)
        for c in cases:
            if c.get("has_evidence"):
                evidence += 1
            _add_tools(tools, c.get("tool_counts"))
            exp = c.get("expected_verdict")
            act = c.get("actual_verdict")
            if exp in matrix:
                matrix[exp][act if act in _VERDICTS else "other"] += 1
        rows.append({
            "slug": f.get("slug"),
            "cases_total": ct, "cases_passed": cp,
            "pass_rate": _round(cp / ct, 3) if ct else 0.0,
            "evidence": sum(1 for c in cases if c.get("has_evidence")),
            "http_requests": int(f.get("total_http_requests", 0) or 0),
            "total_tokens": int(f.get("input_tokens", 0) or 0) + int(f.get("output_tokens", 0) or 0),
            "tool_calls": int(f.get("total_tool_calls", 0) or 0),
            "duration_s": f.get("duration_s"),
        })
    return {
        "headline": {"pass_rate": _round(passed / total, 3) if total else 0.0,
                     "passed": passed, "total": total,
                     "evidence_rate": _round(evidence / total, 3) if total else 0.0},
        "totals": {"cases": total, "passed": passed, "evidence": evidence,
                   "input_tokens": in_tok, "output_tokens": out_tok, "http_requests": http,
                   "duration_s": _round(dur, 1), "fixtures": len(fixtures)},
        "verdict_matrix": matrix,
        "fixtures": rows,
        "tools": _top_tools(tools),
    }


def _exploit_fixture_detail(f: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for c in f.get("cases") or []:
        cases.append({
            "id": c.get("id"),
            "finding_name": c.get("finding_name"),
            "vuln_class": c.get("vuln_class"),
            "expected_verdict": c.get("expected_verdict"),
            "actual_verdict": c.get("actual_verdict"),
            "passed": bool(c.get("passed")),
            "has_evidence": bool(c.get("has_evidence")),
            "http_requests": c.get("http_requests"),
            "total_tokens": int(c.get("input_tokens", 0) or 0) + int(c.get("output_tokens", 0) or 0),
            "duration_s": c.get("duration_s"),
            "tool_counts": c.get("tool_counts") or {},
        })
    return {"cases": cases}


# ───────────────────────── shared ─────────────────────────


def _top_tools(tools: dict[str, int], n: int = 14) -> list[dict[str, Any]]:
    items = sorted(tools.items(), key=lambda kv: -kv[1])[:n]
    return [{"name": k, "count": v} for k, v in items]


def list_eval_runs() -> list[dict[str, Any]]:
    if not EVAL_ROOT.is_dir():
        return []
    out = []
    for path in sorted(EVAL_ROOT.rglob(_RESULTS_NAME)):
        data = _safe_load(path)
        if data is None:
            continue
        kind = _kind(data)
        summary = _vuln_summary(data) if kind == "vuln" else _exploit_summary(data)
        out.append({
            "id": _id_for(path),
            "name": "root" if path.parent == EVAL_ROOT else path.parent.name,
            "kind": kind,
            "model": data.get("model"),
            "prompt_version": data.get("prompt_version"),
            "timestamp": data.get("timestamp"),
            "fixtures": summary["totals"]["fixtures"],
            "headline": summary["headline"],
        })
    out.sort(key=lambda r: (r["kind"], r["name"]))
    return out


def get_eval_run(run_id: str) -> Optional[dict[str, Any]]:
    path = _path_for(run_id)
    if path is None:
        return None
    data = _safe_load(path)
    if data is None:
        return None
    kind = _kind(data)
    if kind == "vuln":
        summary = _vuln_summary(data)
        detail = {f.get("slug"): _vuln_fixture_detail(f) for f in data.get("fixtures") or []}
    else:
        summary = _exploit_summary(data)
        detail = {f.get("slug"): _exploit_fixture_detail(f) for f in data.get("fixtures") or []}
    return {
        "id": run_id,
        "name": "root" if path.parent == EVAL_ROOT else path.parent.name,
        "kind": kind,
        "model": data.get("model"),
        "prompt_version": data.get("prompt_version"),
        "timestamp": data.get("timestamp"),
        "summary": summary,
        "detail": detail,
    }
