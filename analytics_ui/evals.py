"""Read + analyze persisted eval-run results for the explorer UI.

Every eval converges on the ``eval/v1`` envelope (see ``tests/eval/results.py``):
a single ``eval_results.json`` per run, tagged with **scenario** (agent / task /
pipeline), a **metric_kind** (detection / verdict / capture / diff / generic), a
**pass_at** repetition count, and a ``fixtures`` list of per-case records. The
envelope embeds a derived ``headline`` + ``totals`` snapshot.

The UI renders one common skeleton (scenario badge, pass@X headline, per-fixture
table, tool/token totals) plus a small domain panel keyed by ``metric_kind``.
This module discovers the envelopes, trusts their embedded headline/totals, and
derives only the presentational panels (per-CWE, verdict matrix) and per-fixture
rows from the cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = _REPO_ROOT / "eval_runs"
_RESULTS_NAME = "eval_results.json"
SCHEMA = "eval/v1"
SCENARIOS = ("agent", "task", "pipeline")


# ───────────────────────── discovery ─────────────────────────


def _safe_load(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data


def _run_id(path: Path) -> str:
    """Stable single-segment id from an ``eval_results.json`` directory."""
    rel = path.parent.relative_to(EVAL_ROOT).as_posix()
    return "root" if rel == "." else rel.replace("/", "~")


def _run_name(path: Path) -> str:
    return "root" if path.parent == EVAL_ROOT else path.parent.name


def _discover() -> list[tuple[str, Path]]:
    if not EVAL_ROOT.is_dir():
        return []
    return [(_run_id(p), p) for p in sorted(EVAL_ROOT.rglob(_RESULTS_NAME))]


def _resolve(run_id: str) -> Optional[Path]:
    for rid, path in _discover():
        if rid == run_id:
            return path
    return None


# ───────────────────────── derivation (panels + fixture rows) ─────────────────────────


def _round(x: float, n: int = 3) -> float:
    return round(x, n)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": _round(p), "recall": _round(r), "f1": _round(f1)}


def _iter_cases(fixtures: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for f in fixtures:
        for c in f.get("cases") or []:
            yield c


def _top_tools(tools: dict[str, Any], n: int = 14) -> list[dict[str, Any]]:
    items = sorted(((k, int(v)) for k, v in (tools or {}).items()
                    if isinstance(v, (int, float))), key=lambda kv: -kv[1])[:n]
    return [{"name": k, "count": v} for k, v in items]


def _per_cwe(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[str, dict[str, int]] = {}
    for c in _iter_cases(fixtures):
        for cwe, agg in ((c.get("detail") or {}).get("per_cwe") or {}).items():
            slot = acc.setdefault(cwe, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                slot[k] += int(agg.get(k, 0) or 0)
    rows = [{"cwe": c, **v, **_prf(v["tp"], v["fp"], v["fn"])} for c, v in acc.items()]
    rows.sort(key=lambda r: (-(r["tp"] + r["fn"]), r["cwe"]))
    return rows


_VERDICTS = ["exploitable", "not_exploitable", "inconclusive"]


def _verdict_matrix(fixtures: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {e: {a: 0 for a in _VERDICTS + ["other"]} for e in _VERDICTS}
    for c in _iter_cases(fixtures):
        d = c.get("detail") or {}
        exp, act = d.get("expected_verdict"), d.get("actual_verdict")
        if exp in matrix:
            matrix[exp][act if act in _VERDICTS else "other"] += 1
    return matrix


def _fixture_rows(metric_kind: str, fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for f in fixtures:
        cases = f.get("cases") or []
        def _sum(key: str) -> float:
            return sum(float((c.get("metrics") or {}).get(key, 0) or 0) for c in cases)
        skill_reads = sum(int(((c.get("metrics") or {}).get("tool_counts") or {}).get("skills_read", 0) or 0)
                          for c in cases)
        passed = sum(1 for c in cases if c.get("passed"))
        row: dict[str, Any] = {
            "slug": f.get("slug"),
            "cases_total": len(cases), "cases_passed": passed,
            "pass_rate": _round(passed / len(cases), 3) if cases else 0.0,
            "total_tokens": int(_sum("total_tokens")),
            "input_tokens": int(_sum("input_tokens")), "output_tokens": int(_sum("output_tokens")),
            "tool_calls": int(_sum("total_tool_calls")), "tool_errors": int(_sum("tool_errors")),
            "skill_reads": skill_reads,
            "http_requests": int(_sum("http_requests")), "duration_s": _round(_sum("duration_s"), 1),
        }
        if metric_kind == "detection" and cases:
            d = cases[0].get("detail") or {}
            row.update({"tp": d.get("tp"), "fp": d.get("fp"), "fn": d.get("fn"),
                        "f1": d.get("f1"), "precision": d.get("precision"),
                        "recall": d.get("recall"), "prompt_version": d.get("prompt_version"),
                        "findings": len(d.get("reported_findings") or [])})
        elif metric_kind == "verdict":
            row["evidence"] = sum(1 for c in cases if (c.get("detail") or {}).get("has_evidence"))
        rows.append(row)
    return rows


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    mk = data.get("metric_kind", "generic")
    fixtures = data.get("fixtures") or []
    rows = _fixture_rows(mk, fixtures)
    totals = dict(data.get("totals") or {})
    totals["tool_errors"] = sum(r["tool_errors"] for r in rows)
    totals["skill_reads"] = sum(r["skill_reads"] for r in rows)
    # skill names come from detection's skills_loaded; other kinds expose the
    # skills_read *count* via tool_counts (names not recorded).
    skill_names = sorted({
        s for f in fixtures for c in (f.get("cases") or [])
        for s in ((c.get("detail") or {}).get("skills_loaded") or [])
    })
    summary: dict[str, Any] = {
        "headline": data.get("headline") or {},
        "totals": totals,
        "fixtures": rows,
        # skills_read is promoted to its own "Skill usage" panel below; exclude
        # it here so it isn't double-displayed in "Tool calls by tool".
        "tools": _top_tools({k: v for k, v in totals.get("tool_counts", {}).items()
                             if k != "skills_read"}),
        "skills": {
            "names": skill_names,
            "reads": totals["skill_reads"],
            "used_fixtures": sum(1 for r in rows if r["skill_reads"] > 0),
            "total_fixtures": len(rows),
        },
    }
    if mk == "detection":
        summary["per_cwe"] = _per_cwe(fixtures)
    elif mk == "verdict":
        summary["verdict_matrix"] = _verdict_matrix(fixtures)
    return summary


def _detail(data: dict[str, Any]) -> dict[str, Any]:
    mk = data.get("metric_kind", "generic")
    out: dict[str, Any] = {}
    for f in data.get("fixtures") or []:
        slug = f.get("slug")
        cases = f.get("cases") or []
        if mk == "detection" and cases:
            d = cases[0].get("detail") or {}
            out[slug] = {
                "reported_findings": d.get("reported_findings") or [],
                "matches": d.get("matches") or [],
                "skills_loaded": d.get("skills_loaded") or [],
                "files_read": d.get("files_read") or [],
                "gt_cwes": d.get("gt_cwes") or [],
                "tool_counts": (cases[0].get("metrics") or {}).get("tool_counts") or {},
            }
        else:
            out[slug] = {"cases": [{
                "id": c.get("id"), "passed": bool(c.get("passed")),
                "pass_count": c.get("pass_count"), "attempts": c.get("attempts"),
                **{k: (c.get("metrics") or {}).get(k) for k in
                   ("total_tokens", "http_requests", "duration_s")},
                "tool_counts": (c.get("metrics") or {}).get("tool_counts") or {},
                **(c.get("detail") or {}),
            } for c in cases]}
    return out


# ───────────────────────── public API ─────────────────────────


def _row(run_id: str, data: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": _run_name(path),
        "scenario": data.get("scenario", "agent"),
        "metric_kind": data.get("metric_kind", "generic"),
        "unit": data.get("unit"),
        "pass_at": data.get("pass_at", 1),
        "model": data.get("model"),
        "prompt_version": data.get("prompt_version"),
        "timestamp": data.get("timestamp"),
        "fixtures": len(data.get("fixtures") or []),
        "headline": data.get("headline") or {},
    }


def list_eval_runs() -> list[dict[str, Any]]:
    out = []
    for run_id, path in _discover():
        data = _safe_load(path)
        if data is not None:
            out.append(_row(run_id, data, path))
    out.sort(key=lambda r: (SCENARIOS.index(r["scenario"]) if r["scenario"] in SCENARIOS else 9,
                            r["metric_kind"], r["name"]))
    return out


def get_eval_run(run_id: str) -> Optional[dict[str, Any]]:
    path = _resolve(run_id)
    if path is None:
        return None
    data = _safe_load(path)
    if data is None:
        return None
    row = _row(run_id, data, path)
    row.update({"summary": _summary(data), "detail": _detail(data)})
    del row["fixtures"], row["headline"]
    return row
