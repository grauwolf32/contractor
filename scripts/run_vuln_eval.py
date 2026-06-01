#!/usr/bin/env python3
"""Run trace-agent vulnerability detection evals and generate an HTML report.

Usage::

    poetry run python scripts/run_vuln_eval.py
    poetry run python scripts/run_vuln_eval.py --fixtures realvuln-pythonssti
    poetry run python scripts/run_vuln_eval.py --output /tmp/vuln-eval --timeout 600
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURES = "realvuln-pythonssti,realvuln-vfapi,realvuln-vampi"
_CWE_RE = re.compile(r"CWE-\d+")


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class FixtureResult:
    slug: str
    prompt_version: str
    model: str
    duration_s: float

    gt_vuln_count: int
    gt_fp_trap_count: int
    gt_cwes: list[str]

    reported_findings: list[dict[str, Any]]

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    f2: float
    matches: list[dict[str, Any]]

    tool_counts: dict[str, int]
    total_tool_calls: int
    tool_errors: int

    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int

    files_read: list[str]
    vuln_tools_used: dict[str, int]
    annotation_tools_used: dict[str, int]
    skills_loaded: list[str]

    per_cwe: dict[str, dict[str, Any]]

    tool_sequence: list[dict[str, str]] = field(default_factory=list)

    gt_cases: list[dict[str, Any]] = field(default_factory=list)
    gt_files: list[str] = field(default_factory=list)
    file_coverage: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_findings_from_artifacts(artifacts: dict[str, str]):
    import yaml as _yaml
    findings = []
    for key, text in artifacts.items():
        if "vulnerability-reports" not in key:
            continue
        try:
            reports = _yaml.safe_load(text) or {}
        except _yaml.YAMLError:
            continue
        if not isinstance(reports, dict):
            continue
        for _name, report in reports.items():
            if not isinstance(report, dict):
                continue
            place = report.get("place", "")
            details = report.get("details", "")
            m = _CWE_RE.search(details)
            findings.append(dict(
                file=place.lstrip("/"),
                cwe=m.group(0) if m else None,
                line=None,
                title=report.get("title"),
                severity=report.get("severity"),
                details=details[:300],
            ))
    return findings


def _extract_findings_from_tool_calls(tool_calls):
    findings = []
    for call in tool_calls:
        if call.name != "report_vulnerability":
            continue
        args = call.args
        details = args.get("details", "")
        m = _CWE_RE.search(details)
        findings.append(dict(
            file=args.get("place", "").lstrip("/"),
            cwe=m.group(0) if m else None,
            line=None,
            title=args.get("title"),
            severity=args.get("severity"),
            details=details[:300],
        ))
    return findings


def _extract_findings(run):
    findings = _extract_findings_from_artifacts(run.agent_run.artifacts)
    if not findings:
        findings = _extract_findings_from_tool_calls(run.agent_run.tool_calls)
    return findings


def _extract_metrics(metrics_events: list[dict]) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    tool_errors = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    llm_calls = 0

    for ev in metrics_events:
        et = str(ev.get("event_type", ""))
        if et == "tool_call":
            tool_counts[ev.get("tool_name", "unknown")] += 1
        elif et == "tool_result":
            if ev.get("result_error"):
                tool_errors += 1
        elif et == "tool_exception":
            tool_errors += 1
        elif et == "llm_usage":
            llm_calls += 1
            usage = ev.get("usage", {})
            if isinstance(usage, dict):
                input_tokens += usage.get("input", 0) or 0
                output_tokens += usage.get("output", 0) or 0
                total_tokens += usage.get("total", 0) or 0

    return dict(
        tool_counts=dict(tool_counts),
        total_tool_calls=sum(tool_counts.values()),
        tool_errors=tool_errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        llm_calls=llm_calls,
    )


def _extract_behavior(tool_calls) -> dict[str, Any]:
    files_read: list[str] = []
    vuln_tools: Counter[str] = Counter()
    annot_tools: Counter[str] = Counter()
    skills: list[str] = []

    vuln_names = {"report_vulnerability", "get_vulnerability", "list_vulnerabilities"}
    annot_names = {"annotate_trace", "annotate_sink", "annotate_validate"}
    read_names = {"read_file", "read_lines"}

    for call in tool_calls:
        if call.name in read_names:
            path = call.args.get("path") or call.args.get("file") or ""
            if path and path not in files_read:
                files_read.append(path)
        elif call.name in vuln_names:
            vuln_tools[call.name] += 1
        elif call.name in annot_names:
            annot_tools[call.name] += 1
        elif call.name == "skills_read":
            name = call.args.get("name", "")
            if name:
                skills.append(name)

    sequence = [
        {"tool": c.name, "args_summary": _summarize_args(c.args)}
        for c in tool_calls
    ]

    return dict(
        files_read=files_read,
        vuln_tools_used=dict(vuln_tools),
        annotation_tools_used=dict(annot_tools),
        skills_loaded=skills,
        tool_sequence=sequence,
    )


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)[:200]


def _per_cwe_breakdown(
    gt: list[dict], matches: list[dict]
) -> dict[str, dict[str, Any]]:
    gt_by_id = {c["id"]: c for c in gt}
    result: dict[str, dict[str, Any]] = {}

    for case in gt:
        if not case.get("is_vulnerable", True):
            continue
        cwe = case["primary_cwe"]
        if cwe not in result:
            result[cwe] = {"expected": 0, "found": 0, "missed_ids": []}
        result[cwe]["expected"] += 1

    for m in matches:
        if m["classification"] == "TP":
            gt_case = gt_by_id.get(m.get("ground_truth_id", ""))
            if gt_case:
                cwe = gt_case["primary_cwe"]
                if cwe in result:
                    result[cwe]["found"] += 1
        elif m["classification"] == "FN":
            gt_case = gt_by_id.get(m.get("ground_truth_id", ""))
            if gt_case and gt_case.get("is_vulnerable", True):
                cwe = gt_case["primary_cwe"]
                if cwe in result:
                    result[cwe]["missed_ids"].append(m["ground_truth_id"])

    return result


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

async def run_eval(
    fixture_slugs: list[str],
    output_dir: Path,
    timeout_s: float,
) -> list[FixtureResult]:
    from dotenv import load_dotenv
    for p in (REPO_ROOT / "cli" / ".env", REPO_ROOT / ".env"):
        if p.exists():
            load_dotenv(p, override=False)

    from contractor.utils import observability
    observability.init()

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm
        model = LiteLlm(model=override, timeout=600)
    else:
        from contractor.utils.settings import DEFAULT_MODEL
        model = DEFAULT_MODEL

    from tests.eval.conftest import _load_fixture
    from tests.eval.vuln_scan_harness import run_vuln_scan
    from tests.eval.scoring import AgentFinding, score_vuln_findings

    scan_prompt = (
        "Scan this codebase for security vulnerabilities. "
        "For each file, identify route handlers and trace the data flow "
        "from user inputs to sinks. Check the per-handler control checklist. "
        "Use `report_vulnerability` for each finding you discover. "
        "Include the CWE ID (e.g., CWE-89), affected file path, function name, "
        "line numbers, and exploitation details in the `details` field. "
        "Also check for cross-cutting issues: hardcoded secrets, weak crypto, "
        "debug mode, missing security headers. "
        "Load reference docs via skills_read before classifying findings."
    )

    results: list[FixtureResult] = []

    for slug in fixture_slugs:
        fixture = _load_fixture(slug)
        gt = fixture.vuln_cases
        if not gt:
            print(f"  [{slug}] skipped — no vuln cases")
            continue

        gt_vulns = sum(1 for c in gt if c.get("is_vulnerable", True))
        gt_fps = sum(1 for c in gt if not c.get("is_vulnerable", True))
        gt_cwes = sorted({c["primary_cwe"] for c in gt if c.get("is_vulnerable", True)})

        print(f"\n{'='*60}")
        print(f"  {slug}: {gt_vulns} vulns + {gt_fps} FP traps, CWEs: {gt_cwes}")
        print(f"{'='*60}")

        t0 = time.monotonic()
        try:
            run = await run_vuln_scan(
                fixture_root=fixture.source_root,
                user_message=scan_prompt,
                model=model,
                agent_kind="trace",
                namespace=f"vuln-eval-{slug}",
                timeout_s=timeout_s,
                with_graph_tools=True,
            )
        except Exception as exc:
            print(f"  [{slug}] ERROR: {exc}")
            continue
        duration = time.monotonic() - t0

        findings_dicts = _extract_findings(run)
        agent_findings = [
            AgentFinding(
                file=f["file"], cwe=f.get("cwe"),
                line=f.get("line"), title=f.get("title"),
                severity=f.get("severity"),
            )
            for f in findings_dicts
        ]
        score = score_vuln_findings(agent_findings, gt)
        matches_dicts = [
            dict(
                classification=m.classification,
                ground_truth_id=m.ground_truth_id,
                finding_file=m.finding_file,
                finding_cwe=m.finding_cwe,
            )
            for m in score.matches
        ]

        metrics = _extract_metrics(run.agent_run.metrics_events)
        behavior = _extract_behavior(run.agent_run.tool_calls)
        per_cwe = _per_cwe_breakdown(gt, matches_dicts)

        gt_files = sorted({c["file"] for c in gt})
        read_normed = {p.lstrip("/") for p in behavior["files_read"]}
        file_cov = {gf: (gf in read_normed or gf.lstrip("/") in read_normed) for gf in gt_files}

        gt_slim = [
            {k: c[k] for k in ("id", "is_vulnerable", "primary_cwe", "file",
                                "function", "severity", "description",
                                "vulnerability_class") if k in c}
            for c in gt
        ]

        fr = FixtureResult(
            slug=slug,
            prompt_version=run.prompt_version,
            model=model.model,
            duration_s=round(duration, 1),
            gt_vuln_count=gt_vulns,
            gt_fp_trap_count=gt_fps,
            gt_cwes=gt_cwes,
            reported_findings=findings_dicts,
            tp=score.tp, fp=score.fp, fn=score.fn, tn=score.tn,
            precision=round(score.precision, 3),
            recall=round(score.recall, 3),
            f1=round(score.f1, 3),
            f2=round(score.f2, 3),
            matches=matches_dicts,
            per_cwe=per_cwe,
            gt_cases=gt_slim,
            gt_files=gt_files,
            file_coverage=file_cov,
            **metrics,
            **behavior,
        )
        results.append(fr)

        print(f"  duration: {duration:.1f}s  findings: {len(findings_dicts)}")
        print(f"  {score.explain()}")
        print(f"  tools: {metrics['total_tool_calls']}  "
              f"tokens: {metrics['total_tokens']}  "
              f"llm_calls: {metrics['llm_calls']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    # ``payload`` is the in-memory shape the HTML report generator consumes.
    payload = {
        "timestamp": timestamp,
        "model": model.model,
        "fixtures": [asdict(r) for r in results],
    }

    # Persist the canonical eval/v1 envelope (scenario=agent, metric_kind=
    # detection — one case per fixture) for analytics-ui.
    from tests.eval.results import CaseResult as EnvCase
    from tests.eval.results import EvalRun as EnvRun
    from tests.eval.results import FixtureResult as EnvFixture
    from tests.eval.results import write_eval_results

    def _cwe_tpfpfn(per_cwe: dict) -> dict:
        out = {}
        for cwe, agg in (per_cwe or {}).items():
            if "tp" in agg or "fp" in agg or "fn" in agg:
                out[cwe] = {k: int(agg.get(k, 0) or 0) for k in ("tp", "fp", "fn")}
            else:
                found, exp = int(agg.get("found", 0) or 0), int(agg.get("expected", 0) or 0)
                out[cwe] = {"tp": min(found, exp), "fp": max(found - exp, 0),
                            "fn": max(exp - found, 0)}
        return out

    fixtures = []
    for r in (asdict(x) for x in results):
        tp, fp, fn, tn = (int(r.get(k, 0) or 0) for k in ("tp", "fp", "fn", "tn"))
        in_tok, out_tok = int(r.get("input_tokens", 0) or 0), int(r.get("output_tokens", 0) or 0)
        case = EnvCase(
            id=r.get("slug"), passed=tp > 0, pass_count=int(tp > 0), attempts=1,
            metrics={"input_tokens": in_tok, "output_tokens": out_tok,
                     "total_tokens": int(r.get("total_tokens", 0) or 0) or (in_tok + out_tok),
                     "total_tool_calls": int(r.get("total_tool_calls", 0) or 0),
                     "llm_calls": int(r.get("llm_calls", 0) or 0),
                     "duration_s": float(r.get("duration_s", 0) or 0),
                     "tool_counts": r.get("tool_counts") or {}},
            detail={"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": r.get("precision"), "recall": r.get("recall"), "f1": r.get("f1"),
                    "prompt_version": r.get("prompt_version"),
                    "per_cwe": _cwe_tpfpfn(r.get("per_cwe") or {}),
                    "reported_findings": r.get("reported_findings") or [],
                    "matches": r.get("matches") or [], "skills_loaded": r.get("skills_loaded") or [],
                    "files_read": r.get("files_read") or [], "gt_cwes": r.get("gt_cwes") or []})
        fixtures.append(EnvFixture(slug=r.get("slug"), cases=[case]))
    eval_run = EnvRun(
        scenario="agent", unit="codereview_agent", pass_at=1, metric_kind="detection",
        model=str(model.model), timestamp=timestamp, fixtures=fixtures,
    )
    results_path = write_eval_results(eval_run, output_dir)
    print(f"\nResults saved to {results_path}")

    from scripts.vuln_eval_report import generate_report
    report_path = output_dir / "report.html"
    generate_report(payload, report_path)
    print(f"Report saved to {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run trace-agent vuln detection evals")
    parser.add_argument(
        "--fixtures", default=DEFAULT_FIXTURES,
        help=f"Comma-separated fixture slugs (default: {DEFAULT_FIXTURES})",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "eval_runs" / "vuln_eval"),
        help="Output directory for results and report",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    slugs = [s.strip() for s in args.fixtures.split(",") if s.strip()]
    asyncio.run(run_eval(slugs, Path(args.output), args.timeout))


if __name__ == "__main__":
    main()
