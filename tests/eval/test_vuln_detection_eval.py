"""End-to-end eval for vulnerability detection against benchmark fixtures.

Runs either ``codereview_agent`` (baseline) or ``trace_agent`` (with vuln
reporting enabled) against each vuln-benchmark fixture and scores the
reported findings against ground-truth.

pass@N:  each fixture is attempted up to N times (default 3).  The test
passes if *any* attempt meets the precision/recall thresholds.  All
attempts are logged so the variance is visible.

Fixture selection:  any fixture directory under ``tests/eval/fixtures/``
that contains a ``vuln-cases.json`` is auto-discovered via the
``vuln_fixture`` parametrized session fixture.

Environment overrides:
    CONTRACTOR_EVAL_MODEL                  — override eval model
    CONTRACTOR_EVAL_VULN_AGENT             — "vuln_scan" (default) or "trace"
    CONTRACTOR_EVAL_VULN_PROMPT_VERSION    — pin a prompt variant
    CONTRACTOR_EVAL_VULN_PASS_AT           — N for pass@N (default 3)
    CONTRACTOR_EVAL_VULN_MIN_RECALL        — recall threshold (default 0.15)
    CONTRACTOR_EVAL_VULN_MIN_PRECISION     — precision threshold (default 0.10)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_events
from tests.eval.scoring import (
    AgentFinding,
    VulnScore,
    dedupe_findings,
    score_vuln_findings,
)
from tests.eval.vuln_scan_harness import (
    UNIT_FOR_KIND,
    AgentKind,
    VulnScanRun,
    run_vuln_scan,
)

# Where per-run records (reported findings + score) are written so a run is
# inspectable after the fact (Langfuse only captures the live trace). Lands
# under the gitignored eval_runs/ tree; override with CONTRACTOR_EVAL_RESULTS_DIR.
_RESULTS_DIR = Path(
    os.environ.get("CONTRACTOR_EVAL_RESULTS_DIR")
    or Path(__file__).resolve().parents[2] / "eval_runs" / "vuln_detection"
)


def _dump_record(
    *, slug: str, agent_kind: str, prompt_version: str | None,
    findings: list[AgentFinding], gt: list[dict[str, Any]], score: VulnScore,
) -> None:
    """Persist the reported findings + score for one (fixture, agent) run.

    Without this the only durable evidence of a run is the score summary in
    the pytest log — the actual finding titles/files (needed to diagnose
    why a class is missed) are lost. Best-effort: never fail the test.
    """
    try:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "slug": slug,
            "agent": agent_kind,
            "prompt_version": prompt_version,
            "score": {
                "tp": score.tp, "fp": score.fp, "fn": score.fn, "tn": score.tn,
                "precision": round(score.precision, 3),
                "recall": round(score.recall, 3),
                "f1": round(score.f1, 3),
            },
            "ground_truth": [
                {"id": c.get("id"), "file": c.get("file"),
                 "primary_cwe": c.get("primary_cwe")} for c in gt
            ],
            "reported": [
                {"file": f.file, "cwe": f.cwe, "line": f.line,
                 "title": f.title, "severity": f.severity} for f in findings
            ],
        }
        out = _RESULTS_DIR / f"{slug}__{agent_kind}__{prompt_version or 'active'}.json"
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

def _agent_kind() -> AgentKind:
    return os.environ.get("CONTRACTOR_EVAL_VULN_AGENT", "vuln_scan")  # type: ignore[return-value]


def _pass_at_n() -> int:
    return int(os.environ.get("CONTRACTOR_EVAL_VULN_PASS_AT", "3"))


def _prompt_version() -> str | None:
    return os.environ.get("CONTRACTOR_EVAL_VULN_PROMPT_VERSION")


def _min_recall() -> float:
    return float(os.environ.get("CONTRACTOR_EVAL_VULN_MIN_RECALL", "0.15"))


def _min_precision() -> float:
    return float(os.environ.get("CONTRACTOR_EVAL_VULN_MIN_PRECISION", "0.10"))


def _vuln_dedup_on() -> bool:
    """Whether deterministic finding dedup/merge (QW7/K) is enabled.

    Gated by ``CONTRACTOR_VULN_DEDUP`` — default OFF reproduces the current
    scoring exactly. Truthy values: ``1``, ``true``, ``yes``, ``on``.
    """
    return os.environ.get("CONTRACTOR_VULN_DEDUP", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Finding extraction
# ---------------------------------------------------------------------------

_CWE_RE = re.compile(r"CWE-\d+")


def _extract_cwe(details: str) -> str | None:
    m = _CWE_RE.search(details)
    return m.group(0) if m else None


def _extract_findings_from_artifacts(
    artifacts: dict[str, str],
) -> list[AgentFinding]:
    """Parse VulnerabilityReport artifacts into AgentFinding objects."""
    findings: list[AgentFinding] = []
    for key, text in artifacts.items():
        if "vulnerability-reports" not in key:
            continue
        try:
            reports = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(reports, dict):
            continue
        for _name, report in reports.items():
            if not isinstance(report, dict):
                continue
            place = report.get("place", "")
            details = report.get("details", "")
            cwe = _extract_cwe(details)
            findings.append(AgentFinding(
                file=place.lstrip("/"),
                cwe=cwe,
                line=None,
                title=report.get("title"),
                severity=report.get("severity"),
            ))
    return findings


def _extract_findings_from_tool_calls(
    tool_calls: list[Any],
) -> list[AgentFinding]:
    """Fallback: extract findings directly from report_vulnerability calls."""
    findings: list[AgentFinding] = []
    for call in tool_calls:
        if call.name != "report_vulnerability":
            continue
        args = call.args
        place = args.get("place", "")
        details = args.get("details", "")
        cwe = _extract_cwe(details)
        findings.append(AgentFinding(
            file=place.lstrip("/"),
            cwe=cwe,
            line=None,
            title=args.get("title"),
            severity=args.get("severity"),
        ))
    return findings


def _extract_findings(run: VulnScanRun) -> list[AgentFinding]:
    findings = _extract_findings_from_artifacts(run.agent_run.artifacts)
    if not findings:
        findings = _extract_findings_from_tool_calls(run.agent_run.tool_calls)
    return findings


# ---------------------------------------------------------------------------
# Scan prompt
# ---------------------------------------------------------------------------

def _build_scan_prompt() -> str:
    return (
        "Scan this codebase for security vulnerabilities. "
        "Use `report_vulnerability` for each finding you discover. "
        "Include the CWE ID, affected file path, function name, "
        "line numbers, and exploitation details in the `details` field."
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_vuln_detection(vuln_fixture, eval_model, eval_sink):
    """Run vulnerability detection with pass@N scoring."""
    if not vuln_fixture.vuln_cases:
        pytest.skip(f"no vuln cases for {vuln_fixture.slug}")

    gt = vuln_fixture.vuln_cases
    vuln_count = sum(1 for c in gt if c.get("is_vulnerable", True))
    fp_trap_count = sum(1 for c in gt if not c.get("is_vulnerable", True))
    agent_kind = _agent_kind()
    n = _pass_at_n()
    min_p = _min_precision()
    min_r = _min_recall()

    attempts: list[tuple[VulnScanRun, list[AgentFinding], VulnScore]] = []
    best_score: VulnScore | None = None
    passed = False

    for attempt in range(1, n + 1):
        try:
            run = await run_vuln_scan(
                fixture_root=vuln_fixture.source_root,
                user_message=_build_scan_prompt(),
                model=eval_model,
                agent_kind=agent_kind,
                namespace=f"vuln-eval-{vuln_fixture.slug}-{attempt}",
                timeout_s=1800.0,
                prompt_version=_prompt_version(),
                with_graph_tools=True,
                artifact_dir=case_artifact_dir(
                    UNIT_FOR_KIND.get(agent_kind, agent_kind),
                    vuln_fixture.slug, f"{vuln_fixture.slug}-a{attempt}"),
            )
        except Exception as exc:
            print(
                f"\n  [{vuln_fixture.slug}] attempt {attempt}/{n} "
                f"agent={agent_kind} ERROR: {exc}"
            )
            continue

        findings = _extract_findings(run)
        if _vuln_dedup_on():
            before = len(findings)
            findings = dedupe_findings(findings)
            if len(findings) < before:
                print(
                    f"\n  [{vuln_fixture.slug}] attempt {attempt}/{n} "
                    f"vuln-dedup merged {before - len(findings)} near-duplicate "
                    f"finding(s): {before} -> {len(findings)}"
                )
        score = score_vuln_findings(findings, gt)
        attempts.append((run, findings, score))
        _dump_record(
            slug=vuln_fixture.slug, agent_kind=agent_kind,
            prompt_version=run.prompt_version, findings=findings, gt=gt, score=score,
        )

        print(
            f"\n  [{vuln_fixture.slug}] attempt {attempt}/{n} "
            f"agent={agent_kind} prompt={run.prompt_version}\n"
            f"  gt: {vuln_count} vulns + {fp_trap_count} FP traps  "
            f"reported: {len(findings)}\n"
            f"  {score.explain()}\n"
            f"  tools={sorted(set(run.agent_run.tool_names()))}"
        )

        if best_score is None or score.f1 > best_score.f1:
            best_score = score

        if score.passes(min_precision=min_p, min_recall=min_r):
            passed = True
            if attempt < n:
                print(f"  -> PASS (stopping early at attempt {attempt})")
            break

    if not attempts:
        pytest.fail(f"all {n} attempts errored for {vuln_fixture.slug}")

    # Summary across attempts
    all_scores = [s for _, _, s in attempts]
    avg_recall = sum(s.recall for s in all_scores) / len(all_scores)
    avg_precision = sum(s.precision for s in all_scores) / len(all_scores)
    best = max(all_scores, key=lambda s: s.f1)

    print(
        f"\n{'=' * 60}\n"
        f"  {vuln_fixture.slug}  agent={agent_kind}  "
        f"pass@{len(attempts)}\n"
        f"  avg: precision={avg_precision:.2f} recall={avg_recall:.2f}\n"
        f"  best: precision={best.precision:.2f} recall={best.recall:.2f} "
        f"f1={best.f1:.2f} f2={best.f2:.2f}\n"
        f"  best: TP={best.tp} FP={best.fp} FN={best.fn} TN={best.tn}\n"
        f"{'=' * 60}"
    )

    best_run, best_findings, _ = max(attempts, key=lambda a: a[2].f1)
    pass_count = sum(1 for s in all_scores if s.passes(min_precision=min_p, min_recall=min_r))
    eval_sink.record(
        scenario="agent", unit=UNIT_FOR_KIND.get(agent_kind, agent_kind),
        metric_kind="detection", fixture=vuln_fixture.slug, pass_at=n,
        model=str(eval_model.model), prompt_version=best_run.prompt_version,
        case=CaseResult(
            id=vuln_fixture.slug, passed=passed, pass_count=pass_count, attempts=len(attempts),
            metrics=metrics_from_events(best_run.agent_run.metrics_events),
            detail={"tp": best.tp, "fp": best.fp, "fn": best.fn, "tn": best.tn,
                    "precision": round(best.precision, 3), "recall": round(best.recall, 3),
                    "f1": round(best.f1, 3), "prompt_version": best_run.prompt_version,
                    "reported_findings": [{"file": getattr(f, "file", None),
                                           "cwe": getattr(f, "cwe", None),
                                           "title": getattr(f, "title", None)} for f in best_findings],
                    "matches": [{"classification": m.classification,
                                 "ground_truth_id": m.ground_truth_id,
                                 "finding_file": m.finding_file, "finding_cwe": m.finding_cwe}
                                for m in best.matches]}),
        artifacts=best_run.agent_run.artifacts,
    )
    assert passed, (
        f"pass@{n} failed for {vuln_fixture.slug} (agent={agent_kind})\n"
        f"best score: {best.explain()}\n"
        f"thresholds: precision>={min_p:.2f} recall>={min_r:.2f}"
    )
