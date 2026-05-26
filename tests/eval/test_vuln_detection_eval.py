"""End-to-end eval for vulnerability detection against benchmark fixtures.

Runs either ``vuln_scan_agent`` (baseline) or ``trace_agent`` (with vuln
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

import os
import re
from typing import Any

import pytest
import yaml

from tests.eval.scoring import AgentFinding, VulnScore, score_vuln_findings
from tests.eval.vuln_scan_harness import AgentKind, VulnScanRun, run_vuln_scan


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
async def test_vuln_detection(vuln_fixture, eval_model):
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
            )
        except Exception as exc:
            print(
                f"\n  [{vuln_fixture.slug}] attempt {attempt}/{n} "
                f"agent={agent_kind} ERROR: {exc}"
            )
            continue

        findings = _extract_findings(run)
        score = score_vuln_findings(findings, gt)
        attempts.append((run, findings, score))

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

    assert passed, (
        f"pass@{n} failed for {vuln_fixture.slug} (agent={agent_kind})\n"
        f"best score: {best.explain()}\n"
        f"thresholds: precision>={min_p:.2f} recall>={min_r:.2f}"
    )
