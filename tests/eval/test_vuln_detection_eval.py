"""End-to-end eval for vulnerability detection against benchmark fixtures.

Runs trace_agent with ``enable_vuln_reporting=True`` against each
vuln-benchmark fixture (RealVuln / eyeballvul) and scores the reported
findings against ground-truth using file + CWE + line-proximity matching.

Fixture selection:  any fixture directory under ``tests/eval/fixtures/``
that contains a ``vuln-cases.json`` is auto-discovered via the
``vuln_fixture`` parametrized session fixture.

Environment overrides:
    CONTRACTOR_EVAL_MODEL              — override eval model
    CONTRACTOR_EVAL_TRACE_PROMPT_VERSION — pin a prompt variant
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import yaml

from tests.eval.scoring import AgentFinding, VulnScore, score_vuln_findings
from tests.eval.trace_harness import run_trace_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scan_prompt(fixture_slug: str) -> str:
    return (
        "You are a security auditor. Scan the entire codebase for security "
        "vulnerabilities. For each vulnerability you find, use the "
        "`report_vulnerability` tool to report it. Include:\n"
        "- A short unique `name` (e.g. 'sqli-get-user')\n"
        "- `place_type` = 'file'\n"
        "- `place` = the file path where the vulnerability exists\n"
        "- `title` = a descriptive title\n"
        "- `summary` = one-sentence description\n"
        "- `severity` = one of: info, low, medium, high, critical\n"
        "- `confidence` = one of: low, medium, high\n"
        "- `details` = technical details including CWE ID (e.g. CWE-89), "
        "affected function name, line numbers, and how the vulnerability "
        "could be exploited\n\n"
        "Start by listing the project files to understand the structure, "
        "then systematically review each source file for vulnerabilities. "
        "Be thorough but precise — avoid false positives."
    )


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
            line = _extract_line(details)
            findings.append(AgentFinding(
                file=place.lstrip("/"),
                cwe=cwe,
                line=line,
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
        line = _extract_line(details)
        findings.append(AgentFinding(
            file=place.lstrip("/"),
            cwe=cwe,
            line=line,
            title=args.get("title"),
            severity=args.get("severity"),
        ))
    return findings


import re

_CWE_RE = re.compile(r"CWE-\d+")
_LINE_RE = re.compile(r"(?:line|L|:)\s*(\d+)", re.IGNORECASE)


def _extract_cwe(details: str) -> str | None:
    m = _CWE_RE.search(details)
    return m.group(0) if m else None


def _extract_line(details: str) -> int | None:
    m = _LINE_RE.search(details)
    return int(m.group(1)) if m else None


def _resolve_prompt_version() -> str | None:
    return os.environ.get("CONTRACTOR_EVAL_TRACE_PROMPT_VERSION")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.eval
@pytest.mark.asyncio
async def test_vuln_detection(vuln_fixture, eval_model):
    """Run vulnerability detection against a benchmark fixture."""
    if not vuln_fixture.vuln_cases:
        pytest.skip(f"no vuln cases for {vuln_fixture.slug}")

    gt = vuln_fixture.vuln_cases
    vuln_count = sum(1 for c in gt if c.get("is_vulnerable", True))
    fp_trap_count = sum(1 for c in gt if not c.get("is_vulnerable", True))

    run = await run_trace_agent(
        fixture_root=vuln_fixture.source_root,
        user_message=_build_scan_prompt(vuln_fixture.slug),
        model=eval_model,
        namespace=f"vuln-eval-{vuln_fixture.slug}",
        enable_vuln_reporting=True,
        timeout_s=1800.0,
        prompt_version=_resolve_prompt_version(),
        with_graph_tools=True,
    )

    findings = _extract_findings_from_artifacts(run.agent_run.artifacts)
    if not findings:
        findings = _extract_findings_from_tool_calls(run.agent_run.tool_calls)

    score = score_vuln_findings(findings, gt)

    print(
        f"\n{'=' * 60}\n"
        f"fixture={vuln_fixture.slug}  prompt={run.prompt_version}\n"
        f"gt: {vuln_count} vulns + {fp_trap_count} FP traps\n"
        f"agent reported: {len(findings)} findings\n"
        f"{score.explain()}\n"
        f"tools_used={sorted(set(run.agent_run.tool_names()))}\n"
        f"{'=' * 60}"
    )

    # Baseline thresholds — deliberately lenient for initial calibration.
    # Tighten once baseline numbers are established.
    min_recall = float(os.environ.get("CONTRACTOR_EVAL_VULN_MIN_RECALL", "0.15"))
    min_precision = float(os.environ.get("CONTRACTOR_EVAL_VULN_MIN_PRECISION", "0.10"))

    assert score.recall >= min_recall, (
        f"vuln recall too low for {vuln_fixture.slug}: "
        f"{score.recall:.2f} < {min_recall:.2f}\n{score.explain()}"
    )
    assert score.precision >= min_precision, (
        f"vuln precision too low for {vuln_fixture.slug}: "
        f"{score.precision:.2f} < {min_precision:.2f}\n{score.explain()}"
    )
