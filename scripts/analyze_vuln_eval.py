#!/usr/bin/env python3
"""Run both vuln_scan and trace agents on a fixture and dump detailed traces.

Usage:
    poetry run python scripts/analyze_vuln_eval.py realvuln-vampi
    poetry run python scripts/analyze_vuln_eval.py realvuln-vfapi --agent trace
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

import yaml

from tests.eval.conftest import _load_fixture
from tests.eval.scoring import AgentFinding, score_vuln_findings
from tests.eval.vuln_scan_harness import AgentKind, run_vuln_scan

_CWE_RE = re.compile(r"CWE-\d+")


def extract_findings(run) -> list[AgentFinding]:
    findings = []
    for key, text in run.agent_run.artifacts.items():
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
            m = _CWE_RE.search(details)
            cwe = m.group(0) if m else None
            findings.append(AgentFinding(
                file=place.lstrip("/"),
                cwe=cwe,
                line=None,
                title=report.get("title"),
                severity=report.get("severity"),
            ))
    if not findings:
        for call in run.agent_run.tool_calls:
            if call.name != "report_vulnerability":
                continue
            args = call.args
            details = args.get("details", "")
            m = _CWE_RE.search(details)
            cwe = m.group(0) if m else None
            findings.append(AgentFinding(
                file=args.get("place", "").lstrip("/"),
                cwe=cwe,
                line=None,
                title=args.get("title"),
                severity=args.get("severity"),
            ))
    return findings


async def analyze(slug: str, agent_kind: AgentKind):
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[1]
    for p in (repo_root / "cli" / ".env", repo_root / ".env"):
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

    fixture = _load_fixture(slug)
    gt = fixture.vuln_cases

    print(f"\n{'=' * 70}")
    print(f"  Fixture: {slug}  Agent: {agent_kind}")
    print(f"  GT: {sum(1 for c in gt if c.get('is_vulnerable', True))} vulns, "
          f"{sum(1 for c in gt if not c.get('is_vulnerable', True))} FP traps")
    print(f"{'=' * 70}\n")

    run = await run_vuln_scan(
        fixture_root=fixture.source_root,
        user_message=(
            "Scan this codebase for security vulnerabilities. "
            "Use `report_vulnerability` for each finding you discover. "
            "Include the CWE ID, affected file path, function name, "
            "line numbers, and exploitation details in the `details` field."
        ),
        model=model,
        agent_kind=agent_kind,
        namespace=f"analyze-{slug}-{agent_kind}",
        timeout_s=900.0,
        with_graph_tools=True,
    )

    # Tool call sequence
    print("TOOL CALL SEQUENCE:")
    print("-" * 50)
    for i, call in enumerate(run.agent_run.tool_calls, 1):
        args_summary = {}
        for k, v in call.args.items():
            s = str(v)
            args_summary[k] = s[:80] + "..." if len(s) > 80 else s
        print(f"  {i:3d}. {call.name}({json.dumps(args_summary)})")
    print()

    # Tool usage stats
    from collections import Counter
    tool_counts = Counter(c.name for c in run.agent_run.tool_calls)
    print("TOOL USAGE COUNTS:")
    for name, count in tool_counts.most_common():
        print(f"  {name:30s} {count:3d}")
    print()

    # Vulnerability reports
    findings = extract_findings(run)
    print(f"FINDINGS REPORTED ({len(findings)}):")
    print("-" * 50)
    for f in findings:
        print(f"  file={f.file}  cwe={f.cwe}  title={f.title}  sev={f.severity}")
    print()

    # Score
    score = score_vuln_findings(findings, gt)
    print("SCORING:")
    print(score.explain())
    print()

    # Artifacts dumped
    print("ARTIFACTS:")
    for key, text in run.agent_run.artifacts.items():
        print(f"  {key}: {len(text)} chars")
    print()

    # Final agent text (abbreviated)
    final = run.agent_run.final_text
    if final:
        print("FINAL AGENT RESPONSE (first 500 chars):")
        print(final[:500])
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--agent", default="vuln_scan", choices=["vuln_scan", "trace"])
    args = parser.parse_args()
    asyncio.run(analyze(args.slug, args.agent))


if __name__ == "__main__":
    main()
