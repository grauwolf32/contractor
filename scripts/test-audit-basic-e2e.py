#!/usr/bin/env python3
"""Real common Audit process proof; reject missing, skipped or stale cases."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts/audit-basic-matrix.json"


def matrix(case_set="core"):
    return json.loads(MATRIX.read_text())["programs" if case_set=="programs" else "go"]


def verify(path, case_set="core"):
    return reports.verify_go_report(path, matrix(case_set), root_pattern=".*")


def run(output,case_set="core"):
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
        raise reports.GateError("CONTRACTOR_TEST_DATABASE_URL is required")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    print(f"Common Audit process evidence: {directory}", flush=True)
    evidence = {"status": "incomplete", "case_set":case_set, **reports.evidence_pin(MATRIX)}
    destination = directory / "executed.json"
    reports.write_evidence(destination, evidence, max_bytes=128 << 10)
    try:
        selected = matrix(case_set)
        roots = sorted({name.split("/", 1)[0] for names in selected.values() for name in names})
        pattern = "^(" + "|".join(re.escape(name) for name in roots) + ")$"
        packages = ["./" + name.removeprefix("github.com/grauwolf32/contractor/") for name in selected]
        report = directory / "go.jsonl"
        status = reports.capture(
            report,
            ["go", "test", "-json", "-tags=e2e,integration", "-count=1", "-timeout=12m",
             *packages, "-run", pattern],
            cwd=ROOT, timeout_seconds=900, max_report_bytes=32 << 20, display_events=False,
        )
        if status:
            raise reports.GateError(f"common Audit process command failed (exit {status}); see {report}")
        cases = verify(report,case_set)
        reports.require_fresh(evidence, MATRIX)
        evidence.update(status="passed", go=cases, reports={"go.jsonl": reports.digest(report)})
        print(f"Common Audit process gate passed: {len(cases)} cases; no selected skips.")
    except BaseException as error:
        evidence.update(status="failed", reason=type(error).__name__)
        raise
    finally:
        reports.write_evidence(destination, evidence, max_bytes=128 << 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--case-set",choices=("core","programs"),default="core")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/audit-basic-gate")
    args = parser.parse_args()
    try:
        if args.verify_report:
            verify(args.verify_report,args.case_set)
        else:
            run(args.output_dir.resolve(),args.case_set)
    except (reports.GateError, OSError, ValueError, TimeoutError) as error:
        print(reports.redact(f"Common Audit gate failed: {error}"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
