#!/usr/bin/env python3
"""Mandatory offline Audit completion release gate; no absent or skipped cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts/audit-completion-matrix.json"


GateError = reports.GateError
redact = reports.redact


def matrix() -> dict:
    return json.loads(MATRIX.read_text())


PATTERN = reports.pattern(name for cases in matrix()["go"].values() for name in cases)


def verify_go_report(path: Path) -> list[str]:
    return reports.verify_go_report(path, matrix()["go"], root_pattern=".*")


def verify_python_report(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise GateError("pytest did not produce a valid JUnit report") from error
    counts: Counter[str] = Counter()
    executed = []
    for case in root.iter("testcase"):
        name = case.get("classname", "") + "::" + case.get("name", "")
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise GateError("Python case did not pass: " + name)
        counts[name.split("[", 1)[0]] += 1
        executed.append(name)
    missing = [name for name, minimum in matrix()["python"].items() if counts[name] < minimum]
    if missing:
        raise GateError("mandatory Python cases did not execute and pass: " + ", ".join(missing))
    return sorted(executed)


def run_command(command: list[str], *, cwd: Path, report: Path) -> None:
    status = reports.capture(report, command, cwd=cwd, timeout_seconds=900, max_report_bytes=32 << 20, display_events=False)
    if status:
        raise GateError(f"test command failed (exit {status}); see {report.name}")


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Unique directories prevent old success reports from surviving a failed retry.
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    print(f"Audit completion evidence: {directory}", flush=True)
    destination = directory / "executed.json"
    evidence = {"status": "incomplete"}
    try:
        reports.write_evidence(destination, evidence, max_bytes=128 << 10)
        if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
            raise GateError("CONTRACTOR_TEST_DATABASE_URL is required; use disposable PostgreSQL")
        python = ROOT / "runtime/.venv/bin/python"
        if not python.is_file() or shutil.which("go") is None or shutil.which("uv") is None:
            raise GateError("Go, uv and the prepared runtime/.venv/bin/python are required")
        pin = reports.evidence_pin(MATRIX)
        evidence.update(pin)
        reports.write_evidence(destination, evidence, max_bytes=128 << 10)
        reports.require_fresh(pin, MATRIX)
        junit = directory / "runtime.xml"
        files = sorted(
            {name.split("::", 1)[0].replace(".", "/") + ".py" for name in matrix()["python"]}
        )
        run_command(
            ["uv", "run", "--frozen", "python", "-m", "pytest", *files, "--junitxml", str(junit)],
            cwd=ROOT / "runtime",
            report=directory / "runtime.log",
        )
        python_cases = verify_python_report(junit)
        report = directory / "go.jsonl"
        packages = [
            "./" + name.removeprefix("github.com/grauwolf32/contractor/") for name in matrix()["go"]
        ]
        run_command(
            [
                "go",
                "test",
                "-json",
                "-race",
                "-tags=integration",
                "-count=1",
                "-timeout=8m",
                *packages,
                "-run",
                PATTERN,
            ],
            cwd=ROOT,
            report=report,
        )
        go_cases = verify_go_report(report)
        reports.require_fresh(pin, MATRIX)
        evidence.update(status="passed", go=go_cases, python=python_cases,
                        reports={"go.jsonl": reports.digest(report), "runtime.xml": reports.digest(junit)})
        reports.write_evidence(destination, evidence, max_bytes=128 << 10)
        print(
            f"Audit completion gate passed: {len(go_cases)} Go cases, "
            f"{len(python_cases)} Runtime cases; no selected skips."
        )
    except BaseException as error:
        evidence["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "timed_out" if isinstance(error, TimeoutError) else "failed"
        evidence["reason"] = type(error).__name__
        reports.write_evidence(destination, evidence, max_bytes=128 << 10)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-go-report", type=Path)
    parser.add_argument("--verify-python-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/audit-completion-gate")
    args = parser.parse_args()
    try:
        if args.verify_go_report is not None:
            verify_go_report(args.verify_go_report)
        elif args.verify_python_report is not None:
            verify_python_report(args.verify_python_report)
        else:
            run(args.output_dir.resolve())
    except (GateError, OSError, ValueError) as error:
        print(redact(f"Audit completion gate failed: {error}"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    previous = signal.signal(signal.SIGTERM, reports.interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Audit completion verification cancelled", file=sys.stderr)
        sys.exit(130)
    finally:
        signal.signal(signal.SIGTERM, previous)
