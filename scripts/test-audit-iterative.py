#!/usr/bin/env python3
"""Fail-closed real-store/process verification of generic iterative Audits."""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests/e2e/audit_iterative_matrix.yml"
MAX_EVIDENCE_BYTES = 64 << 10
MAX_REPORT_BYTES = 32 << 20
DEADLINES = {"postgres": 1800, "process": 2400}
CASE = re.compile(r"Test[A-Za-z0-9_]+(?:/[^\x00-\x1f]{1,256})*")


GateError = reports.GateError
pattern = reports.pattern


def matrix() -> dict:
    if MATRIX.stat().st_size > 256 << 10:
        raise GateError("iterative matrix exceeds its bound")
    value = json.loads(MATRIX.read_text(), object_pairs_hook=reports.json_object)
    if not isinstance(value, dict) or value.get("schema_version") != "2.0":
        raise GateError("invalid iterative matrix schema")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > 1024:
        raise GateError("empty or oversized iterative matrix")
    selected = {boundary: {} for boundary in ("postgres", "process", "harness")}
    coverage, identities = set(), set()
    for entry in cases:
        if not isinstance(entry, dict):
            raise GateError("invalid iterative matrix case")
        boundary, package, case = (entry.get(key) for key in ("boundary", "package", "case"))
        if not isinstance(boundary, str) or boundary not in selected or not isinstance(package, str) or not package or len(package) > 512 or not isinstance(case, str) or not case or len(case) > 1024:
            raise GateError("invalid iterative matrix identity")
        identity = (boundary, package, case)
        if identity in identities:
            raise GateError("duplicate iterative matrix case")
        identities.add(identity)
        if boundary != "harness" and (not package.startswith("github.com/grauwolf32/contractor/") or CASE.fullmatch(case) is None):
            raise GateError("invalid Go case identity")
        covers = entry.get("covers")
        if not isinstance(covers, list) or any(not isinstance(item, str) or item not in {f"A{i}" for i in range(1, 7)} for item in covers) or len(covers) != len(set(covers)):
            raise GateError("invalid acceptance coverage")
        coverage.update(covers)
        selected[boundary].setdefault(package, []).append(case)
    if any(not cases or len(cases) > 64 for cases in selected.values()) or coverage != {f"A{i}" for i in range(1, 7)}:
        raise GateError("iterative boundaries or acceptance coverage are incomplete")
    for module, names in selected["harness"].items():
        if re.fullmatch(r"test_[a-z0-9_]+", module) is None:
            raise GateError("invalid harness module identity")
        tree = ast.parse((ROOT / "scripts" / (module + ".py")).read_text())
        declared = {node.name + "." + member.name for node in tree.body if isinstance(node, ast.ClassDef)
                    for member in node.body if isinstance(member, ast.FunctionDef) and member.name.startswith("test_")}
        if set(names) - declared:
            raise GateError("harness matrix names an undeclared unit test")
    return selected


def verify(report: Path, boundary: str) -> list[str]:
    if boundary not in DEADLINES:
        raise GateError("unknown verification boundary")
    return reports.verify_go_report(report, matrix()[boundary])


def _prerequisites(boundary: str) -> None:
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
        raise GateError("CONTRACTOR_TEST_DATABASE_URL is required; use disposable PostgreSQL")
    if shutil.which("go") is None:
        raise GateError("Go is required")
    if boundary == "process" and (shutil.which("uv") is None or not (ROOT / "runtime/.venv/bin/python").is_file() or not (ROOT / "runtime/uv.lock").is_file()):
        raise GateError("the locked Runtime and uv are required")


def run(boundary: str, output_dir: Path) -> Path:
    if boundary not in DEADLINES:
        raise GateError("unknown verification boundary")
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=boundary + "-", dir=output_dir))
    evidence = {"schema": "contractor.audit-iterative-gate.v2", "status": "incomplete", "boundary": boundary}
    destination = directory / "result.json"
    try:
        pin = reports.evidence_pin(MATRIX)
        evidence.update(pin)
        _prerequisites(boundary)
        reports.require_fresh(pin, MATRIX)
        selected = matrix()[boundary]
        evidence["environment"] = {"os": platform.system(), "arch": platform.machine(), "database_configured": True, "runtime_required": boundary == "process"}
        reports.write_evidence(destination, evidence)
        package_names = ["./" + package.removeprefix("github.com/grauwolf32/contractor/") for package in selected]
        command = ["go", "test", "-json", "-p=2", "-count=1", "-tags=" + ("integration" if boundary == "postgres" else "e2e"), "-timeout=" + ("25m" if boundary == "postgres" else "35m"), *package_names, "-run", pattern(case for cases in selected.values() for case in cases)]
        if boundary == "postgres":
            command.insert(3, "-race")
        report = directory / "go.jsonl"
        # Use the existing capture owner for redaction, cancellation and join.
        status = reports.capture(report, command, cwd=ROOT, timeout_seconds=DEADLINES[boundary], max_report_bytes=MAX_REPORT_BYTES, retain_output=False)
        if status:
            raise GateError("test command failed")
        reports.require_fresh(pin, MATRIX)
        evidence["cases"] = verify(report, boundary)
        evidence["counts"] = {"required": sum(map(len, selected.values())), "passed": len(evidence["cases"]), "skipped": 0}
        evidence["report_digest"] = reports.digest(report)
        evidence["status"] = "passed"
        reports.write_evidence(destination, evidence)
        return destination
    except BaseException as error:
        evidence.pop("cases", None)
        evidence.pop("counts", None)
        evidence["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "timed_out" if isinstance(error, TimeoutError) else "failed"
        evidence["reason"] = type(error).__name__
        reports.write_evidence(destination, evidence)
        raise
    finally:
        print(f"Iterative Audit evidence: {destination}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=tuple(DEADLINES), required=True)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/audit-iterative-gate")
    args = parser.parse_args()
    if args.verify_only is not None:
        cases = verify(args.verify_only, args.boundary)
        print(f"Iterative Audit {args.boundary}: {len(cases)} mandatory cases passed")
    else:
        run(args.boundary, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    previous = signal.signal(signal.SIGTERM, reports.interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Iterative Audit verification cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(reports.redact(f"Iterative Audit verification failed: {error}"), file=sys.stderr)
        sys.exit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
