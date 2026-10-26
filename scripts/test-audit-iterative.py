#!/usr/bin/env python3
"""Fail-closed real-store/process verification of generic iterative Audits."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests/e2e/audit_iterative_matrix.yml"
MAX_EVIDENCE_BYTES = 64 << 10
MAX_REPORT_BYTES = 32 << 20
DEADLINES = {"postgres": 1800, "process": 2400}
CASE = re.compile(r"Test[A-Za-z0-9_]+(?:/[^\x00-\x1f]{1,256})*")


class GateError(ValueError):
    pass


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise GateError("duplicate JSON member")
        value[key] = item
    return value


def matrix() -> dict:
    if MATRIX.stat().st_size > 256 << 10:
        raise GateError("iterative matrix exceeds its bound")
    value = json.loads(MATRIX.read_text(), object_pairs_hook=_object)
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise GateError("invalid iterative matrix schema")
    for boundary in ("postgres", "process", "harness"):
        selected = value.get(boundary)
        if not isinstance(selected, dict) or not selected or len(selected) > 64:
            raise GateError("empty or oversized iterative matrix boundary")
        for package, cases in selected.items():
            if not isinstance(package, str) or not package or not isinstance(cases, list) or not cases:
                raise GateError("invalid iterative matrix selection")
            if len(cases) > 1024 or any(not isinstance(case, str) or not case or len(case) > 1024 for case in cases) or len(set(cases)) != len(cases):
                raise GateError("invalid iterative matrix cases")
            if boundary != "harness" and (not package.startswith("github.com/grauwolf32/contractor/") or any(CASE.fullmatch(case) is None for case in cases)):
                raise GateError("invalid Go case identity")
    for module, names in value["harness"].items():
        if re.fullmatch(r"test_[a-z0-9_]+", module) is None:
            raise GateError("invalid harness module identity")
        tree = ast.parse((ROOT / "scripts" / (module + ".py")).read_text())
        declared = {node.name + "." + member.name for node in tree.body if isinstance(node, ast.ClassDef)
                    for member in node.body if isinstance(member, ast.FunctionDef) and member.name.startswith("test_")}
        if set(names) - declared:
            raise GateError("harness matrix names an undeclared unit test")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {f"A{i}" for i in range(1, 7)}:
        raise GateError("acceptance coverage is incomplete")
    for records in coverage.values():
        if not isinstance(records, list) or not records:
            raise GateError("acceptance coverage is empty")
        for record in records:
            if not isinstance(record, dict) or record.get("boundary") not in {"postgres", "process", "harness"}:
                raise GateError("invalid acceptance coverage boundary")
            if record.get("case") not in value[record["boundary"]].get(record.get("package"), []):
                raise GateError("acceptance coverage names an unselected case")
    return value


def pattern(cases) -> str:
    names = sorted({case.split("/", 1)[0] for case in cases})
    if not names:
        raise GateError("empty test selection")
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def verify(report: Path, boundary: str) -> list[str]:
    if boundary not in DEADLINES:
        raise GateError("unknown verification boundary")
    required = matrix()[boundary]
    expected = {(package, case) for package, cases in required.items() for case in cases}
    ancestors = {(package, "/".join(case.split("/")[:length])) for package, case in expected for length in range(1, len(case.split("/")) + 1)}
    roots = {package: {case.split("/", 1)[0] for case in cases} for package, cases in required.items()}
    started, passed, packages = set(), set(), set()
    with report.open(encoding="utf-8") as source:
        while True:
            line = source.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise GateError("Go event exceeds its line bound")
            event = json.loads(line, object_pairs_hook=_object)
            if not isinstance(event, dict) or not isinstance(event.get("Action"), str):
                raise GateError("invalid Go JSON event")
            for field in ("Package", "Test", "Output"):
                if field in event and not isinstance(event[field], str):
                    raise GateError("invalid Go JSON event field")
            package, case, action = event.get("Package"), event.get("Test"), event["Action"]
            if package not in required or case is not None and case.split("/", 1)[0] not in roots[package]:
                continue
            if action in {"skip", "fail", "build-fail"}:
                raise GateError("a selected Go case or package did not pass")
            if case:
                key = (package, case)
                if len(started) >= 8192 and key not in started:
                    raise GateError("selected test identities exceed their bound")
                if action == "run":
                    if key in started:
                        raise GateError("selected case ran more than once")
                    started.add(key)
                elif action == "pass":
                    if key not in started or key in passed:
                        raise GateError("selected case pass has no unique prior run")
                    passed.add(key)
            elif action == "pass":
                if package in packages or any(key not in passed for key in ancestors if key[0] == package):
                    raise GateError("package passed without complete selected execution")
                packages.add(package)
    if not expected or ancestors - passed or set(required) - packages:
        raise GateError("mandatory cases or package passes are missing")
    return sorted(f"{package}/{case}" for package, case in expected)


def _write(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=True) + "\n"
    if len(encoded.encode()) > MAX_EVIDENCE_BYTES:
        raise GateError("evidence exceeds its bound")
    pending = path.with_suffix(".pending")
    pending.write_text(encoded)
    pending.replace(path)


def _source() -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise GateError("source revision is unavailable")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, capture_output=True, timeout=5)
    if dirty.returncode != 0:
        raise GateError("source state is unavailable")
    return {"revision": revision, "dirty": bool(dirty.stdout)}


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
    evidence = {"schema": "contractor.audit-iterative-gate.v1", "status": "incomplete", "boundary": boundary}
    destination = directory / "result.json"
    try:
        evidence["source"] = _source()
        _prerequisites(boundary)
        selected = matrix()[boundary]
        evidence["matrix_digest"] = "sha256:" + hashlib.sha256(MATRIX.read_bytes()).hexdigest()
        evidence["environment"] = {"os": platform.system(), "arch": platform.machine(), "database_configured": True, "runtime_required": boundary == "process"}
        _write(destination, evidence)
        package_names = ["./" + package.removeprefix("github.com/grauwolf32/contractor/") for package in selected]
        command = ["go", "test", "-json", "-p=2", "-count=1", "-tags=" + ("integration" if boundary == "postgres" else "e2e"), "-timeout=" + ("25m" if boundary == "postgres" else "35m"), *package_names, "-run", pattern(case for cases in selected.values() for case in cases)]
        if boundary == "postgres":
            command.insert(3, "-race")
        report = directory / "go.jsonl"
        # Use the existing capture owner for redaction, cancellation and join.
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            status = reports.capture(report, command, timeout_seconds=DEADLINES[boundary], max_report_bytes=MAX_REPORT_BYTES)
        finally:
            os.chdir(previous)
        if status:
            raise GateError("test command failed")
        if _source() != evidence["source"]:
            raise GateError("source revision or dirty state changed during verification")
        if "sha256:" + hashlib.sha256(MATRIX.read_bytes()).hexdigest() != evidence["matrix_digest"]:
            raise GateError("verification matrix changed during execution")
        evidence["cases"] = verify(report, boundary)
        evidence["counts"] = {"required": len(evidence["cases"]), "passed": len(evidence["cases"]), "skipped": 0}
        evidence["report_digest"] = "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest()
        evidence["status"] = "passed"
        _write(destination, evidence)
        return destination
    except BaseException as error:
        evidence.pop("cases", None)
        evidence.pop("counts", None)
        evidence["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "timed_out" if isinstance(error, TimeoutError) else "failed"
        evidence["reason"] = type(error).__name__
        _write(destination, evidence)
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
        redact = reports.gate("test-audit-completion-e2e.py")["redact"]
        print(redact(f"Iterative Audit verification failed: {error}"), file=sys.stderr)
        sys.exit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
