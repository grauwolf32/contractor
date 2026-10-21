#!/usr/bin/env python3
"""Execute the mandatory real-store, process and browser composition matrix."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests/e2e/audit_composition_matrix.yml"
PROCESS_TESTS = (
    "TestAuditCompositionAcrossProductionProcesses",
    "TestOpenAPIAuditScanAcrossProductionProcesses",
)


def matrix() -> dict:
    # JSON is a YAML subset; keeping this gate stdlib-only avoids bootstrap deps.
    value = json.loads(MATRIX.read_text())
    for boundary in ("integration", "process"):
        if not value.get(boundary) or any(not cases for cases in value[boundary].values()):
            raise ValueError(f"empty {boundary} matrix")
    return value


def pattern(cases) -> str:
    return "^(" + "|".join(sorted({re.escape(case.split("/")[0]) for case in cases})) + ")$"


def verify(report: Path, required: dict[str, list[str]]) -> int:
    started, passed, packages = set(), set(), set()
    for line in report.read_text().splitlines():
        event = json.loads(line)
        package, case, action = event.get("Package"), event.get("Test"), event.get("Action")
        if package not in required:
            continue
        if action in {"skip", "fail"}:
            raise ValueError(f"mandatory {package}/{case or '<package>'}: {action}")
        if case and action == "run":
            started.add((package, case))
        if action == "pass":
            if case:
                passed.add((package, case))
            else:
                packages.add(package)
    expected = {(package, case) for package, cases in required.items() for case in cases}
    expected |= {(package, case.split("/")[0]) for package, case in expected}
    missing = expected - (started & passed)
    if missing or packages != set(required):
        raise ValueError("mandatory cases/packages absent: " + ", ".join(
            sorted(f"{package}/{case}" for package, case in missing)
            + sorted(set(required) - packages)))
    return sum(len(cases) for cases in required.values())


def execute(boundary: str, required: dict, directory: Path) -> int:
    report = directory / f"{boundary}.jsonl"
    command = ["go", "test", "-json", f"-tags={'integration' if boundary == 'integration' else 'e2e'}",
               "-count=1", "-timeout=24m"]
    command += ["./" + package.removeprefix("github.com/grauwolf32/contractor/") for package in required]
    command += ["-run", pattern(case for cases in required.values() for case in cases)]
    with report.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, check=False, timeout=1500)
    if result.returncode:
        # The URL is a secret-bearing input; never echo it from a failed test.
        text = report.read_text().replace(os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "\0"), "[database]")
        print(text)
        raise ValueError(f"{boundary} exited {result.returncode}; report {report}")
    count = verify(report, required)
    print(f"Audit composition {boundary}: {count} mandatory cases, zero skips", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--boundary", choices=("integration", "process"), default="process")
    args = parser.parse_args()
    cases = matrix()
    if args.verify_only:
        verify(args.verify_only, cases[args.boundary])
        return
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise ValueError("CONTRACTOR_TEST_DATABASE_URL is required")
    for binary in ("go", "node", "sqlmap", "nuclei"):
        if not shutil.which(binary):
            raise ValueError(f"required executable missing: {binary}")
    for relative in ("runtime/.venv/bin/python", "ui/node_modules/@playwright/test/cli.js"):
        if not (ROOT / relative).is_file():
            raise ValueError(f"required dependency missing: {relative}")
    with tempfile.TemporaryDirectory(prefix="audit-composition-") as temporary:
        directory = Path(os.environ.get("CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR", temporary)).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # The process browser itself rejects unavailable Chromium, zero cases,
        # skipped/failed attempts and a missing production journey report.
        os.environ["CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR"] = str(directory)
        counts = {boundary: execute(boundary, cases[boundary], directory) for boundary in ("integration", "process")}
        journey = json.loads((directory / "journey.json").read_text())
        if journey.get("passed") is not True:
            raise ValueError("production browser journey absent or incomplete")
        (directory / "gate.json").write_text(json.dumps({"mandatoryCases": counts, "browserJourneys": 1, "skipped": 0}, indent=2) + "\n")
    print("Audit composition acceptance passed: real store, processes and browser")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Audit composition acceptance failed: {error}") from error
