#!/usr/bin/env python3
"""Execute the mandatory real-store, process and browser composition matrix."""

import argparse
import json
import os
import shutil
import signal
import tempfile
from pathlib import Path

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests/e2e/audit_composition_matrix.yml"

def matrix(case_set: str = "all") -> dict:
    # JSON is a YAML subset; keeping this gate stdlib-only avoids bootstrap deps.
    value = json.loads(MATRIX.read_text())
    workspace = value.get("workspace", {})
    for boundary in ("process",):
        selected = workspace.get(boundary, {})
        names = {name.rsplit("/", 1)[-1] for cases in selected.values() for name in cases}
        if names != {"handoff", "publication", "replay", "cas", "authority"}:
            raise ValueError(f"incomplete workspace {boundary} case set")
    if case_set == "workspace":
        value = workspace
    else:
        for boundary in ("integration", "process"):
            for package, cases in workspace[boundary].items():
                value[boundary].setdefault(package, []).extend(cases)
    for boundary in ("integration", "process"):
        if not value.get(boundary) or any(not cases for cases in value[boundary].values()):
            raise ValueError(f"empty {boundary} matrix")
    return value


pattern = reports.pattern


def verify(report: Path, required: dict[str, list[str]], *, root_pattern=".*") -> int:
    reports.verify_go_report(report, required, root_pattern=root_pattern)
    return sum(len(cases) for cases in required.values())


def execute(boundary: str, required: dict, directory: Path) -> list[str]:
    report = directory / f"{boundary}.jsonl"
    command = ["go", "test", "-json", f"-tags={'integration' if boundary == 'integration' else 'e2e'}",
               "-count=1", "-timeout=24m"]
    command += ["./" + package.removeprefix("github.com/grauwolf32/contractor/") for package in required]
    command += ["-run", pattern(case for cases in required.values() for case in cases)]
    status = reports.capture(report, command, cwd=ROOT, timeout_seconds=1500,
                             max_report_bytes=32 << 20)
    if status:
        raise ValueError(f"{boundary} exited {status}; report {report}")
    executed = reports.verify_go_report(report, required, root_pattern=".*")
    count = sum(map(len, required.values()))
    print(f"Audit composition {boundary}: {count} mandatory cases, zero skips", flush=True)
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=("all", "workspace"), default="all")
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--boundary", choices=("integration", "process"), default="process")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.verify_only, matrix(args.case_set)[args.boundary],
               root_pattern=None if args.case_set == "workspace" else ".*")
        return
    with tempfile.TemporaryDirectory(prefix="audit-composition-") as temporary:
        directory = Path(os.environ.get("CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR", temporary)).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / ("gate-workspace.json" if args.case_set == "workspace" else "gate.json")
        evidence = {"status": "incomplete"}
        try:
            reports.write_evidence(destination, evidence)
            cases = matrix(args.case_set)
            if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
                raise ValueError("CONTRACTOR_TEST_DATABASE_URL is required")
            for binary in (("go",) if args.case_set == "workspace" else ("go", "node", "sqlmap", "nuclei")):
                if not shutil.which(binary):
                    raise ValueError(f"required executable missing: {binary}")
            for relative in (("runtime/.venv/bin/python",) if args.case_set == "workspace" else ("runtime/.venv/bin/python", "ui/node_modules/@playwright/test/cli.js")):
                if not (ROOT / relative).is_file():
                    raise ValueError(f"required dependency missing: {relative}")
            pin = reports.evidence_pin(MATRIX)
            evidence.update(pin)
            reports.write_evidence(destination, evidence)
            reports.require_fresh(pin, MATRIX)
            # The browser rejects absent Chromium, skips and missing production journeys.
            os.environ["CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR"] = str(directory)
            # A failed retry cannot borrow a prior browser success in a caller-owned directory.
            journey_path = directory / "journey.json"
            journey_path.unlink(missing_ok=True)
            executed = {boundary: execute(boundary, cases[boundary], directory) for boundary in ("integration", "process")}
            journey = {} if args.case_set == "workspace" else json.loads(journey_path.read_text())
            if args.case_set != "workspace" and journey.get("passed") is not True:
                raise ValueError("production browser journey absent or incomplete")
            reports.require_fresh(pin, MATRIX)
            evidence.update(status="passed", cases=executed,
                            mandatoryCases={boundary: sum(map(len, cases[boundary].values())) for boundary in executed},
                            caseSet=args.case_set, browserJourneys=0 if args.case_set == "workspace" else 1, skipped=0,
                            reports={path.name: reports.digest(path) for path in
                                     (directory / "integration.jsonl", directory / "process.jsonl", journey_path) if path.is_file()})
            reports.write_evidence(destination, evidence)
        except BaseException as error:
            evidence["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "timed_out" if isinstance(error, TimeoutError) else "failed"
            evidence["reason"] = type(error).__name__
            reports.write_evidence(destination, evidence)
            raise
    boundaries = "real store and processes" if args.case_set == "workspace" else "real store, processes and browser"
    print(f"Audit composition acceptance passed: {boundaries}")


if __name__ == "__main__":
    previous = signal.signal(signal.SIGTERM, reports.interrupt)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, TimeoutError) as error:
        raise SystemExit(reports.redact(f"Audit composition acceptance failed: {error}")) from error
    finally:
        signal.signal(signal.SIGTERM, previous)
