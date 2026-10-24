#!/usr/bin/env python3
"""Capture canonical test runs and apply the existing gates to their reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def gate(name: str) -> dict:
    return runpy.run_path(str(ROOT / "scripts" / name))


def interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def capture(report: Path, command: list[str]) -> int:
    """Stream a command's combined output; an interrupted writer joins its process group."""
    if not command:
        raise ValueError("a test command is required")
    report.parent.mkdir(parents=True, exist_ok=True)
    audit = gate("test-audit-completion-e2e.py")
    redact = audit["redact"]
    with report.open("w", encoding="utf-8") as output:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                if process.stdout is None:
                    raise RuntimeError("test command has no output stream")
                for line in process.stdout:
                    try:
                        event = audit["parse_go_event"](line)
                    except audit["GateError"]:
                        line = redact(line)
                    else:
                        # Keep framework identities intact; arbitrary diagnostic
                        # fields must not bypass redaction in captured output.
                        event = {
                            key: value
                            for key, value in event.items()
                            if key
                            in {
                                "Time",
                                "Action",
                                "Package",
                                "Test",
                                "Elapsed",
                                "Output",
                                "FailedBuild",
                            }
                        }
                        if "Output" in event:
                            event["Output"] = redact(event["Output"])
                        line = json.dumps(event) + "\n"
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                return process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise


def verify_runtime(report: Path) -> None:
    """Retain each gate's exact module scope and minimum parametrized case counts."""
    source = ET.parse(report).getroot()
    audit = gate("test-audit-completion-e2e.py")
    findings = gate("test-findings-e2e.py")
    selections = (
        (audit, {name.split("::", 1)[0] for name in audit["matrix"]()["python"]}),
        (
            findings,
            {name.removesuffix(".py").replace("/", ".") for name in findings["PYTHON_FILES"]},
        ),
    )
    with tempfile.TemporaryDirectory(prefix="release-runtime-report-") as directory:
        for index, (module, modules) in enumerate(selections):
            selected = ET.Element("testsuite")
            for case in source.iter("testcase"):
                classname = case.get("classname", "")
                if any(classname == name or classname.startswith(name + ".") for name in modules):
                    selected.append(case)
            path = Path(directory) / f"{index}.xml"
            ET.ElementTree(selected).write(path, encoding="utf-8")
            module["verify_python_report"](path)


def verify_go(report: Path, kind: str) -> None:
    """Filter a canonical stream to the original gate's package and test scope."""
    module = gate(f"test-{kind}-e2e.py")
    if kind == "audit-completion":
        packages = module["matrix"]()["go"]
        pattern = module["PATTERN"]
    else:
        packages = {module["PACKAGE"]}
        pattern = "^(" + "|".join(module["PROCESS_TESTS"]) + ")$"
    with tempfile.TemporaryDirectory(prefix="release-go-report-") as directory:
        path = Path(directory) / "selected.jsonl"
        with path.open("w") as output, report.open() as source:
            for line in source:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Go report must contain JSON objects")
                if event.get("Package") not in packages:
                    continue
                test = event.get("Test")
                if test is None or re.search(pattern, test.split("/", 1)[0]) is not None:
                    output.write(line)
        module["verify_go_report"](path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("run", "verify-runtime", "verify-audit", "verify-findings", "findings-pattern"),
    )
    parser.add_argument("--report", type=Path)
    # Keep the child command separate from the report runner options.
    arguments = sys.argv[1:]
    delimiter = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:delimiter])
    command = arguments[delimiter + 1 :]
    if args.action == "findings-pattern":
        print("|".join(gate("test-findings-e2e.py")["PROCESS_TESTS"]))
        return 0
    if args.report is None:
        parser.error("--report is required")
    if args.action == "run":
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            return capture(args.report, command)
        finally:
            signal.signal(signal.SIGTERM, previous)
    if command:
        parser.error("a command is only accepted by run")
    if args.action == "verify-runtime":
        verify_runtime(args.report)
    else:
        verify_go(args.report, "audit-completion" if args.action == "verify-audit" else "findings")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Release test execution interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Release test report failed: {error}", file=sys.stderr)
        sys.exit(1)
