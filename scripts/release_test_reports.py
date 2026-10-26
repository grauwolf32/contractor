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
import threading
import math
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CLEANUP_GRACE_SECONDS = 5


def gate(name: str) -> dict:
    return runpy.run_path(str(ROOT / "scripts" / name))


def interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _group_alive(process: subprocess.Popen) -> bool:
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def _cleanup_group(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + CLEANUP_GRACE_SECONDS
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=CLEANUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # Give descendants time to join their own resources even if the command
    # leader exited immediately on SIGTERM.
    while _group_alive(process) and time.monotonic() < deadline:
        time.sleep(0.01)
    _signal_group(process, signal.SIGKILL)
    process.wait()


def capture(
    report: Path, command: list[str], *, timeout_seconds: float | None = None,
    max_report_bytes: int | None = None,
) -> int:
    """Capture redacted output with an independent deadline and joined process group."""
    if not command:
        raise ValueError("a test command is required")
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("capture deadline must be positive and finite")
    if max_report_bytes is not None and max_report_bytes < 1:
        raise ValueError("capture report limit must be positive")
    report.parent.mkdir(parents=True, exist_ok=True)
    audit = gate("test-audit-completion-e2e.py")
    redact = audit["redact"]
    with report.open("w", encoding="utf-8") as output:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        ) as process:
            finished, expired = threading.Event(), threading.Event()
            def watchdog() -> None:
                if finished.wait(timeout_seconds):
                    return
                expired.set()
                _signal_group(process, signal.SIGTERM)
                if not finished.wait(CLEANUP_GRACE_SECONDS):
                    _signal_group(process, signal.SIGKILL)
            watcher = None
            if timeout_seconds is not None:
                watcher = threading.Thread(target=watchdog, name="test-command-deadline")
                watcher.start()
            try:
                if process.stdout is None:
                    raise RuntimeError("test command has no output stream")
                size = 0
                while True:
                    # Go splits output into small events. Bound malformed child
                    # output too, even when it never writes a newline.
                    line = process.stdout.readline(65537)
                    if not line:
                        break
                    if len(line) > 65536:
                        raise ValueError("test command output line exceeds its bound")
                    try:
                        event = audit["parse_go_event"](line)
                    except audit["GateError"]:
                        line = redact(line)
                    else:
                        event = {key: value for key, value in event.items() if key in {
                            "Time", "Action", "Package", "Test", "Elapsed", "Output", "FailedBuild",
                        }}
                        if "Output" in event:
                            event["Output"] = redact(event["Output"])
                        line = json.dumps(event) + "\n"
                    size += len(line.encode("utf-8"))
                    if max_report_bytes is not None and size > max_report_bytes:
                        raise ValueError("test command report exceeds its bound")
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                status = process.wait()
                if expired.is_set():
                    raise TimeoutError("test command deadline expired")
                if _group_alive(process):
                    raise RuntimeError("test command left its owned process group running")
                return status
            except BaseException:
                _cleanup_group(process)
                raise
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join()


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
        choices=("run", "verify-runtime", "verify-audit", "verify-findings", "findings-pattern",
                 "verify-iterative-postgres", "verify-iterative-process", "iterative-pattern"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout-seconds", type=float)
    # Keep the child command separate from the report runner options.
    arguments = sys.argv[1:]
    delimiter = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:delimiter])
    command = arguments[delimiter + 1 :]
    if args.action == "iterative-pattern":
        if command or args.report is not None or args.timeout_seconds is not None:
            parser.error("iterative-pattern takes no command or report options")
        selected = gate("test-audit-iterative.py")["matrix"]()["process"]
        print("|".join(sorted({name.split("/", 1)[0] for cases in selected.values() for name in cases})))
        return 0
    if args.action == "findings-pattern":
        print("|".join(gate("test-findings-e2e.py")["PROCESS_TESTS"]))
        return 0
    if args.report is None:
        parser.error("--report is required")
    if args.action == "run":
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            return capture(args.report, command, timeout_seconds=args.timeout_seconds)
        finally:
            signal.signal(signal.SIGTERM, previous)
    if command:
        parser.error("a command is only accepted by run")
    if args.timeout_seconds is not None:
        parser.error("a deadline is only accepted by run")
    if args.action.startswith("verify-iterative-"):
        gate("test-audit-iterative.py")["verify"](args.report, args.action.removeprefix("verify-iterative-"))
    elif args.action == "verify-runtime":
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
