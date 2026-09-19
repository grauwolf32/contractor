#!/usr/bin/env python3
"""Mandatory offline Audit completion release gate; no absent or skipped cases."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PATTERN = "AuditCompletion|^TestImporter|^TestStandard"


class GateError(Exception):
    pass


def matrix() -> dict:
    return json.loads((ROOT / "scripts/audit-completion-matrix.json").read_text())


def verify_go_report(path: Path) -> list[str]:
    required = matrix()["go"]
    started, passed, packages = set(), set(), set()
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError as error:
            raise GateError("Go report is not a JSON event stream") from error
        if not isinstance(event, dict):
            raise GateError("Go report contains an invalid event")
        package, action, test = event.get("Package"), event.get("Action"), event.get("Test")
        if action in {"skip", "fail"}:
            raise GateError(f"Go case did not pass: {package}/{test or ''} ({action})")
        if test:
            key = (package, test)
            if action == "run":
                started.add(key)
            elif action == "pass":
                passed.add(key)
        elif action == "pass":
            packages.add(package)
    missing = {
        f"{package}/{name}"
        for package, names in required.items()
        for name in names
        if (package, name) not in started & passed
    }
    missing.update(set(required) - packages)
    if missing:
        raise GateError(
            "mandatory Go cases did not execute and pass: " + ", ".join(sorted(missing))
        )
    return sorted(f"{package}/{name}" for package, name in started & passed)


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


def redact(text: str) -> str:
    dsn = os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "")
    if dsn:
        text = text.replace(dsn, "[test database]").replace(
            json.dumps(dsn)[1:-1], "[test database]"
        )
        secrets = []
        try:
            parsed = urlsplit(dsn)
            if parsed.password:
                secrets.extend((parsed.password, unquote(parsed.password)))
        except ValueError:
            pass
        password = re.search(r"(?:^|\s)password\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))", dsn)
        if password:
            secrets.append(password.group(1) or password.group(2))
        for secret in secrets:
            if not secret:
                continue
            text = text.replace(secret, "[redacted]").replace(
                json.dumps(secret)[1:-1], "[redacted]"
            )
    return text


def run_command(command: list[str], *, cwd: Path, report: Path, go: bool = False) -> None:
    # Capture stderr too: compiler/connection failures must not print a DSN.
    with report.open("w") as saved:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = redact(line)
            saved.write(line)
            if go:
                try:
                    event = json.loads(line)
                    if event.get("Action") == "output":
                        print(event.get("Output", ""), end="", flush=True)
                except (ValueError, AttributeError):
                    print(line, end="", flush=True)
            else:
                print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise GateError(f"test command failed (exit {status}); see {report.name}")


def run(output_dir: Path) -> None:
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
        raise GateError("CONTRACTOR_TEST_DATABASE_URL is required; use disposable PostgreSQL")
    python = ROOT / "runtime/.venv/bin/python"
    if not python.is_file() or shutil.which("go") is None or shutil.which("uv") is None:
        raise GateError("Go, uv and the prepared runtime/.venv/bin/python are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Unique directories prevent old success reports from surviving a failed retry.
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    print(f"Audit completion evidence: {directory}", flush=True)
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
        go=True,
    )
    go_cases = verify_go_report(report)
    (directory / "executed.json").write_text(
        json.dumps({"status": "passed", "go": go_cases, "python": python_cases}, indent=2) + "\n"
    )
    print(
        f"Audit completion gate passed: {len(go_cases)} Go cases, "
        f"{len(python_cases)} Runtime cases; no selected skips."
    )


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
    sys.exit(main())
