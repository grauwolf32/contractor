#!/usr/bin/env python3
"""Capture canonical test runs and apply the existing gates to their reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import disk_heavy_policy

ROOT = Path(__file__).resolve().parents[1]
CLEANUP_GRACE_SECONDS = 5
TRACE_DELIVERY_CASES = frozenset(
    f"test_worker_budget_failure_exports_before_normal_or_forced_exit[{fault}-{operation}]"
    for fault in ("none", "worker", "http", "tool", "execution")
    for operation in ("finalize", "abort", "lease", "reconcile")
)


class GateError(ValueError):
    pass


def json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise GateError("duplicate JSON member")
        value[key] = item
    return value


def parse_go_event(line: str) -> dict:
    try:
        event = json.loads(line, object_pairs_hook=json_object)
    except ValueError as error:
        raise GateError("Go report is not a JSON event stream") from error
    if not isinstance(event, dict) or not isinstance(event.get("Action"), str):
        raise GateError("Go report contains an invalid event")
    for field in ("Package", "Test", "Output"):
        if field in event and not isinstance(event[field], str):
            raise GateError("Go report contains an invalid event field")
    return event


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


def pattern(cases) -> str:
    names = sorted({case.split("/", 1)[0] for case in cases})
    if not names:
        raise GateError("empty test selection")
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def verify_go_report(report: Path, required: dict[str, list[str]], *, root_pattern: str | None = None) -> list[str]:
    """Require complete execution, retaining failures in the family's actual root scope."""
    if not required or any(not cases for cases in required.values()):
        raise GateError("empty test selection")
    expected = {(package, case) for package, cases in required.items() for case in cases}
    mandatory = expected | {(package, case.split("/", 1)[0]) for package, case in expected}
    roots = {package: {case.split("/", 1)[0] for case in cases} for package, cases in required.items()}
    started, passed, packages, skipped = set(), set(), set(), set()
    with report.open(encoding="utf-8") as source:
        while True:
            line = source.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise GateError("Go event exceeds its line bound")
            event = parse_go_event(line)
            package, case, action = event.get("Package"), event.get("Test"), event["Action"]
            if package not in required:
                continue
            if case:
                root = case.split("/", 1)[0]
                selected = re.search(root_pattern, root) is not None if root_pattern is not None else root in roots[package]
                if not selected:
                    continue
            if action == "skip" and case and case == root and disk_heavy_policy.optional(package, case):
                key = (package, case)
                if package in packages or key not in started or key in passed | skipped or any(
                    child[0] == package and child[1].startswith(root + "/") for child in started
                ):
                    raise GateError("optional skip is duplicated or outside its root lifetime")
                skipped.add(key)
                continue
            if action in {"skip", "fail", "build-fail"}:
                raise GateError("a selected Go case or package did not pass")
            if case:
                if package in packages:
                    raise GateError("focused test event follows its terminal package")
                key = (package, case)
                if len(started) >= 8192 and key not in started:
                    raise GateError("selected test identities exceed their bound")
                if action == "run":
                    # T.Run names may themselves contain slashes. Go JSON cannot
                    # distinguish those literal names from intermediate parents;
                    # only the top-level test is a guaranteed ancestor.
                    parent = (package, root)
                    if key in started or case != root and (parent not in started or parent in passed | skipped):
                        raise GateError("selected case run is duplicated or outside its root lifetime")
                    started.add(key)
                elif action == "pass":
                    descendants = {child for child in mandatory | started
                                   if child[0] == package and child[1].startswith(root + "/")} if case == root else set()
                    if key not in started or key in passed or descendants - passed:
                        raise GateError("selected case pass is duplicated, premature or has no prior run")
                    passed.add(key)
            elif action == "pass":
                if package in packages or any(
                    key not in passed and (package, key[1].split("/", 1)[0]) not in skipped
                    for key in mandatory | started if key[0] == package
                ):
                    raise GateError("package passed without complete selected execution")
                packages.add(package)
    missing = {key for key in mandatory - passed if (key[0], key[1].split("/", 1)[0]) not in skipped}
    if not expected or missing or started != passed | skipped or set(required) - packages:
        raise GateError("mandatory cases or package passes are missing")
    if skipped:
        print(f"Optional disk-heavy cases skipped: {len(skipped)} (not passing coverage)")
    return sorted(f"{package}/{case}" for package, case in passed)


def disk_heavy_skips(report: Path) -> list[str]:
    """Actual optional root skips, for truthful retained execution evidence."""
    skipped = []
    with report.open(encoding="utf-8") as source:
        while line := source.readline(65537):
            if len(line) > 65536:
                raise GateError("Go event exceeds its line bound")
            event = parse_go_event(line)
            if event.get("Action") == "skip" and event.get("Test") and disk_heavy_policy.optional(event.get("Package", ""), event["Test"]):
                skipped.append(event["Package"] + "/" + event["Test"])
    return skipped


def source_state() -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise GateError("source revision is unavailable")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                           cwd=ROOT, capture_output=True, check=False, timeout=5)
    if dirty.returncode:
        raise GateError("source state is unavailable")
    return {"revision": revision, "dirty": bool(dirty.stdout)}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_pin(matrix: Path) -> dict:
    return {"source": source_state(), "matrix": {
        "path": matrix.relative_to(ROOT).as_posix() if matrix.is_relative_to(ROOT) else matrix.name,
        "digest": digest(matrix),
    }}


def require_fresh(pin: dict, matrix: Path) -> None:
    if pin["source"]["dirty"]:
        raise GateError("clean source is required to publish execution evidence")
    current = evidence_pin(matrix)
    if current["source"] != pin["source"]:
        raise GateError("source revision or dirty state changed during verification")
    if current["matrix"] != pin["matrix"]:
        raise GateError("verification matrix changed during execution")


def write_evidence(path: Path, value: dict, *, max_bytes: int = 64 << 10) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=True) + "\n"
    if max_bytes < 1 or len(encoded.encode()) > max_bytes:
        raise GateError("evidence exceeds its bound")
    pending = path.with_suffix(".pending")
    pending.write_text(encoded)
    pending.replace(path)


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
    max_report_bytes: int | None = None, retain_output: bool = True, cwd: Path | None = None,
    display_events: bool = True,
) -> int:
    """Capture redacted output with an independent deadline and joined process group."""
    if not command:
        raise ValueError("a test command is required")
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("capture deadline must be positive and finite")
    if max_report_bytes is not None and max_report_bytes < 1:
        raise ValueError("capture report limit must be positive")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as output:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, cwd=cwd,
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
                    displayed = None
                    try:
                        event = parse_go_event(line)
                    except GateError:
                        line = "Invalid test event: " + redact(line) if retain_output else "Invalid test event\n"
                    else:
                        event = {key: value for key, value in event.items() if key in {
                            "Time", "Action", "Package", "Test", "Elapsed", "Output", "FailedBuild",
                        }}
                        if "Output" in event:
                            if retain_output:
                                event["Output"] = redact(event["Output"])
                            else:
                                del event["Output"]
                        line = json.dumps(event) + "\n"
                        if not display_events:
                            displayed = event.get("Output", "")
                    size += len(line.encode("utf-8"))
                    if max_report_bytes is not None and size > max_report_bytes:
                        raise ValueError("test command report exceeds its bound")
                    output.write(line)
                    output.flush()
                    print(line if displayed is None else displayed, end="", flush=True)
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


def verify_python_report(
    report: Path,
    required: dict[str, int],
    *,
    modules: set[str] | None = None,
) -> list[str]:
    """Require parametrized case counts, rejecting skipped/failed selected cases."""
    if not required or any(type(count) is not int or count < 1 for count in required.values()):
        raise GateError("empty or invalid Python selection")
    if report.stat().st_size > 32 << 20:
        raise GateError("Python report exceeds its bound")
    try:
        source = ET.parse(report).getroot()
    except ET.ParseError as error:
        raise GateError("pytest did not produce a valid JUnit report") from error
    counts: Counter[str] = Counter()
    executed = set()
    for case in source.iter("testcase"):
        module, name = case.get("classname", ""), case.get("name", "")
        if modules is not None and module not in modules:
            continue
        identity = module + "::" + name
        if identity in executed or len(executed) >= 8192:
            raise GateError("Python case identities are duplicated or exceed their bound")
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise GateError("Python case did not pass: " + identity)
        executed.add(identity)
        counts[identity.split("[", 1)[0]] += 1
    if missing := [name for name, minimum in required.items() if counts[name] < minimum]:
        raise GateError("mandatory Python cases did not execute and pass: " + ", ".join(missing))
    return sorted(executed)


def compaction_matrix() -> dict:
    path = ROOT / "scripts/worker-compaction-matrix.json"
    if path.stat().st_size > 256 << 10:
        raise GateError("compaction matrix exceeds its bound")
    value = json.loads(path.read_text(), object_pairs_hook=json_object)
    if set(value) != {"schemaVersion", "go", "process", "python"} or value["schemaVersion"] != 1:
        raise GateError("invalid compaction matrix")
    if any(
        not isinstance(value[key], dict) or not value[key] for key in ("go", "process", "python")
    ):
        raise GateError("empty compaction matrix boundary")
    for key in ("go", "process"):
        for package, names in value[key].items():
            if (
                not package.startswith("github.com/grauwolf32/contractor/")
                or not isinstance(names, list)
                or not names
                or len(set(names)) != len(names)
                or any(
                    not re.fullmatch(r"Test[A-Za-z0-9_]+(?:/[^\x00-\x1f]+)*", name)
                    for name in names
                )
            ):
                raise GateError("invalid compaction Go identity")
    for name, minimum in value["python"].items():
        if (
            not re.fullmatch(r"tests\.test_[a-z0-9_]+::test_[a-z0-9_]+", name)
            or type(minimum) is not int
            or minimum < 1
        ):
            raise GateError("invalid compaction Python identity")
    return value


def verify_compaction(report: Path, boundary: str, scope: str | None = None) -> list[str]:
    required = compaction_matrix()[boundary]
    if boundary == "python":
        return verify_python_report(
            report, required, modules={name.split("::", 1)[0] for name in required}
        )
    if scope is not None:
        required = {
            package: [name for name in names if re.fullmatch(scope, name.split("/", 1)[0])]
            for package, names in required.items()
        }
        required = {package: names for package, names in required.items() if names}
    return verify_go_report(report, required)


def verify_runtime(report: Path) -> None:
    """Retain each gate's exact module scope and minimum parametrized case counts."""
    source = ET.parse(report).getroot()
    traces = {
        case.get("name"): case
        for case in source.iter("testcase")
        if case.get("classname") == "tests.test_runtime_trace_delivery_process"
    }
    if missing := TRACE_DELIVERY_CASES - traces.keys():
        raise GateError("mandatory Python trace delivery cases did not execute: " + ", ".join(sorted(missing)))
    for name in TRACE_DELIVERY_CASES:
        if any(traces[name].find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise GateError("Python trace delivery case did not pass: " + name)
    audit = gate("test-audit-completion-e2e.py")
    findings = gate("test-findings-e2e.py")
    engagement = gate("test-engagement-http-process.py")
    selections = (
        (audit, {name.split("::", 1)[0] for name in audit["matrix"]()["python"]}),
        (
            findings,
            {name.removesuffix(".py").replace("/", ".") for name in findings["PYTHON_FILES"]},
        ),
        (engagement, {name.split("::", 1)[0] for name in engagement["matrix"]()["python"]}),
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
    """Validate canonical output in the original family's exact root scope."""
    module = gate(f"test-{kind}-e2e.py")
    if kind == "audit-completion":
        verify_go_report(report, module["matrix"]()["go"], root_pattern=module["PATTERN"])
    else:
        verify_go_report(report, {module["PACKAGE"]: module["REQUIRED_TESTS"]},
                         root_pattern=pattern(module["PROCESS_TESTS"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("run", "verify-runtime", "verify-audit", "verify-findings", "verify-engagement-accounting", "findings-pattern",
                 "verify-iterative-postgres", "verify-iterative-process", "iterative-pattern",
                 "verify-compaction-go", "verify-compaction-process", "verify-compaction-runtime",
                 "compaction-pattern"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--scope", help="Exact consolidated process root selection")
    # Keep the child command separate from the report runner options.
    arguments = sys.argv[1:]
    delimiter = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:delimiter])
    command = arguments[delimiter + 1 :]
    if args.scope is not None and args.action != "verify-compaction-process":
        parser.error("--scope is only accepted for consolidated compaction processes")
    if args.action == "compaction-pattern":
        if command or args.report is not None or args.timeout_seconds is not None:
            parser.error("compaction-pattern takes no command or report")
        print(
            pattern(
                sorted(
                    {
                        name.split("/", 1)[0]
                        for names in compaction_matrix()["process"].values()
                        for name in names
                    }
                )
            )[2:-2]
        )
        return 0
    if args.action == "iterative-pattern":
        if command or args.report is not None or args.timeout_seconds is not None:
            parser.error("iterative-pattern takes no command or report options")
        selected = gate("test-audit-iterative.py")["matrix"]()["process"]
        print(pattern(case for cases in selected.values() for case in cases)[2:-2])
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
    if args.action.startswith("verify-compaction-"):
        boundary = {"go": "go", "process": "process", "runtime": "python"}[
            args.action.removeprefix("verify-compaction-")
        ]
        executed = verify_compaction(args.report, boundary, args.scope)
        print(f"Compaction {boundary}: {len(executed)} selected cases passed")
    elif args.action.startswith("verify-iterative-"):
        gate("test-audit-iterative.py")["verify"](args.report, args.action.removeprefix("verify-iterative-"))
    elif args.action == "verify-runtime":
        verify_runtime(args.report)
    elif args.action == "verify-engagement-accounting":
        gate("test-engagement-http-process.py")["verify_accounting_report"](args.report)
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
        print(redact(f"Release test report failed: {error}"), file=sys.stderr)
        sys.exit(1)
