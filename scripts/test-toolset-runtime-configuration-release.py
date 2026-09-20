#!/usr/bin/env python3
"""Offline MCP release gate: real DB/process/browser, mandatory cases, no skips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts/toolset-runtime-configuration-matrix.json"
# Reuse the established credential-safe diagnostic redactor, including DSNs.
redact = runpy.run_path(str(ROOT / "scripts/test-audit-completion-e2e.py"))["redact"]


class GateError(Exception):
    pass


def verify_go(path: Path, required: dict[str, list[str]]) -> list[str]:
    started, passed = set(), set()
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
            action = event["Action"]
        except (ValueError, KeyError, TypeError) as error:
            raise GateError("Go did not emit a valid test event stream") from error
        name = event.get("Test")
        if action in {"skip", "fail"}:
            raise GateError(f"Go case failed or skipped: {name}")
        if name:
            key = (
                event.get("Package", "").removeprefix(
                    "github.com/grauwolf32/contractor/"
                ),
                name,
            )
            if action == "run":
                started.add(key)
            elif action == "pass":
                passed.add(key)
    expected = {
        (package, name) for package, names in required.items() for name in names
    }
    if missing := expected - (started & passed):
        raise GateError("Mandatory Go cases absent: " + repr(sorted(missing)))
    return sorted(f"{package}/{name}" for package, name in started & passed)


def verify_junit(path: Path, required: list[str], *, minimum: int = 1) -> list[str]:
    root = ET.parse(path).getroot()
    passed = []
    for case in root.iter("testcase"):
        name = case.get("classname", "") + "::" + case.get("name", "")
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise GateError("Mandatory runtime/UI case failed or skipped: " + name)
        passed.append(name)
    expected = {
        (
            module.removesuffix(".py").replace("/", ".")
            if module.endswith(".py")
            else module
        )
        + "::"
        + case
        for name in required
        for module, case in [name.split("::", 1)]
    }
    if expected - set(passed) or len(passed) < minimum:
        raise GateError(
            "Mandatory runtime/UI cases missing: "
            + repr(sorted(expected - set(passed)))
        )
    return sorted(passed)


def verify_browser(path: Path, required: list[str]) -> list[str]:
    report = json.loads(path.read_text())
    stats = report["stats"]
    if any(
        stats.get(key, 0) for key in ("skipped", "unexpected", "flaky")
    ) or stats.get("expected") != len(required):
        raise GateError("Browser count/failure/skip/retry requirements were not met")
    passed = []

    def visit(suite: dict) -> None:
        for spec in suite.get("specs", []):
            tests = spec.get("tests", [])
            if not tests or not spec.get("ok"):
                raise GateError("Browser case has no passing execution")
            for test in tests:
                results = test.get("results", [])
                if (
                    len(results) != 1
                    or results[0].get("status") != "passed"
                    or test.get("expectedStatus") != "passed"
                ):
                    raise GateError("Browser case skipped, retried or failed")
            passed.append(spec["title"])
        for child in suite.get("suites", []):
            visit(child)

    for suite in report["suites"]:
        visit(suite)
    if sorted(passed) != sorted(required):
        raise GateError("Browser mandatory case identities differ")
    return sorted(passed)


def command(
    args: list[str],
    cwd: Path,
    report: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict:
    print("Running " + " ".join(args), flush=True)
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=os.environ | (env or {}),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise GateError(f"Command exceeded {timeout}s") from error
        output.seek(0)
        text = output.read(16 * 1024 * 1024 + 1)
    if len(text) > 16 * 1024 * 1024:
        raise GateError("Command evidence exceeded 16 MiB")
    report.write_text(redact(text))
    if status:
        print(redact(text[-8000:]), flush=True)
        raise GateError(f"Command failed ({status}); evidence {report}")
    elapsed = round(time.monotonic() - started, 3)
    print(f"Passed in {elapsed}s; {report.name}", flush=True)
    return {
        "command": args,
        "cwd": str(cwd.relative_to(ROOT)) or ".",
        "seconds": elapsed,
        "exit_code": status,
        "report": report.name,
    }


def source_fingerprint(durable: Path) -> str:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if path.resolve() == durable.resolve() or not path.is_file():
            continue
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def run(output_dir: Path, durable: Path) -> None:
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
        raise GateError(
            "CONTRACTOR_TEST_DATABASE_URL is required (disposable PostgreSQL)"
        )
    for tool in ("go", "uv", "corepack", "node", "make"):
        if not shutil.which(tool):
            raise GateError(f"Required executable missing: {tool}")
    python = ROOT / "runtime/.venv/bin/python"
    for path in (
        python,
        ROOT / "ui/node_modules/@playwright/test",
        ROOT / "tests/e2e/testdata/toolsets/workflows/toolset_shared.yaml",
    ):
        if not path.exists():
            raise GateError(
                "Required Runtime/browser/fixture prerequisite missing: " + str(path)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir)).resolve()
    print("Evidence directory: " + str(directory), flush=True)
    matrix = json.loads(MATRIX.read_text())
    source_digest = source_fingerprint(durable)
    commands = []

    def execute(args: list[str], cwd: Path, name: str, timeout: int = 600, env=None):
        result = command(args, cwd, directory / name, timeout, env)
        commands.append(result)
        return directory / name

    execute(
        [
            str(python),
            "-c",
            "import importlib.metadata as m,json; import mcp,google.adk,httpx,jsonschema; v={n:m.version(n) for n in ('mcp','google-adk','httpx','jsonschema')}; assert v['mcp']=='1.30.0'; print(json.dumps(v))",
        ],
        ROOT,
        "runtime-versions.json",
        30,
    )
    execute(
        [
            "node",
            "--input-type=module",
            "-e",
            "import {chromium} from '@playwright/test'; if(!/^v24\\.20\\./.test(process.version)) throw new Error('Node 24.20.x required by ui/package.json'); const browser=await chromium.launch({headless:true}); console.log(JSON.stringify({node:process.version,chromium:browser.version()})); await browser.close();",
        ],
        ROOT / "ui",
        "browser-versions.json",
        45,
    )
    execute(["uv", "lock", "--check"], ROOT / "runtime", "runtime-lock.log", 90)
    execute(
        [
            "make",
            "verify-wire-contracts",
            "test-wire-cross-language",
            "verify-public-api",
            "test-config",
        ],
        ROOT,
        "contracts.log",
    )
    files = sorted({name.split("::", 1)[0] for name in matrix["python"]})
    execute(
        [
            str(python),
            "-m",
            "pytest",
            "-W",
            "error",
            *files,
            "--junitxml",
            str(directory / "runtime.xml"),
        ],
        ROOT / "runtime",
        "runtime.log",
        300,
    )
    python_cases = verify_junit(directory / "runtime.xml", matrix["python"])
    pattern = (
        "^("
        + "|".join(re.escape(name) for names in matrix["go"].values() for name in names)
        + ")$"
    )
    execute(
        [
            "go",
            "test",
            "-json",
            "-race",
            "-tags=integration",
            "-count=1",
            "-timeout=6m",
            *["./" + package for package in matrix["go"]],
            "-run",
            pattern,
        ],
        ROOT,
        "go.jsonl",
        420,
    )
    go_cases = verify_go(directory / "go.jsonl", matrix["go"])
    execute(
        ["make", "ui-generate-check", "ui-lint", "ui-typecheck", "ui-test", "ui-build"],
        ROOT,
        "ui.log",
        360,
    )
    execute(
        [
            "corepack",
            "pnpm",
            "exec",
            "vitest",
            "run",
            "src/routes/workflows/toolset-configurations.test.tsx",
            "--reporter=junit",
            "--outputFile=" + str(directory / "ui.xml"),
        ],
        ROOT / "ui",
        "ui-mandatory.log",
        90,
    )
    ui_cases = verify_junit(directory / "ui.xml", matrix["ui"])
    execute(
        [
            "go",
            "test",
            "-json",
            "-tags=e2e",
            "-count=1",
            "-timeout=15m",
            "./tests/e2e",
            "-run",
            "^TestToolsetRuntimeConfigurationAcrossProcesses$",
        ],
        ROOT,
        "process.jsonl",
        930,
        {"CONTRACTOR_TOOLSET_BROWSER_REPORT": str(directory / "browser.json")},
    )
    process_cases = verify_go(
        directory / "process.jsonl", {"tests/e2e": matrix["process"]}
    )
    browser_cases = verify_browser(directory / "browser.json", matrix["browser"])
    if source_fingerprint(durable) != source_digest:
        raise GateError("Source files changed during verification; rerun the gate")
    evidence = {
        "source_sha256": source_digest,
        "task": "V61-009",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "commands": commands,
        "prerequisites": {
            "postgresql": "isolated schema per production process/persistence test",
            "runtime": json.loads((directory / "runtime-versions.json").read_text()),
            "browser": json.loads((directory / "browser-versions.json").read_text()),
        },
        "mandatory_skips": 0,
        "cases": {
            "python": python_cases,
            "go": go_cases,
            "process": process_cases,
            "ui": ui_cases,
            "browser": browser_cases,
        },
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "evidence_directory": str(directory.relative_to(ROOT)),
        "boundaries": [
            "Server/PostgreSQL/Scheduler/private mTLS allocation/Python Runtime/ADK",
            "local real TCP Streamable HTTP MCP",
            "scripted OpenAI-compatible model endpoint",
            "built production UI and Chromium with real public APIs; lost replies dropped after server commits",
        ],
        "limits": [
            "Offline fixture evidence, not live-model quality or external MCP interoperability certification",
            "Only Streamable HTTP, no legacy SSE/stdio fallback, OAuth flows, images or resource fetching",
            "No deployment, production Run or user Audit performed",
        ],
    }
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    durable.parent.mkdir(parents=True, exist_ok=True)
    durable.write_text(json.dumps(evidence, indent=2) + "\n")
    print("Release gate passed with zero mandatory skips", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".local/toolset-release"
    )
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "tasks/evidence/v61-009.json"
    )
    options = parser.parse_args()
    try:
        run(options.output_dir, options.evidence)
    except (GateError, OSError, ValueError, ET.ParseError) as error:
        print("Toolset release gate failed: " + redact(str(error)), file=sys.stderr)
        sys.exit(1)
