#!/usr/bin/env python3
"""Required deterministic managed Evals gate; missing DB or skipped tests fail."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked(command):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def go_gate(name, arguments, evidence):
    command = ["go", "test", "-json", "-count=1", *arguments]
    print("+ " + " ".join(command), flush=True)
    passed, skipped, failures = set(), [], []
    with (evidence / (name + ".jsonl")).open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            log.write(line)
            try:
                event = json.loads(line)
            except ValueError:
                print(line, end="", flush=True)
                continue
            key = (event.get("Package"), event.get("Test"))
            if event["Action"] == "skip":
                skipped.append(key)
            elif event["Action"] == "pass" and key[1]:
                passed.add(key)
            elif event["Action"] == "fail":
                failures.append(key)
            if event["Action"] == "output" and (
                "FAIL" in event.get("Output", "") or "error:" in event.get("Output", "")
            ):
                print(event["Output"], end="", flush=True)
        status = process.wait()
    if status or skipped or failures or not passed:
        raise RuntimeError(
            f"{name}: exit={status}, skipped={skipped}, failed={failures}; see {evidence}"
        )
    print(f"{name}: {len(passed)} tests/subtests passed, no skips", flush=True)
    return {"command": command, "passed": len(passed), "skipped": 0}


def main():
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise RuntimeError(
            "CONTRACTOR_TEST_DATABASE_URL must name a disposable PostgreSQL database"
        )
    evidence = Path(
        os.environ.get(
            "CONTRACTOR_EVAL_EVIDENCE_DIR", ROOT / ".local/evidence/managed-evals"
        )
    ).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    os.environ["CONTRACTOR_EVAL_EVIDENCE_DIR"] = str(evidence)
    result = {
        "implementation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktreeChanges": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "gates": {},
    }
    result["gates"]["domain-store-service"] = go_gate(
        "domain-store-service",
        [
            "-race",
            "-timeout=8m",
            "./internal/evaldomain",
            "./internal/evalstore",
            "./internal/evalservice",
            "./internal/evalcoordinator",
        ],
        evidence,
    )
    result["gates"]["public-api"] = go_gate(
        "public-api",
        [
            "-race",
            "-timeout=5m",
            "./internal/httpapi/public",
            "-run",
            "Eval",
        ],
        evidence,
    )
    checked(["make", "ui-install", "ui-browser-install"])
    checked(["uv", "sync", "--project", "runtime", "--locked"])
    result["gates"]["process-browser"] = go_gate(
        "process-browser",
        [
            "-tags=e2e",
            "-timeout=26m",
            "./tests/ui-stack",
        ],
        evidence,
    )
    (evidence / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Managed Evals required gates passed: {evidence}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
