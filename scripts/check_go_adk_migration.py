"""Characterize the supported Go ADK and an isolated, mechanically adapted v2 candidate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE = "github.com/grauwolf32/contractor/"
STREAMLINE = MODULE + "internal/planner/streamline"
BASELINE = "google.golang.org/adk"
CANDIDATE = BASELINE + "/v2"
EXTRA_FILES = (
    "scripts/check_go_adk_migration.py",
    "scripts/test_check_go_adk_migration.py",
    "internal/planner/streamline/go_adk_characterization_test.go",
    "internal/planner/streamline/go_adk_history_characterization_test.go",
    "internal/planner/streamline/finish_append_precedence_postgres_test.go",
)
# Exact identifiers are compiler evidence; no provider output belongs in repository evidence.
API_DIAGNOSTICS = {
    "tool_context_removed": r"undefined: agent\.ToolContext",
    "callback_context_removed": r"undefined: agent\.CallbackContext",
    "event_context_required": r"not enough arguments in call to adksession\.NewEvent",
}
OPTIONAL_SKIPS = {
    MODULE
    + "tests/integration/streamline:TestLiveGatewayToolCall": "explicit live Gateway opt-in",
    MODULE
    + "tests/integration/streamline:TestLiveRouterWorkflow": "explicit live Gateway opt-in",
}


HISTORY_CASES = (
    "complete_pair",
    "call_only",
    "response_only",
    "interrupted_parallel_pair",
    "repeated_name_distinct_ids",
    "empty_ids",
    "response_only_with_text",
    "mixed_complete_orphan_response_with_text",
)
OBSERVATION_MARKER = "GO_ADK_OBSERVATION "
IMPORT_DECLARATION = re.compile(
    r'(?ms)^import\s+(?:\((.*?)^\)|((?:(?:[A-Za-z_]\w*|\.)\s+)?"[^"\n]*"))'
)


def expected_observations(candidate: bool) -> dict[str, dict[str, Any]]:
    history = {
        "complete_pair": (["first"], ["first"], ["one"], False),
        "call_only": ([] if candidate else ["first"], [], [], False),
        "response_only": ([], [], [], False),
        "interrupted_parallel_pair": (
            ["first"] if candidate else ["first", "second"],
            ["first"],
            ["one"],
            False,
        ),
        "repeated_name_distinct_ids": (
            ["first", "second"],
            ["first", "second"],
            ["one", "two"],
            False,
        ),
        "empty_ids": ([""], [""], ["one"], False),
        "response_only_with_text": ([], [], [], candidate),
        "mixed_complete_orphan_response_with_text": (
            ["first"],
            ["first"] if candidate else ["first", "orphan"],
            ["one"] if candidate else ["one", "two"],
            True,
        ),
    }
    records = {}
    for name, (calls, responses, values, useful) in history.items():
        records["history:" + name] = {
            "group": "history",
            "case": name,
            "calls": calls,
            "responses": responses,
            "values": values,
            "useful_text": useful,
            "raw_preserved": True,
            "model_calls": 1,
            "tool_calls": 0,
        }
    for profile in ("streamline", "router"):
        for name in ("unknown_call", "multiple_calls", "finish_at_budget"):
            finished = name == "finish_at_budget"
            records["planner_admission:" + profile + "/" + name] = {
                "group": "planner_admission",
                "case": profile + "/" + name,
                "model_calls": 3 if finished else 2,
                "worker_calls": 1 if finished else 0,
                "outcome": "succeeded" if finished else "failed",
                "plan_transitions": 3 if finished else 0,
            }
    for name in ("parent_cancel", "consumer_break"):
        records["lifecycle:" + name] = {
            "group": "lifecycle",
            "case": name,
            "model_calls": 1,
            "active_after_return": 0,
            "iteration_cancelled": name == "parent_cancel",
            "caller_cancelled": name == "parent_cancel",
            "joined": True,
        }
    return records


def safe_observation(payload: Any) -> dict[str, Any]:
    """Retain bounded fixed fixture tokens, never arbitrary model or provider content."""
    baseline, candidate = expected_observations(False), expected_observations(True)
    if not isinstance(payload, dict):
        raise MigrationError("Unapproved SDK observation shape")
    group, case = payload.get("group"), payload.get("case")
    if not isinstance(group, str) or not isinstance(case, str):
        raise MigrationError("Unapproved SDK observation identity")
    key = group + ":" + case
    if key not in baseline or set(payload) != set(baseline[key]):
        raise MigrationError("Unapproved SDK observation fields")
    safe = {"group": group, "case": case}
    for field, expected in baseline[key].items():
        if field in safe:
            continue
        value = payload[field]
        alternative = candidate[key][field]
        if isinstance(expected, list):
            allowed = set(expected) | set(alternative)
            if isinstance(value, list) and len(value) <= 16:
                safe[field] = [
                    item if isinstance(item, str) and item in allowed else "unexpected"
                    for item in value
                ]
            else:
                safe[field] = "unexpected"
        elif isinstance(expected, bool):
            safe[field] = value if isinstance(value, bool) else "unexpected"
        elif isinstance(expected, int):
            safe[field] = (
                value if type(value) is int and 0 <= value <= 10000 else "unexpected"
            )
        elif isinstance(expected, str):
            safe[field] = value if value in (expected, alternative) else "unexpected"
        else:
            raise MigrationError("Unapproved SDK observation field type")
    return safe


def validate_observations(
    records: list[dict[str, Any]], *, candidate: bool
) -> dict[str, Any]:
    expected = expected_observations(candidate)
    seen = {}
    for record in records:
        if not isinstance(record, dict):
            raise MigrationError("SDK projection observation is not an object")
        group, case = record.get("group"), record.get("case")
        if not isinstance(group, str) or not isinstance(case, str):
            raise MigrationError(
                "SDK projection observation has no stable case identity"
            )
        key = group + ":" + case
        if key not in expected or record != expected[key]:
            raise MigrationError(
                "SDK projection observation differs from its explicit disposition: "
                + key
            )
        if key in seen and seen[key] != record:
            raise MigrationError(
                "Repeated SDK projection observation is nondeterministic: " + key
            )
        seen[key] = record
    if seen.keys() != expected.keys():
        raise MigrationError("Required SDK projection observation is absent")
    return {
        "case_count": len(seen),
        "observations": seen,
        "disposition": "Raw history and Contractor invariants are unchanged; v2 prunes unanswered calls/orphan responses and preserves useful orphan-adjacent text in prompt projection only.",
    }


class MigrationError(RuntimeError):
    """An incomplete or unexpected characterization cannot count as accepted evidence."""


def case_manifest() -> dict[str, list[str]]:
    prefix = STREAMLINE + ":"
    profiles = ("streamline", "router")
    raw = (
        "core-number",
        "core-bool",
        "core-array",
        "memory-number",
        "memory-null",
        "memory-extra",
        "state-fraction",
        "state-bool",
        "state-null",
        "state-extra",
    )
    return {
        "schemas": [
            prefix + f"TestGoADKMigrationSchemas/{p}/{f}"
            for p in profiles
            for f in ("core", "memory", "state")
        ],
        "arguments": [
            prefix + f"TestGoADKMigrationRawArguments/{p}/{v}"
            for p in profiles
            for v in raw
        ],
        "callbacks": [
            prefix + "TestGoADKMigrationModelCallbacks/" + name
            for name in (
                "success",
                "before-rejection",
                "provider-error",
                "rejected-response",
                "after-rejection",
            )
        ],
        "gateway_admission": [
            prefix + f"TestGoADKMigrationHTTPAdmission/{p}/{name}"
            for p in profiles
            for name in ("success", "model-budget", "rejected-usage")
        ],
        "history": [
            prefix + "TestGoADKMigrationHistory/" + name for name in HISTORY_CASES
        ],
        "planner_admission": [
            prefix + f"TestGoADKMigrationHistoryPlannerAdmission/{p}/{name}"
            for p in profiles
            for name in ("unknown_call", "multiple_calls", "finish_at_budget")
        ],
        "lifecycle": [
            prefix + "TestGoADKMigrationLifecycle/" + name
            for name in ("parent_cancel", "consumer_break")
        ],
        "session_failure_precedence": [
            *[
                prefix
                + f"TestPostgresModeledPlannerAppendFailurePrecedence/{p}/finish/{outcome}/{fault}"
                for p in profiles
                for outcome in ("succeeded", "failed-retryable", "failed-final")
                for fault in ("invalid", "unavailable")
            ],
            *[
                prefix
                + f"TestPostgresModeledPlannerAppendFailurePrecedence/{p}/ordinary/none/{fault}"
                for p in profiles
                for fault in ("invalid", "unavailable")
            ],
            *[
                prefix
                + f"TestModeledPlannerCancellationKeepsCompletionOwnership/{p}/{boundary}"
                for p in profiles
                for boundary in ("before-finish", "completion-write")
            ],
        ],
        "final_append_cancellation": [
            prefix
            + f"TestPostgresModeledPlannerFinalOutcomeSurvivesAppendCancellation/{p}/{fault}"
            for p in profiles
            for fault in ("invalid", "unavailable")
        ],
        "budgets": [
            prefix + "TestStreamlineStopsWhenModelCallBudgetIsExhausted",
            prefix + "TestStreamlineStopsWhenTokenBudgetIsExceeded",
            prefix + "TestStreamlineFinishesAfterWorkerBudgetAndTextTurn",
            MODULE
            + "internal/planner/router:TestRouterFinishesAfterWorkerBudgetAndTextTurn",
        ],
        "session": [
            MODULE
            + "internal/planner/session:TestADKSessionPersistsOnlyBoundedRedactedEventFacts",
            MODULE
            + "internal/planner/session:TestPostgresSessionPersistsCompletionForRecovery",
            MODULE
            + "tests/integration/streamline:TestPostgresGatewayWorkerFlowRecoversWithoutSemanticReplay",
        ],
    }


def validate_cases(
    events: list[dict[str, Any]], manifest: dict[str, list[str]]
) -> dict[str, Any]:
    if not manifest or any(
        not cases or len(cases) != len(set(cases)) for cases in manifest.values()
    ):
        raise MigrationError(
            "Every required group needs a nonempty unique case inventory"
        )
    started, passed, packages, skipped, failed = set(), set(), set(), set(), set()
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("Action"), str):
            raise MigrationError("Go report contains an invalid event")
        package, test, action = event.get("Package"), event.get("Test"), event["Action"]
        if not isinstance(package, str):
            continue  # Go build-output events have ImportPath instead of Package.
        if test is not None and not isinstance(test, str):
            raise MigrationError("Go report contains an invalid case identity")
        key = package + ":" + test if test else None
        if action == "run" and key:
            started.add(key)
        elif action == "pass":
            if key:
                passed.add(key)
            else:
                packages.add(package)
        elif action == "skip" and key:
            skipped.add(key)
        elif action == "fail":
            failed.add(key or package)
    missing = sorted(
        {case for cases in manifest.values() for case in cases} - (started & passed)
    )
    required_packages = {
        case.split(":", 1)[0] for cases in manifest.values() for case in cases
    }
    unknown_skips = sorted(skipped - OPTIONAL_SKIPS.keys())
    if missing or failed or unknown_skips or required_packages - packages:
        raise MigrationError(
            json.dumps(
                {
                    "missing_cases": missing,
                    "failed": sorted(failed),
                    "unexpected_skips": unknown_skips,
                    "missing_package_pass": sorted(required_packages - packages),
                },
                sort_keys=True,
            )
        )
    parents = set()
    for case in started:
        package, test = case.split(":", 1)
        pieces = test.split("/")
        parents.update(
            package + ":" + "/".join(pieces[:i]) for i in range(1, len(pieces))
        )
    return {
        "groups": {
            group: {"passed": len(cases), "cases": cases}
            for group, cases in manifest.items()
        },
        "executed_leaf_cases": len((started & passed) - parents),
        "optional_skips": [
            {"case": case, "reason": OPTIONAL_SKIPS[case]} for case in sorted(skipped)
        ],
        "passed_packages": sorted(packages),
    }


def expected_compile_failure(exit_code: int, output: str) -> dict[str, bool]:
    found = {
        name: re.search(pattern, output) is not None
        for name, pattern in API_DIAGNOSTICS.items()
    }
    if exit_code != 1 or not all(found.values()):
        raise MigrationError(
            "Imports-only phase did not reproduce every required API compile failure"
        )
    return found


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(
    arguments: list[str], cwd: Path, *, environment=None
) -> subprocess.CompletedProcess[str]:
    # Every phase owns and joins its process group before temporary checkout cleanup.
    with subprocess.Popen(
        arguments,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(arguments, 124, stdout, stderr)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(
            arguments, process.returncode, stdout, stderr
        )


def checked(arguments: list[str], cwd: Path, *, environment=None) -> str:
    result = command(arguments, cwd, environment=environment)
    if result.returncode:
        raise MigrationError(
            f"Setup command failed: {arguments[:3]} (exit {result.returncode})"
        )
    return result.stdout


def environment() -> dict[str, str]:
    selected = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(
                (
                    "CONTRACTOR_RUN_",
                    "CONTRACTOR_LIVE_",
                    "CONTRACTOR_TEST_OTLP",
                    "CONTRACTOR_GO_ADK_",
                )
            )
        )
    }
    selected.update(GOMAXPROCS="2", GOTOOLCHAIN="local", GOWORK="off")
    selected.pop("GOFLAGS", None)
    selected["TMPDIR"] = str(Path(selected.get("TMPDIR", "/tmp")).resolve(strict=True))
    return selected


def import_inventory(root: Path) -> list[str]:
    found = []
    for folder in ("cmd", "internal", "tests", "tools"):
        for path in sorted((root / folder).rglob("*.go")):
            if any(
                re.search(r'"google\.golang\.org/adk(?:/[^"\n]*)?"', block[0])
                for block in IMPORT_DECLARATION.finditer(path.read_text())
            ):
                found.append(path.relative_to(root).as_posix())
    if not found:
        raise MigrationError("No Go ADK consumers found")
    return sorted(found)


def clone_snapshot(source: Path, target: Path) -> None:
    checked(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--local",
            str(source),
            str(target),
        ],
        source,
    )
    tracked = checked(["git", "ls-files", "-z"], source).split("\0")
    for relative in [*filter(None, tracked), *EXTRA_FILES]:
        current, destination = source / relative, target / relative
        if not current.is_file():
            if relative in EXTRA_FILES:
                raise MigrationError(f"Required source fixture absent: {relative}")
            if destination.is_file():
                destination.unlink()
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, destination)


def mechanical_imports(root: Path) -> list[str]:
    files = import_inventory(root)
    for relative in files:
        path = root / relative
        text = path.read_text()
        # Restrict changes to import declarations, never model prompts or tool payloads.
        text = IMPORT_DECLARATION.sub(
            lambda declaration: re.sub(
                r'("google\.golang\.org/adk)(?=/|")',
                r"\1/v2",
                declaration[0],
            ),
            text,
        )
        path.write_text(text)
    return files


def mechanical_contexts(root: Path, files: list[str]) -> dict[str, int]:
    counts = {"tool_context": 0, "callback_context": 0, "event_context": 0}
    for relative in files:
        path = root / relative
        text = path.read_text()
        for old, key in (
            ("ToolContext", "tool_context"),
            ("CallbackContext", "callback_context"),
        ):
            text, count = re.subn(r"\bagent\." + old + r"\b", "agent.Context", text)
            counts[key] += count
        if ".NewEvent(" in text:
            if not relative.endswith("_test.go"):
                raise MigrationError(
                    "NewEvent production call requires an explicit mechanical context mapping"
                )
            context = (
                "ctx"
                if relative.endswith("go_adk_history_characterization_test.go")
                else "t.Context()"
            )
            text, count = re.subn(
                r'\b(adksession|session)\.NewEvent\(("[^"\n]*")\)',
                lambda match, selected=context: (
                    f"{match[1]}.NewEvent({selected}, {match[2]})"
                ),
                text,
            )
            counts["event_context"] += count
        path.write_text(text)
    if not all(counts.values()):
        raise MigrationError("Mechanical scope lost a known API change")
    checked(["gofmt", "-w", *files], root)
    return counts


def json_values(raw: str) -> list[dict[str, Any]]:
    decoder, offset, values = json.JSONDecoder(), 0, []
    while offset < len(raw):
        if raw[offset].isspace():
            offset += 1
            continue
        value, offset = decoder.raw_decode(raw, offset)
        if not isinstance(value, dict):
            raise MigrationError("Module graph value is not an object")
        values.append(value)
    return values


def module_graph(root: Path, env: dict[str, str]) -> dict[str, Any]:
    raw = checked(["go", "list", "-m", "-json", "all"], root, environment=env)
    safe = []
    for module in json_values(raw):
        if module.get("Replace"):
            raise MigrationError(
                "An unexpected module replace prevents a reproducible candidate"
            )
        safe.append(
            {
                key: module[key]
                for key in ("Path", "Version", "Sum", "GoModSum", "GoVersion", "Main")
                if key in module
            }
        )
    safe.sort(key=lambda value: value["Path"])
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
    return {
        "modules": safe,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "go_mod_sha256": digest(root / "go.mod"),
        "go_sum_sha256": digest(root / "go.sum"),
    }


def sdk_sources(root: Path, selection: str, env: dict[str, str]) -> dict[str, Any]:
    data = json.loads(
        checked(["go", "mod", "download", "-json", selection], root, environment=env)
    )
    if data.get("Error") or not data.get("Sum") or not data.get("GoModSum"):
        raise MigrationError("SDK module download did not supply verified checksums")
    directory = Path(data["Dir"])
    files = (
        "go.mod",
        "agent/context.go",
        "agent/run_config.go",
        "agent/llmagent/llmagent.go",
        "model/llm.go",
        "session/service.go",
        "session/session.go",
        "tool/functiontool/function.go",
        "internal/toolinternal/tool.go",
        "internal/llminternal/contents_processor.go",
        "runner/runner.go",
        "runner/run_node.go",
        "runner/agent_node.go",
    )
    return {
        **{key: data[key] for key in ("Path", "Version", "Sum", "GoModSum")},
        "source_sha256": {
            name: digest(directory / name)
            for name in files
            if (directory / name).is_file()
        },
    }


def execute_tests(
    name: str, root: Path, arguments: list[str], env: dict[str, str], phases: list
) -> list[dict]:
    result = command(arguments, root, environment=env)
    events: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    phase = {
        "phase": name,
        "command": arguments,
        "exit_code": result.returncode,
        "case_events": events,
        "observations": observations,
    }
    # Record the real process outcome even if its observation protocol is malformed.
    phases.append(phase)
    for line in (result.stdout + result.stderr).splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # make recipes/build diagnostics are not case evidence.
        if isinstance(event, dict) and isinstance(event.get("Action"), str):
            if event["Action"] in ("run", "pass", "skip", "fail"):
                events.append(
                    {
                        key: event[key]
                        for key in ("Action", "Package", "Test")
                        if key in event
                    }
                )
            output = event.get("Output", "")
            if isinstance(output, str) and OBSERVATION_MARKER in output:
                try:
                    payload = json.loads(output.split(OBSERVATION_MARKER, 1)[1])
                except ValueError as error:
                    raise MigrationError("Malformed SDK observation") from error
                observations.append(safe_observation(payload))
    phase["case_counts"] = {
        action: sum(
            event.get("Action") == action and "Test" in event for event in events
        )
        for action in ("run", "pass", "skip", "fail")
    }
    print(
        f"{name}: exit {result.returncode}, {sum(e.get('Action') == 'run' for e in events)} started case records",
        flush=True,
    )
    return list(events)


def run(source: Path, version: str, evidence: dict[str, Any]) -> dict[str, Any]:
    phases = evidence["phases"]
    env = environment()
    if not re.search(
        r"google\.golang\.org/adk v1\.6\.0(?:\s|$)", (source / "go.mod").read_text()
    ):
        raise MigrationError(
            "Characterization requires the supported v1.6.0 production baseline"
        )
    if not env.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise MigrationError("A dedicated configured PostgreSQL database is required")
    manifest = case_manifest()
    if any(not values for values in manifest.values()):
        raise MigrationError("Characterization fixture manifest is incomplete")
    revision = checked(["git", "rev-parse", "HEAD"], source).strip()
    toolchain = checked(["go", "version"], source, environment=env).strip()
    inventory = import_inventory(source)
    evidence.update(
        source_revision=revision,
        toolchain=toolchain,
        consumer_files=inventory,
        source_sha256={
            path: digest(source / path)
            for path in [*inventory, *EXTRA_FILES, "go.mod", "go.sum"]
        },
        manifest=manifest,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="go-adk-characterization-"
        ) as temporary:
            directory = Path(temporary).resolve()
            baseline, imports, adapted = [
                directory / name for name in ("baseline", "imports-only", "adapted")
            ]
            clone_snapshot(source, baseline)
            clone_snapshot(baseline, imports)
            clone_snapshot(baseline, adapted)
            baseline_events = execute_tests(
                "baseline-planners",
                baseline,
                [
                    "go",
                    "test",
                    "-p=1",
                    "-json",
                    "-mod=readonly",
                    "-count=1",
                    "-timeout=20m",
                    "./internal/planner/...",
                ],
                env,
                phases,
            )
            make_env = {**env, "GOFLAGS": "-json -p=1 -mod=readonly"}
            baseline_events = baseline_events + execute_tests(
                "baseline-streamline",
                baseline,
                ["make", "test-streamline"],
                make_env,
                phases,
            )
            evidence["baseline_graph"] = module_graph(baseline, env)
            evidence["baseline_sdk"] = sdk_sources(baseline, BASELINE + "@v1.6.0", env)
            changed = mechanical_imports(imports)
            for arguments in (
                ["go", "mod", "edit", "-droprequire=" + BASELINE],
                ["go", "mod", "edit", "-require=" + CANDIDATE + "@" + version],
                ["go", "mod", "edit", "-go=1.26.6"],
                ["go", "mod", "tidy"],
            ):
                checked(arguments, imports, environment=env)
            evidence["candidate_graph"] = module_graph(imports, env)
            evidence["candidate_sdk"] = sdk_sources(
                imports, CANDIDATE + "@" + version, env
            )
            compile_command = [
                "go",
                "test",
                "-p=1",
                "-json",
                "-mod=readonly",
                "-gcflags=-e",
                "-run=^$",
                "./internal/planner/streamline",
                "./internal/planner/session",
                "./internal/planner/router",
            ]
            result = command(
                compile_command,
                imports,
                environment=env,
            )
            compile_phase = {
                "phase": "imports-only-compile",
                "command": compile_command,
                "exit_code": result.returncode,
                "candidate_build_passed": result.returncode == 0,
            }
            phases.append(compile_phase)
            compile_phase["expected_api_diagnostics"] = expected_compile_failure(
                result.returncode, result.stdout + result.stderr
            )
            mechanical_imports(adapted)
            for filename in ("go.mod", "go.sum"):
                shutil.copy2(imports / filename, adapted / filename)
            evidence["mechanical_api_counts"] = mechanical_contexts(adapted, changed)
            evidence["adapted_source_sha256"] = {
                path: digest(adapted / path) for path in changed
            }
            candidate_events = execute_tests(
                "adapted-planners",
                adapted,
                [
                    "go",
                    "test",
                    "-p=1",
                    "-json",
                    "-mod=readonly",
                    "-count=1",
                    "-timeout=20m",
                    "./internal/planner/...",
                ],
                env,
                phases,
            )
            candidate_events = candidate_events + execute_tests(
                "adapted-streamline",
                adapted,
                ["make", "test-streamline"],
                make_env,
                phases,
            )
            execute_tests(
                "adapted-consumer-test-compile",
                adapted,
                [
                    "go",
                    "test",
                    "-p=1",
                    "-json",
                    "-mod=readonly",
                    "-run=^$",
                    *sorted({"./" + str(Path(path).parent) for path in changed}),
                ],
                env,
                phases,
            )
            execute_tests(
                "adapted-production-build",
                adapted,
                ["go", "build", "-p=1", "-mod=readonly", "./cmd/...", "./internal/..."],
                env,
                phases,
            )
            validation_errors = []
            for name, events in (
                ("baseline", baseline_events),
                ("adapted", candidate_events),
            ):
                try:
                    evidence[name + "_cases"] = validate_cases(events, manifest)
                except MigrationError as error:
                    validation_errors.append({"phase": name, "error": str(error)})
            for name, candidate in (("baseline", False), ("adapted", True)):
                records = [
                    record
                    for phase in phases
                    if phase["phase"].startswith(name + "-")
                    for record in phase.get("observations", [])
                ]
                try:
                    evidence[name + "_observations"] = validate_observations(
                        records, candidate=candidate
                    )
                except MigrationError as error:
                    validation_errors.append(
                        {"phase": name + "-observations", "error": str(error)}
                    )
            evidence["validation_errors"] = validation_errors
            evidence["accepted"] = not validation_errors and all(
                phase["exit_code"] == 0
                for phase in phases
                if phase["phase"] != "imports-only-compile"
            )
    finally:
        if (
            any(
                digest(source / path) != checksum
                for path, checksum in evidence["source_sha256"].items()
            )
            or import_inventory(source) != inventory
            or checked(["git", "rev-parse", "HEAD"], source).strip() != revision
        ):
            evidence["accepted"] = False
            raise MigrationError("Source checkout changed during characterization")
    return evidence


def compact_evidence(evidence: dict[str, Any]) -> None:
    """Keep exact case actions without repeating output/start event metadata."""
    for previous in evidence.get("previous_runs", []):
        compact_evidence(previous)
    for phase in evidence.get("phases", []):
        if "case_events" not in phase:
            continue
        cases: dict[str, set[str]] = {}
        packages: dict[str, set[str]] = {}
        for event in phase.pop("case_events"):
            action, package, test = (
                event.get("Action"),
                event.get("Package"),
                event.get("Test"),
            )
            if action not in ("run", "pass", "skip", "fail") or not package:
                continue
            if test:
                cases.setdefault(package + ":" + test, set()).add(action)
            else:
                packages.setdefault(package, set()).add(action)
        phase["case_actions"] = {
            key: "/".join(
                action for action in ("run", "pass", "skip", "fail") if action in seen
            )
            for key, seen in sorted(cases.items())
        }
        phase["package_actions"] = {
            key: "/".join(sorted(seen)) for key, seen in sorted(packages.items())
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("v2.6.0",), required=True)
    parser.add_argument("--write-evidence", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.write_evidence.resolve()
        == (ROOT / "tasks/evidence/v419-go-adk-26-review.json").resolve()
    ):
        parser.error(
            "The original review evidence is immutable; choose a characterization record"
        )
    previous_runs = []
    if args.write_evidence.exists():
        previous = json.loads(args.write_evidence.read_text())
        previous_runs = previous.pop("previous_runs", [])
        if not isinstance(previous_runs, list) or len(previous_runs) >= 32:
            raise MigrationError(
                "Evidence history limit reached; choose a new evidence file"
            )
        previous_runs.append(previous)
    phases = []
    evidence: dict[str, Any] = {"accepted": False, "phases": phases}
    try:
        evidence = run(ROOT, args.candidate, evidence)
    except (MigrationError, OSError, ValueError) as error:
        evidence["accepted"] = False
        evidence["setup_failure"] = str(error)
    except KeyboardInterrupt:
        evidence["accepted"] = False
        evidence["interrupted"] = True
    evidence.update(
        previous_runs=previous_runs,
        schema_version="1.0",
        date=datetime.now(UTC).date().isoformat(),
        candidate=CANDIDATE + " " + args.candidate,
        scope="Test-only isolated characterization; not race/process/compaction/release acceptance",
    )
    compact_evidence(evidence)
    args.write_evidence.parent.mkdir(parents=True, exist_ok=True)
    staging = args.write_evidence.with_suffix(".staging")
    staging.write_text(json.dumps(evidence, indent=2) + "\n")
    staging.replace(args.write_evidence)
    return 130 if evidence.get("interrupted") else (0 if evidence["accepted"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
