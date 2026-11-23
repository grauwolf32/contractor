#!/usr/bin/env python3
"""Measure finite custom-Gateway workloads in isolated, locked Runtime children.

Samples are informational. Callback durations overlap and are never summed as
disjoint wall time; no duration or speedup is an acceptance threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from importlib import metadata, util
from pathlib import Path

import tomllib


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--turns", type=int, choices=(50, 150), required=True)
    parser.add_argument("--mode", choices=("shared", "isolated"), required=True)
    parser.add_argument("--samples", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fixture-sha256", help=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)
    for name in ("repository", "fixture", "workloads"):
        setattr(parsed, name, getattr(parsed, name).resolve(strict=True))
    require(
        parsed.fixture.is_file() and parsed.workloads.is_file(),
        "fixture files are required",
    )
    for relative in (
        "runtime/src",
        "runtime/tests",
        "runtime/uv.lock",
        "runtime/pyproject.toml",
    ):
        require(
            (parsed.repository / relative).exists(),
            "repository is not a Runtime source root",
        )
    workload = json.loads(parsed.workloads.read_text())
    require(workload.get("schema_version") == 1, "unsupported workload schema")
    require(
        parsed.turns in workload["main_turns"]
        and parsed.mode in workload["session_modes"],
        "selection is absent from the workload inventory",
    )
    parsed.width = workload["calls_per_tool_turn"]
    parsed.subtasks = workload["sequential_subtasks"]
    parsed.ceilings = workload["ceilings"]
    require(
        type(parsed.width) is int and parsed.width > 0, "tool width must be positive"
    )
    require(
        type(parsed.subtasks) is int and parsed.subtasks > 0,
        "subtask count must be positive",
    )
    require(
        parsed.turns + 1 <= parsed.ceilings["model_calls"], "model ceiling exceeded"
    )
    require(
        parsed.width * (parsed.turns - 1) <= parsed.ceilings["tool_calls"],
        "tool ceiling exceeded",
    )
    require(parsed.subtasks <= 1000, "subtask ceiling exceeded")
    if parsed.fixture_sha256 is not None:
        require(
            sha256(parsed.fixture) == parsed.fixture_sha256,
            "fixture changed during collection",
        )
    return parsed


def git(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL
    )


def source(module_name: str, root: Path) -> dict:
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve(strict=True)
    require(path.is_relative_to(root), "import escaped its selected source root")
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def disable_network() -> None:
    """Deny Internet/DNS sockets; asyncio's local wakeup socket remains usable."""
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded_connect(instance, address):
        require(
            instance.family not in (socket.AF_INET, socket.AF_INET6),
            "network is disabled for measurements",
        )
        return connect(instance, address)

    def guarded_connect_ex(instance, address):
        require(
            instance.family not in (socket.AF_INET, socket.AF_INET6),
            "network is disabled for measurements",
        )
        return connect_ex(instance, address)

    def denied(*_args, **_kwargs):
        raise RuntimeError("DNS is disabled for measurements")

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = denied
    socket.gethostbyname = denied
    socket.gethostbyname_ex = denied


def counters(raw: dict, args: argparse.Namespace) -> dict:
    """Validate semantic identities, then discard all raw IDs and tool payloads."""
    turns, width, subtasks = args.turns, args.width, args.subtasks
    per_task = width * (turns - 1)
    expected_tools = per_task * subtasks
    require(
        (raw["turns"], raw["width"], raw["mode"], raw["subtasks"])
        == (turns, width, args.mode, subtasks),
        "fixture selection differs",
    )
    logical = raw["logical_ids"]
    require(
        len(logical) == subtasks
        and len(set(logical)) == subtasks
        and all(isinstance(item, str) and item for item in logical),
        "logical invocation identity differs",
    )
    effects = raw["effects"]
    require(len(effects) == expected_tools, "effect count differs")
    require(
        raw["starts"] == [index * per_task for index in range(subtasks)]
        and raw["finishes"] == [(index + 1) * per_task for index in range(subtasks)],
        "effects crossed logical invocation boundaries",
    )
    semantic_effects = []
    for owner in range(subtasks):
        selected = effects[owner * per_task : (owner + 1) * per_task]
        require(
            all(
                item[0] == owner and item[1] == index and item[2] == logical[owner]
                for index, item in enumerate(selected)
            ),
            "effect identity/order differs",
        )
        semantic_effects.extend([[item[0], item[1]] for item in selected])
    work = raw["work"]
    require(
        work["model_calls"] == turns * subtasks and work["finalizer_calls"] == subtasks,
        "model/finalizer count differs",
    )
    created, retained = work["created_session_ids"], work["session_ids"]
    session_count = 1 if args.mode == "shared" else subtasks
    require(
        len(created) == session_count and len(set(created)) == session_count,
        "session creation count differs",
    )
    require(
        (
            args.mode == "shared"
            and len(retained) == subtasks
            and set(retained) == set(created)
        )
        or (args.mode == "isolated" and not retained),
        "session identity/mode differs",
    )
    expected_callbacks = {
        "before_run_callback": subtasks,
        "before_model_callback": turns * subtasks,
        "after_model_callback": turns * subtasks,
        "before_tool_callback": expected_tools,
        "after_tool_callback": expected_tools,
    }
    require(work["callback_calls"] == expected_callbacks, "callback count differs")
    safe_work = {}
    for name, value in work.items():
        if name in {"created_session_ids", "session_ids"}:
            continue
        require(isinstance(name, str) and name.isidentifier(), "invalid counter name")
        if isinstance(value, dict):
            require(
                set(value) <= set(expected_callbacks)
                and all(finite_number(item) for item in value.values()),
                "invalid callback counters",
            )
        else:
            require(finite_number(value), "invalid numeric work counter")
        safe_work[name] = value
    requests = raw["request_sizes"]
    snapshots = raw["snapshot_bytes"]
    require(
        len(requests) == (turns + 1) * subtasks
        and all(type(item) is int and item > 0 for item in requests)
        and sum(requests) == work["request_bytes"]
        and max(requests) == work["max_request_bytes"],
        "request accounting differs",
    )
    require(
        len(snapshots) == subtasks
        and all(
            type(item) is int and 0 < item <= args.ceilings["state_bytes"]
            for item in snapshots
        ),
        "snapshot count/size differs",
    )
    retained_count = min(expected_tools, args.ceilings["tool_records"])
    require(
        raw["retained_tool_records"] == retained_count
        and raw["reported_tool_calls"] == expected_tools,
        "retained/aggregate tool count differs",
    )
    ordinals = []
    for name in ("retained_first_call_id", "retained_last_call_id"):
        match = re.fullmatch(r"tool-([0-9]+)", raw[name] or "")
        require(match is not None, "retained call ordinal is unavailable")
        ordinals.append(int(match.group(1)))
    require(
        ordinals == [expected_tools - retained_count + 1, expected_tools],
        "retained call range differs",
    )
    require(
        type(raw["detail_truncated"]) is bool
        and raw["detail_truncated"] == (expected_tools > retained_count),
        "detail truncation differs",
    )
    require(
        type(raw["report_bytes"]) is int
        and 0 < raw["report_bytes"] <= args.ceilings["report_bytes"],
        "report size differs",
    )
    require(
        finite_number(raw["elapsed_seconds"])
        and raw["elapsed_seconds"] >= work["provider_seconds"],
        "elapsed accounting differs",
    )
    return {
        "work": safe_work,
        "logical_invocations": len(logical),
        "effects": len(effects),
        "semantic_effects_sha256": hashlib.sha256(
            json.dumps(semantic_effects, separators=(",", ":")).encode()
        ).hexdigest(),
        "created_sessions": session_count,
        "session_identity_validated": True,
        "retained_tool_records": retained_count,
        "retained_call_ordinals": ordinals,
        "reported_tool_calls": raw["reported_tool_calls"],
        "detail_truncated": raw["detail_truncated"],
        "report_bytes": raw["report_bytes"],
        "request_count": len(requests),
        "snapshot_bytes": snapshots,
        "elapsed_seconds": raw["elapsed_seconds"],
        "framework_seconds_excluding_mock_handler": raw["elapsed_seconds"]
        - work["provider_seconds"],
        "callback_aggregate_seconds_non_disjoint": sum(
            work["callback_seconds"].values()
        ),
        "results_snapshot_independence_and_cleanup_validated_by_fixture": True,
    }


def sample(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    repository = args.repository
    expected_path = [str(repository / "runtime/src"), str(repository / "runtime/tests")]
    require(
        os.environ.get("PYTHONPATH", "").split(os.pathsep) == expected_path,
        "measurement PYTHONPATH must contain only its selected source/tests",
    )
    project = tomllib.loads((repository / "runtime/pyproject.toml").read_text())
    pins = [
        value.split("==", 1)[1]
        for value in project["project"]["dependencies"]
        if value.startswith("google-adk==")
    ]
    require(len(pins) == 1, "exact SDK pin is required")
    require(sys.version_info[:2] == (3, 13), "supported Python 3.13 is required")
    versions = {
        name: metadata.version(name)
        for name in ("google-adk", "google-genai", "authlib", "a2a-sdk")
    }
    require(
        versions["google-adk"] == pins[0], "installed SDK differs from repository pin"
    )
    disable_network()
    with (
        open(os.devnull, "w") as quiet,
        contextlib.redirect_stdout(quiet),
        contextlib.redirect_stderr(quiet),
        warnings.catch_warnings(record=True) as emitted_warnings,
    ):
        began = time.perf_counter()
        production = {
            name: source(name, repository / "runtime/src")
            for name in (
                "contractor_runtime.worker.runtime",
                "contractor_runtime.llm.openai",
                "contractor_runtime.allocation",
                "contractor_runtime.factories",
            )
        }
        sdk_source = source("google.adk", repository / "runtime/.venv")
        core_import_seconds = time.perf_counter() - began
        began = time.perf_counter()
        specification = util.spec_from_file_location(
            "contractor_adk_scale_fixture", args.fixture
        )
        require(
            specification is not None and specification.loader is not None,
            "fixture cannot be imported",
        )
        fixture = util.module_from_spec(specification)
        sys.modules[specification.name] = fixture
        specification.loader.exec_module(fixture)
        test_source = source("test_adk_runtime", repository / "runtime/tests")
        fixture_import_seconds = time.perf_counter() - began
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="contractor-adk-scale-") as folder:
            raw = asyncio.run(
                fixture.run_workload(
                    Path(folder),
                    turns=args.turns,
                    width=args.width,
                    mode=args.mode,
                    subtasks=args.subtasks,
                )
            )
        fixture_execution_seconds = time.perf_counter() - began
    safe = counters(raw, args)
    warning_counts = Counter(
        (item.category.__name__, hashlib.sha256(str(item.message).encode()).hexdigest())
        for item in emitted_warnings
    )
    require(
        sha256(args.fixture) == args.fixture_sha256, "fixture changed during sample"
    )
    safe.update(
        {
            "status": "measured",
            "selection": {
                "turns": args.turns,
                "width": args.width,
                "mode": args.mode,
                "subtasks": args.subtasks,
            },
            "versions": versions,
            "python": sys.version.split()[0],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "warnings_default_policy": [
                {"category": category, "message_sha256": digest, "count": count}
                for (category, digest), count in sorted(warning_counts.items())
            ],
            "repository_revision": git(repository, "rev-parse", "HEAD")
            .decode()
            .strip(),
            "repository_clean": not bool(git(repository, "status", "--porcelain")),
            "tracked_patch_sha256": hashlib.sha256(
                git(repository, "diff", "--binary", "HEAD", "--")
            ).hexdigest(),
            "lock_sha256": sha256(repository / "runtime/uv.lock"),
            "fixture_sha256": args.fixture_sha256,
            "workloads_sha256": sha256(args.workloads),
            "measurement_script_sha256": sha256(Path(__file__).resolve()),
            "production_sources": production,
            "test_helper_source": test_source,
            "sdk_source": sdk_source,
            "source_roots_verified": True,
            "network": "Internet sockets and DNS denied; fixture uses MockTransport/trust_env=False",
            "timings": {
                "core_import_seconds": core_import_seconds,
                "fixture_import_seconds": fixture_import_seconds,
                "fixture_execution_seconds": fixture_execution_seconds,
                "fixture_setup_reporting_cleanup_residual_seconds": max(
                    0.0,
                    fixture_execution_seconds
                    - safe["work"]["construction_seconds"]
                    - safe["elapsed_seconds"],
                ),
                "child_measured_total_seconds": time.perf_counter() - started,
            },
        }
    )
    return safe


def failure(error: BaseException) -> dict:
    result = {"status": "unavailable", "error_type": type(error).__name__}
    if (
        isinstance(error, AttributeError)
        and isinstance(error.name, str)
        and error.name.isidentifier()
    ):
        result["missing_attribute"] = error.name
    result["stack"] = [
        {"file": Path(item.filename).name, "function": item.name, "line": item.lineno}
        for item in traceback.extract_tb(error.__traceback__)
    ]
    return result


def collect(args: argparse.Namespace) -> dict:
    interpreter = args.repository / "runtime/.venv/bin/python"
    require(interpreter.is_file(), "the repository's locked default venv is required")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(args.repository / name) for name in ("runtime/src", "runtime/tests")
    )
    fixture_hash = sha256(args.fixture)
    results = []
    for index in range(args.samples):
        command = [
            str(interpreter),
            str(Path(__file__).resolve()),
            "--repository",
            str(args.repository),
            "--fixture",
            str(args.fixture),
            "--workloads",
            str(args.workloads),
            "--turns",
            str(args.turns),
            "--mode",
            args.mode,
            "--sample",
            "--fixture-sha256",
            fixture_hash,
        ]
        began = time.perf_counter()
        with subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        ) as child:
            try:
                output, _ = child.communicate(timeout=600)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        item = json.loads(output)
        item["sample"] = index + 1
        item["fresh_child_process_wall_seconds"] = time.perf_counter() - began
        results.append(item)
        if child.returncode or item["status"] != "measured":
            break
    return {
        "schema_version": 1,
        "informational_only": True,
        "samples_requested": args.samples,
        "samples_completed": len(results),
        "fixture_sha256": fixture_hash,
        "samples": results,
        "measurement_script_sha256": sha256(Path(__file__).resolve()),
        "timing_notes": [
            "Callback aggregate durations overlap; do not subtract them as disjoint wall time.",
            "Session read_event_copies counts returned event entries, not all internal heap copies.",
            "Framework residual subtracts only the measured MockTransport handler time.",
            "Historical/current production code and locked dependencies both differ; no causal SDK speedup claim.",
            "The Runtime uses its custom BaseLlm object, not the string-model construction path.",
        ],
    }


def main() -> int:
    args = arguments()
    try:
        report = sample(args) if args.sample else collect(args)
    except Exception as error:  # noqa: BLE001 - emit no provider payload or raw traceback
        report = failure(error)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        require(
            not args.output.exists(), "refusing to replace prior measurement evidence"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return (
        0
        if report.get("status") == "measured"
        or (
            report.get("samples_completed") == args.samples
            and all(
                item.get("status") == "measured" for item in report.get("samples", [])
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
