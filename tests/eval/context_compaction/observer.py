"""Evaluator-private view of existing OTLP and controlled SDK projection evidence.

CONTRACTOR_CONTEXT_EVAL_EVIDENCE points outside Runtime workspaces. No credentials,
model answers or private scoring truth are read or emitted here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# The consumer launches this script with -I -S, which excludes its directory
# from sys.path. Load the pinned sibling client explicitly, without ambient
# PYTHONPATH/site-package imports or a second API client implementation.
_client_spec = importlib.util.spec_from_file_location(
    "context_compaction_client", Path(__file__).with_name("harness.py")
)
if _client_spec is None or _client_spec.loader is None:
    raise RuntimeError("evaluator client unavailable")
h = importlib.util.module_from_spec(_client_spec)
_client_spec.loader.exec_module(h)

MAX_BYTES = 16 * 1024 * 1024
MAX_SPANS = 4096
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def attributes(items):
    return {item["key"]: item["value"].get("stringValue") for item in items}


def observe(directory, run_id, *, planner_stages=(), projection_requests=None):
    require(ID.fullmatch(run_id), "invalid Run identity")
    directory = Path(directory)
    require(not (directory / "collector-gap").exists(), "collector coverage unavailable")
    folder = directory / "traces" / run_id
    files = sorted(folder.glob("*.json"))
    require(0 < len(files) <= MAX_SPANS, "trace evidence unavailable")
    spans, stages, size = {}, {}, 0
    for path in files:
        size += path.stat().st_size
        require(size <= MAX_BYTES, "Run trace evidence exceeds bound")
        document = json.loads(path.read_bytes())
        for resource in document.get("resourceSpans", []):
            labels = attributes(resource["resource"]["attributes"])
            require(labels.get("contractor.run.id") == run_id, "foreign trace evidence")
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    identity = (span["traceId"], span["spanId"])
                    require(identity not in spans or spans[identity] == span, "conflicting span")
                    spans[identity] = span
                    stages[identity] = labels.get("contractor.stage_execution.id")
                    require(len(spans) <= MAX_SPANS, "Run spans exceed bound")
    tools, checkpoints, retrieved, effects, planners = [], [], set(), [], set()
    for identity, span in spans.items():
        values = attributes(span.get("attributes", []))
        if (
            span["name"] == "contractor.planner.invocation"
            and int(span["endTimeUnixNano"]) > int(span["startTimeUnixNano"])
            and span.get("status", {}).get("code") in {"STATUS_CODE_OK", "STATUS_CODE_ERROR"}
        ):
            planners.add(stages[identity])
        if span.get("status", {}).get("code") != "STATUS_CODE_OK":
            continue
        phase = values.get("model.phase")
        if values.get("operation.kind") == "context_compaction" or (
            span["name"] == "contractor.worker.model"
            and (
                phase == "compaction"
                or (
                    phase is None
                    and str(values.get("model.alias", "")).endswith("/summarizer-model")
                )
            )
        ):
            checkpoints.append(int(span["endTimeUnixNano"]))
        if span["name"] != "contractor.worker.tool":
            continue
        name = values.get("tool.name")
        require(
            name in {"read_artifact", "write_text_artifact", "read_context_history"},
            "unmeasured tool capability",
        )
        argument = json.loads(values["contractor.input"])
        output = json.loads(values["contractor.output"])
        require(
            all(
                not (
                    isinstance(value, dict)
                    and set(value) == {"truncated", "preview"}
                    and value["truncated"] is True
                )
                for value in (argument, output)
            ),
            "telemetry content truncated",
        )
        if output.get("ok") is False:
            continue
        tools.append(span)
        if name == "read_artifact":
            require(isinstance(output.get("text"), str), "source observation missing")
            retrieved.update(re.findall(r"\bobs-P[0-9]+\b", output["text"]))
        elif name == "write_text_artifact":
            # Identical repeated writes are duplicate effects; an updated ledger
            # with different content is a distinct operation.
            encoded = json.dumps(argument, sort_keys=True, separators=(",", ":")).encode()
            effects.append(hashlib.sha256(encoded).hexdigest())
    require(set(planner_stages).issubset(planners), "Planner invocation trace unavailable")
    projection = directory / "projections" / (run_id + ".jsonl")
    if projection_requests is not None:
        require(projection.exists(), "controlled request projection unavailable")
    if projection.exists():
        require(projection.stat().st_size <= 2 * 1024 * 1024, "projection evidence exceeds bound")
        lines = projection.read_text().splitlines()
        if projection_requests is not None:
            require(len(lines) == projection_requests, "controlled request coverage incomplete")
        for line in lines:
            record = json.loads(line)
            require(record["runId"] == run_id, "foreign controlled evidence")
            if record["questionSha256"] and (
                record["nativeGeneration"] > 0 or record["maskedResponses"] > 0
            ):
                checkpoints.append(record["recordedAtNs"])
    first = min(checkpoints) if checkpoints else None
    return {
        "mutations": 0,  # All admitted tools operate on artifacts, not source/environment.
        "effects": effects,
        "usefulAfterCheckpoint": sum(
            first is not None and int(tool["startTimeUnixNano"]) > first for tool in tools
        ),
        "retrieved": sorted(retrieved),
        "newProbes": [],  # No HTTP/probe capability is admitted; no choice claim is made.
    }


def verify_export(diagnostics, allocations, *, has_planner):
    require(bool(diagnostics), "Runtime export evidence unavailable")
    declared = {item["logicalAgent"]: item for item in allocations}
    require(len(declared) == len(allocations), "duplicate allocation provenance")
    require(
        len(diagnostics) == len(declared)
        and {row["logicalAgent"] for row in diagnostics} == set(declared),
        "allocation report coverage unavailable",
    )
    needs_planner_trace = False
    for row in diagnostics:
        require(
            row.get("status") == "available" and row.get("complete"), "Runtime report incomplete"
        )
        allocation = declared[row["logicalAgent"]]
        if not allocation["runtimeAdapters"]:
            origins = allocation["origins"]
            require(
                has_planner
                and "plannerTelemetry" in origins
                and "workerTelemetry" not in origins
                and row["adapters"] == [],
                "allocation has no declared trace exporter",
            )
            needs_planner_trace = True
            continue
        require("otlp-http@1" in allocation["runtimeAdapters"], "Worker OTLP not declared")
        adapters = [a for a in row["adapters"] if a["ref"] == "otlp-http@1"]
        require(len(adapters) == 1, "Runtime OTLP exporter unavailable")
        adapter = adapters[0]
        require(
            adapter.get("flushSucceeded") is True
            and adapter.get("failedOperations") == "0"
            and bool(adapter.get("droppedSpans"))
            and all(value == "0" for value in adapter["droppedSpans"].values()),
            "Runtime OTLP export has gaps",
        )
    return needs_planner_trace


def main():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "observer input exceeds bound")
    request = json.loads(raw)
    client = h.Client(
        os.environ["CONTRACTOR_CONTEXT_EVAL_API"], os.environ["CONTRACTOR_CONTEXT_EVAL_TOKEN"]
    )
    run_id = request["runId"]
    require(ID.fullmatch(run_id), "invalid Run identity")
    run = client.request("GET", "/v1/runs/" + run_id)
    require(run["state"] in h.TERMINAL and run["deletable"], "Run has not released")
    planner_stages = set()
    controlled = run["workflow"].split("@", 1)[0] in {
        "context-eval-controlled-compact",
        "context-eval-masking",
    }
    projection_requests = 0 if controlled else None
    for attempt in run["attempts"]:
        diagnostics = client.pages(
            "/v1/runs/"
            + run_id
            + "/attempts/"
            + attempt["stageExecutionId"]
            + "/runtime-diagnostics",
            limit=10,
        )
        if verify_export(
            diagnostics,
            attempt["runtimeConfiguration"]["allocations"],
            has_planner="planner" in attempt["executionConfig"],
        ):
            planner_stages.add(attempt["stageExecutionId"])
        if controlled:
            for row in diagnostics:
                phases = row.get("modelPhases", {})
                require("main" in phases, "controlled main-model accounting unavailable")
                projection_requests += phases["main"]["modelCalls"]
    result = observe(
        os.environ["CONTRACTOR_CONTEXT_EVAL_EVIDENCE"],
        run_id,
        planner_stages=planner_stages,
        projection_requests=projection_requests,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
