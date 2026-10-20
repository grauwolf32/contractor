"""Content-free completion facts through real Runtime reporting and shared DTO data."""

import asyncio
import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fakes.model import text_result
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource
from test_adk_runtime import stage_request
from test_allocation import make_service, make_spec
from test_audit_completion_runtime import runtime_for, submit
from test_audit_result_publication import assigned

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError, ArtifactValue
from contractor_runtime.contracts import (
    ExecutionReport,
    WorkerAllocationMetricsState,
    WorkerCompletionDiagnostics,
)
from contractor_runtime.server import create_app
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.worker.sessions import WorkerSessionLifecycleError

ROOT = Path(__file__).parents[2]
CASES = json.loads((ROOT / "api/testdata/audit-completion/diagnostics.json").read_text())
SCHEMAS = {
    path.name: json.loads(path.read_text(encoding="utf-8"))
    for path in (ROOT / "api/v1alpha1").glob("*.schema.json")
}
REGISTRY = Registry().with_resources(
    (schema["$id"], Resource.from_contents(schema)) for schema in SCHEMAS.values()
)


def _schema(filename, definition):
    reference = f"{SCHEMAS[filename]['$id']}#/$defs/{definition}"
    return Draft202012Validator({"$ref": reference}, registry=REGISTRY)


def _shared_document(case, live):
    if live:
        raw = {
            "counters": {},
            "toolCalls": [],
            "errors": [],
            "finalOutcome": None,
            "truncated": False,
            "completion": case["value"],
        }
        return (
            WorkerAllocationMetricsState,
            raw,
            _schema("agent-state.schema.json", "allocationMetrics"),
        )
    raw = {
        "reportId": "report-1",
        "complete": True,
        "metrics": {"tools": {}},
        "toolCalls": [],
        "errors": [],
        "truncated": False,
        "completion": case["value"],
    }
    return ExecutionReport, raw, _schema("allocation.schema.json", "executionReport")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("live", [False, True], ids=["report", "live-state"])
def test_shared_completion_diagnostics_contract(case, live):
    # The models, the JSON Schema and Go (TestSharedWorkerCompletionDiagnostics)
    # give each case one verdict; an unknown kind or phase rejects the whole
    # report or live State instead of being dropped (AWC-R115).
    model, raw, schema = _shared_document(case, live)
    # JSON Schema cannot compare counts across fields; only those cases are
    # schema-valid although every codec rejects them.
    schema_valid = not list(schema.iter_errors(raw))
    assert schema_valid == (case["valid"] or case.get("semanticOnly", False))
    encoded = json.dumps(raw)
    if padding := case.get("jsonWhitespaceBytes", 0):
        encoded = encoded.replace('"completion": {', '"completion": {' + " " * padding, 1)
    if not case["valid"]:
        with pytest.raises(ValidationError):
            model.model_validate(raw)
        with pytest.raises(ValidationError):
            model.model_validate_json(encoded)
        return
    for decoded in (model.model_validate(raw), model.model_validate_json(encoded)):
        assert decoded.completion is not None
        assert decoded.completion.model_dump(by_alias=True, exclude_none=True) == case["value"]


def test_shared_completion_diagnostics_reject_unknown_kind_and_phase():
    unknown = {case["name"]: case for case in CASES if case["name"].startswith("unknown-")}
    assert {"unknown-kind", "unknown-phase"} <= unknown.keys()
    assert not any(case["valid"] for case in unknown.values())
    for case in unknown.values():
        # The producer revalidates what it records, so even an unvalidated
        # object never reaches a report.
        metrics = MetricsState()
        with pytest.raises(ValidationError):
            metrics.record_completion(WorkerCompletionDiagnostics.model_construct(**case["value"]))
        assert metrics.build_report(report_id="report-1", duration_ms=0).completion is None


def test_shared_completion_diagnostics_gate_the_agent_state_endpoint(
    tmp_path, runtime_capabilities
):
    # The Runtime validates every State snapshot at its wire boundary. A
    # completion object that fails the contract makes the endpoint answer 503
    # agent_state_unavailable; it is never served with the object removed.
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        worker_state = service._context.worker_state
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            for case in CASES:
                worker_state._state["metrics"]["completion"] = copy.deepcopy(case["value"])
                response = await client.get("/private/v1/allocations/allocation-1/agent-state")
                if case["valid"]:
                    assert response.status_code == 200, case["name"]
                    served_snapshot = response.json()
                    validator = Draft202012Validator(
                        SCHEMAS["agent-state.schema.json"], registry=REGISTRY
                    )
                    assert not list(validator.iter_errors(served_snapshot)), case["name"]
                    served = served_snapshot["state"]["metrics"]["completion"]
                    assert served == case["value"], case["name"]
                else:
                    assert response.status_code == 503, case["name"]
                    assert response.json()["code"] == "agent_state_unavailable", case["name"]
                    assert response.json()["retryable"] is False, case["name"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode", ["missing", "partial", "invalid", "published", "forbidden", "transport", "conflict"]
)
def test_shared_real_runtime_reports_counts_phases_and_no_fictitious_calls(tmp_path, mode):
    async def scenario():
        expected, _ = assigned(batch=True)
        items = [item.value for item in expected.items]
        if mode == "missing":
            responses = [text_result("Finished")] * 3
        elif mode in {"partial", "invalid"}:
            value = replace(items[0], evidence=()) if mode == "invalid" else items[0]
            responses = [submit(value)] + [text_result("Finished")] * 3
        else:
            responses = [submit(items[0]), submit(items[1], "second"), text_result("Finished")]
        runtime, model, client, state, _ = await runtime_for(tmp_path, responses, batch=True)
        if mode == "forbidden":
            client.write_error = ArtifactAPIError(403, "allocation_write_fenced", False)
        elif mode in {"transport", "conflict"}:
            client.write_error = ArtifactTransportError("private-transport-marker")
            original_read = client.read_artifact

            async def read_publication(ref, *, max_bytes):
                if ref.namespace == "inputs":
                    return await original_read(ref, max_bytes=max_bytes)
                if mode == "transport":
                    raise ArtifactTransportError("private-transport-marker")
                return ArtifactValue(
                    ref.model_copy(update={"revision": "conflicting-r1"}),
                    "application/zip",
                    b"different bytes",
                    datetime.now(UTC),
                    datetime.now(UTC),
                )

            client.read_artifact = read_publication
        phases = []
        original = runtime._completion.phase_sink

        async def observe(value):
            counters = dict(state.metrics.counters)
            await original(value)
            assert state.metrics.counters == counters
            phases.append(value.phase)
            assert (await state.agent_state_snapshot()).state.metrics.completion == value

        runtime._completion.phase_sink = observe
        result = await runtime.invoke(stage_request())
        report = state.metrics.build_report(report_id="report-1", duration_ms=1)
        diagnostic = report.completion
        assert diagnostic is not None
        assert diagnostic.kind == "audit-check-results@1" and diagnostic.total_count == 2
        assert (
            diagnostic.accepted_count
            == {
                "missing": 0,
                "partial": 1,
                "invalid": 0,
                "published": 2,
                "forbidden": 2,
                "transport": 2,
                "conflict": 2,
            }[mode]
        )
        assert diagnostic.phase == ("published" if mode == "published" else "failed")
        assert diagnostic.failure_code == (result.failure.code if result.failure else None)
        assert diagnostic.reminder_count == (2 if mode in {"missing", "partial", "invalid"} else 0)
        assert report.metrics.model_calls == len(model.requests)
        assert not state.metrics.counters.get("llm_errors", 0)
        if mode == "published":
            assert phases == ["sealed", "publishing", "published"]
        if mode in {"forbidden", "transport", "conflict"}:
            assert phases == ["sealed", "publishing", "failed"]
            assert result.failure.retryable == (mode == "transport")
            assert result.failure.code == (
                "audit_result_publication_conflict"
                if mode == "conflict"
                else "audit_result_publication_failed"
            )
            assert len(client.writes) == (2 if mode == "transport" else 1)
        text = report.model_dump_json(by_alias=True, exclude_none=True)
        assert items[0].summary not in text and items[0].evidence[0].summary not in text
        assert "itemKey" not in text and "acceptedCoverage" not in text
        assert "private-transport-marker" not in text
        # Report and live-state copies cannot mutate retained terminal facts.
        diagnostic.accepted_count = 63
        assert (
            state.metrics.build_report(report_id="again", duration_ms=1).completion.accepted_count
            != 63
        )
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_shared_cancelled_collection_reports_partial_data_without_publication(tmp_path):
    async def scenario():
        expected, _ = assigned(batch=True)
        runtime, model, client, state, _ = await runtime_for(
            tmp_path,
            [submit(expected.items[0].value), text_result("partial"), text_result("done")],
            batch=True,
            block_call=3,
        )
        task = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(model.blocked.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        diagnostic = state.metrics.build_report(report_id="report-1", duration_ms=1).completion
        assert diagnostic == WorkerCompletionDiagnostics(
            kind="audit-check-results@1",
            phase="failed",
            acceptedCount=1,
            totalCount=2,
            reminderCount=1,
            failureCode="worker_cancelled",
        )
        assert not client.writes
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_shared_ordinary_reports_omit_completion_and_shared_reuse_resets_it(tmp_path):
    async def scenario():
        old = MetricsState().build_report(report_id="old", duration_ms=0)
        assert "completion" not in old.model_dump(by_alias=True)
        expected, _ = assigned()
        runtime, _, _, state, _ = await runtime_for(
            tmp_path,
            [submit(expected.items[0].value), text_result("done")] + [text_result("done")] * 3,
        )
        assert (await runtime.invoke(stage_request())).result is not None
        assert (
            state.metrics.build_report(report_id="first", duration_ms=1).completion.phase
            == "published"
        )
        assert (await runtime.invoke(stage_request())).failure.code == "audit_result_incomplete"
        assert (
            state.metrics.build_report(report_id="second", duration_ms=1).completion.accepted_count
            == 0
        )
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["state_sync", "session_release", "cancel_release"])
def test_shared_post_publication_cleanup_failure_overrides_completion_diagnostics(
    tmp_path, monkeypatch, boundary
):
    async def scenario():
        expected, _ = assigned()
        runtime, _, client, state, _ = await runtime_for(
            tmp_path, [submit(expected.items[0].value), text_result("done")]
        )
        entered, proceed = asyncio.Event(), asyncio.Event()
        release = runtime._session_lifecycle.finish_invocation

        async def fail_sync(*args):
            raise WorkerSessionLifecycleError("state_sync_failed")

        async def delayed_release():
            entered.set()
            await proceed.wait()
            await release()
            if boundary == "session_release":
                raise WorkerSessionLifecycleError("delete_failed")

        if boundary == "state_sync":
            monkeypatch.setattr(runtime, "_sync_worker_state", fail_sync)
        else:
            monkeypatch.setattr(runtime._session_lifecycle, "finish_invocation", delayed_release)
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        if boundary != "state_sync":
            await asyncio.wait_for(entered.wait(), 2)
            if boundary == "cancel_release":
                invocation.cancel()
            proceed.set()
        if boundary == "cancel_release":
            with pytest.raises(asyncio.CancelledError):
                await invocation
            code = "worker_cancelled"
        else:
            result = await invocation
            assert result.result is None
            code = result.failure.code
            assert code == "worker_session_lifecycle_failed"
            assert result.state_revision == (await state.snapshot())["stateRevision"]
        assert len(client.writes) == 1
        diagnostic = state.metrics.build_report(report_id="cleanup", duration_ms=1).completion
        assert diagnostic.phase == "failed" and diagnostic.failure_code == code
        assert diagnostic.accepted_count == diagnostic.total_count == 1
        assert (await state.agent_state_snapshot()).state.metrics.completion == diagnostic
        assert not state.metrics.counters.get("llm_errors", 0)
        assert state.begins == state.completions == 1
        assert not runtime._invoke_lock.locked()
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
