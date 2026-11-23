"""Explicit container representations through the pinned Worker/ADK path."""

import asyncio
import json
import threading

import pytest
from audit_fixtures import accept_fixture_sources, arguments, assigned
from fakes.audit_judgments import acknowledged_judgments
from fakes.tool_arguments import http_receiver, received, run_calls
from fakes.worker_state import WorkerState

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditResultEncoder
from contractor_runtime.toolsets.audit_results.v2 import (
    SubmitResultsTool,
)
from contractor_runtime.toolsets.common.arguments import container_argument
from contractor_runtime.worker.function_tools import ContractorFunctionTool


@pytest.mark.parametrize("route", ["direct", "proxy", "controlled"])
def test_native_and_serialized_http_containers_reach_sockets_once(tmp_path, route):
    async def scenario():
        async with http_receiver(tmp_path, route) as (tool, state, url, requests):
            native = {
                "url": url,
                "method": "POST",
                "body_type": "form",
                "headers": {"X-Note": "unchanged"},
                "query": {"amount": "01"},
                "body": {"amount": "01", "flag": True},
            }
            encoded = {**native, **{k: json.dumps(native[k]) for k in ("headers", "query", "body")}}
            results = await run_calls(tmp_path, tool, state, [native, encoded])
            assert all(result["status"] == 200 for result in results)
            assert "normalization" not in results[0]
            assert results[1]["normalization"] == {
                "fields": [
                    {"field": name, "from_type": "string", "to_type": "object"}
                    for name in ("headers", "query", "body")
                ],
                "omitted": 0,
            }
            assert len(requests) == 2
            for request in requests:
                target, headers, body = received(request)
                assert target.endswith("/allowed?amount=01")
                assert headers["x-note"] == "unchanged"
                assert body == b"amount=01&flag=true"
            assert state.metrics.counters["tool_calls"] == 2
            assert state.metrics.counters.get("tool_errors", 0) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_audit_declared_arrays_record_canonical_results(tmp_path, batch, serialized):
    from test_audit_completion_runtime import assigned as completion_assignment

    async def scenario():
        expected, _ = completion_assignment(batch=batch, invocation="invocation-1")
        values = [arguments(item.value, keyed=True) for item in expected.items]
        for value in values:
            if serialized:
                value["limitations"] = json.dumps(value["limitations"])
        supplied = {"items": json.dumps(values) if serialized else values} if batch else values[0]
        from datetime import UTC, datetime, timedelta

        from fakes.model import text_result, tool_call
        from test_adk_runtime import stage_request
        from test_audit_completion_runtime import runtime_for

        name = "submit_results" if batch else "submit_result"
        worker, _, client, state, _ = await runtime_for(
            tmp_path,
            [
                tool_call(name, supplied, call_id="first"),
                tool_call(name, supplied, call_id="replay"),
                text_result("Done"),
            ],
            batch=batch,
            total_tokens=10000,
        )
        try:
            outcome = await worker.invoke(stage_request())
            assert outcome.failure is None, outcome.failure
            assert len(client.writes) == 1
            result = json.loads(client.writes[0])
            assert [item["rationale"] for item in result["items"]] == [
                item.value.rationale for item in expected.items
            ]
            assert [item["tested_conditions"] for item in result["items"]] == [
                list(item.value.tested_conditions) for item in expected.items
            ]
            assert state.metrics.counters["tool_calls"] == 2
            assert state.metrics.counters.get("tool_errors", 0) == 0
        finally:
            await worker.finalize(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value",
    [
        "null",
        "[]",
        "1",
        '"{}"',
        '{"key":1,"key":2}',
        '{"x":NaN}',
        '{"x":1e999}',
        "{} trailing",
        "{'x': 1}",
        "```json\n{}\n```",
        '{"x":"\\ud800"}',
        "\ud800",
        {"x": float("inf")},
        {"x": "\ud800"},
    ],
)
def test_invalid_http_containers_never_send(tmp_path, value):
    async def scenario():
        async with http_receiver(tmp_path, "direct") as (tool, state, url, requests):
            (result,) = await run_calls(tmp_path, tool, state, [{"url": url, "headers": value}])
            assert result["error"]["code"] == "http_request_invalid"
            assert requests == []

    asyncio.run(scenario())


def test_incremental_byte_depth_node_bounds_and_nested_strings():
    def invalid(field, message, **details):
        return ToolInputError(field + ": " + message)

    def decode(value, **bounds):
        return container_argument(
            value,
            field="payload",
            kinds=(dict, list),
            max_bytes=100,
            max_depth=bounds.get("depth", 3),
            max_nodes=bounds.get("nodes", 6),
            invalid=invalid,
        )

    for value in (["x" * 96], ' ["' + "x" * 93 + '"] '):
        decode(value)
    for value in (["x" * 97], ' ["' + "x" * 95 + '"] ', [[[[0]]]], [0] * 6):
        with pytest.raises(ToolInputError):
            decode(value)
    native = {"text": '{"amount":1}', "amount": "01", "truth": True}
    assert decode(native)[0] is native
    assert decode(json.dumps(native)) == (native, True)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ToolInputError):
        decode(cyclic)


def test_declarations_preserve_closed_records_and_exact_id_types():
    tool = ContractorFunctionTool(
        SubmitResultsTool(None, None), logical_invocation_id=lambda _: "test"
    )
    schema = tool._get_declaration().parameters_json_schema
    assert schema["additionalProperties"] is False
    record = schema["$defs"]["BatchResultArgument"]
    assert record["additionalProperties"] is False
    assert schema["properties"]["items"]["anyOf"][1] == {"type": "string"}
    values = {"items": [{"http_request_ids": [True, 1.0, "1"]}]}
    assert tool._preprocess_args(values) == values
    assert tool.contractor_raw_argument_error({"items": [{"secret-key": 1}]}) is not None


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"http_request_ids": [True]},
        {"http_request_ids": [1.0]},
        {"http_request_ids": ["1"]},
        {"secret-key": "secret-canary"},
        {"limitations": '"[]"'},
        {"evidence": '["evidence-000", "evidence-000"]'},
    ],
)
def test_invalid_audit_batch_has_no_partial_mutation(tmp_path, change, serialized):
    from types import SimpleNamespace

    async def scenario():
        expected, inputs = assigned(batch=True)
        collector = InvocationAuditCollector(
            inputs, CanonicalAuditResultEncoder(), persist=acknowledged_judgments(inputs)
        )
        tool = SubmitResultsTool(collector, WorkerState().metrics)
        items = [arguments(item.value, keyed=True) for item in expected.items]
        items[-1].update(change)
        result = await tool(
            SimpleNamespace(invocation_id="invocation-1"),
            json.dumps(items) if serialized else items,
        )
        assert result["error"]["code"] == "audit_result_invalid"
        assert "secret" not in json.dumps(result)
        assert (await collector.snapshot()).items == ()

    asyncio.run(scenario())


def test_whole_audit_call_is_admitted_before_decoding_individual_fields(monkeypatch):
    from types import SimpleNamespace

    from contractor_runtime.toolsets.audit_results import arguments as owner
    from contractor_runtime.toolsets.audit_results.contracts import MAX_COLLECTED_BYTES

    async def scenario():
        expected, inputs = assigned(count=64)
        collector = InvocationAuditCollector(
            inputs, CanonicalAuditResultEncoder(), persist=acknowledged_judgments(inputs)
        )
        tool = SubmitResultsTool(collector, WorkerState().metrics)
        items = [arguments(item.value, keyed=True) for item in expected.items]
        for item in items:
            item["tested_conditions"] = json.dumps(["x" * 32768] * 8)
            assert len(item["tested_conditions"]) < MAX_COLLECTED_BYTES

        def unexpected(*args, **kwargs):
            raise AssertionError("oversized aggregate reached field decoding")

        monkeypatch.setattr(owner, "container_argument", unexpected)
        result = await tool(SimpleNamespace(invocation_id="invocation-1"), items)
        assert result["error"]["field"] == "arguments"
        assert (await collector.snapshot()).items == ()

    asyncio.run(scenario())


def test_real_structural_limits_native_and_serialized():
    from contractor_runtime.toolsets.common.arguments import (
        MAX_CONTAINER_DEPTH,
        MAX_CONTAINER_NODES,
    )

    def decode(value):
        return container_argument(
            value,
            field="body",
            kinds=(list,),
            max_bytes=1 << 20,
            max_depth=MAX_CONTAINER_DEPTH,
            max_nodes=MAX_CONTAINER_NODES,
            invalid=lambda field, message, **details: ToolInputError(field + ": " + message),
        )

    nested = []
    for _ in range(MAX_CONTAINER_DEPTH - 1):
        nested = [nested]
    wide = [0] * (MAX_CONTAINER_NODES - 1)
    for value in (nested, wide):
        for representation in (value, json.dumps(value)):
            decode(representation)
    for value in ([nested], [*wide, 0]):
        for representation in (value, json.dumps(value)):
            with pytest.raises(ToolInputError):
                decode(representation)


def test_cancelled_parsing_joins_before_return(monkeypatch, tmp_path):
    from fakes.tool_arguments import http_receiver

    from contractor_runtime.toolsets.http import tools as owner

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = owner.request_containers

    def blocked(*args):
        entered.set()
        release.wait(5)
        try:
            return original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(owner, "request_containers", blocked)

    async def scenario():
        async with http_receiver(tmp_path, "direct") as (tool, _state, url, requests):
            task = asyncio.create_task(tool(url, headers="{}"))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert finished.is_set() and not requests
            finally:
                release.set()
                await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "proxy", "controlled"])
def test_http_field_error_then_correction_is_counted_once(tmp_path, route):
    async def scenario():
        async with http_receiver(tmp_path, route) as (tool, state, url, requests):
            results = await run_calls(
                tmp_path,
                tool,
                state,
                [
                    {"url": url, "headers": '{"secret-canary":'},
                    {"url": url, "headers": '{"X-Note":"secret-canary"}'},
                ],
            )
            error = results[0]["error"]
            assert error["field"] == "headers"
            assert error["expected_type"] == "object"
            assert error["actual_type"] == "string"
            assert error["code"] == "http_request_invalid" and not error["retryable"]
            assert "Provide one strict JSON object" in error["message"]
            assert "secret-canary" not in json.dumps(error)
            assert "secret-canary" not in json.dumps(results[1]["normalization"])
            assert len(requests) == 1 and results[1]["status"] == 200
            assert state.metrics.counters["tool_calls"] == 2
            assert state.metrics.counters["tool_errors"] == 1
            assert "secret-canary" not in repr(state.metrics.tool_calls)
            report = state.metrics.build_report(report_id="arguments", duration_ms=1)
            assert report.metrics.worker_budget.observed_tool_calls == 2

    asyncio.run(scenario())


def test_audit_preserves_inner_field_diagnostic_through_runner_correction(tmp_path):
    from datetime import UTC, datetime, timedelta

    from fakes.model import text_result, tool_call
    from test_adk_runtime import stage_request
    from test_audit_completion_runtime import assigned as completion_assignment
    from test_audit_completion_runtime import runtime_for

    async def scenario():
        expected, _ = completion_assignment(batch=True)
        good = [arguments(item.value, keyed=True) for item in expected.items]
        bad = [good[0], {**good[1], "limitations": '["secret-canary"'}]
        worker, _, client, state, _ = await runtime_for(
            tmp_path,
            [
                tool_call("submit_results", {"items": json.dumps(bad)}, call_id="invalid"),
                tool_call("submit_results", {"items": json.dumps(good)}, call_id="fixed"),
                text_result("Done"),
            ],
            batch=True,
            total_tokens=10000,
        )
        try:
            assert (await worker.invoke(stage_request())).failure is None
            session = await worker._session_service.get_session(
                app_name=worker._app_name,
                user_id=worker._user_id,
                session_id=worker._session_lifecycle.shared_session_id,
            )
            results = [
                part.function_response.response
                for event in session.events
                for part in (event.content.parts if event.content else []) or []
                if part.function_response is not None
            ]
            error = results[0]["error"]
            assert error["field"] == "items[1].limitations"
            assert error["expected_type"] == "array" and error["actual_type"] == "string"
            assert error["code"] == "audit_result_invalid" and not error["retryable"]
            assert "secret-canary" not in json.dumps(error)
            assert results[1]["complete"] and len(client.writes) == 1
            assert state.metrics.counters["tool_calls"] == 2
            assert state.metrics.counters["tool_errors"] == 1
        finally:
            await worker.finalize(datetime.now(UTC) + timedelta(seconds=5))

    asyncio.run(scenario())


def test_normalization_notices_are_bounded_and_never_contain_values():
    from types import SimpleNamespace

    from contractor_runtime.toolsets.common.argument_diagnostics import MAX_NORMALIZATION_NOTICES

    async def scenario():
        expected, inputs = assigned(count=64)
        collector = InvocationAuditCollector(
            inputs, CanonicalAuditResultEncoder(), persist=acknowledged_judgments(inputs)
        )
        await accept_fixture_sources(collector)
        tool = SubmitResultsTool(collector, WorkerState().metrics)
        items = [arguments(item.value, keyed=True) for item in expected.items]
        for item in items:
            item["limitations"] = json.dumps(["secret-canary"])
        result = await tool(SimpleNamespace(invocation_id="invocation-1"), json.dumps(items))
        assert result["complete"]
        notice = result["normalization"]
        assert len(notice["fields"]) == MAX_NORMALIZATION_NOTICES
        assert notice["omitted"] == 1 + 64 - MAX_NORMALIZATION_NOTICES
        assert "secret-canary" not in json.dumps(notice)

    asyncio.run(scenario())
