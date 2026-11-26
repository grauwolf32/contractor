"""Exercise the custom HTTP adapter through real ADK roles and serialized OTLP."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fakes.worker_state import WorkerState
from google.adk.tools import ToolContext
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from test_adk_runtime import (
    RecordingInstrumentation,
    assert_slot_released,
    create_runtime,
    stage_request,
)
from test_agent_skill_toolset import FakeSkillClient, resolved_value, skill_package
from test_agent_skill_toolset import build_context as skill_context
from test_gateway_recovery import Authority
from test_otlp_adapter import HEADER_SECRET, adapter_context, telemetry_settings
from test_worker_compaction import COMPACTOR_MODEL, compact_context

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.content import MAX_CONTENT_BYTES
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.core.tool_metrics import RecordedToolCall
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory

GATEWAY_CREDENTIAL = "gateway-credential-conformance-canary"
GATEWAY_URL = "https://private-gateway-conformance.example/v1"
REASONING = "reasoning-conformance-canary"
RESULT = "result-content-conformance-canary"
INSTRUCTION = "Preserve {contractor} and {unset_name} literally. instruction-content-canary"


def chat(message, *, finish_reason="stop"):
    return {
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            # Reasoning is a subset of completion, not another three tokens.
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
        "provider_extension": {"opaque": "provider-extension-conformance-canary"},
    }


def tool_reply():
    return chat(
        {
            "content": None,
            "reasoning_content": REASONING,
            "tool_calls": [
                {
                    "id": "owned-probe-call",
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "arguments": json.dumps(
                            {"required_text": "argument-content-canary", "optional_note": None}
                        ),
                    },
                }
            ],
        },
        finish_reason="tool_calls",
    )


def structured_reply():
    return chat({"content": json.dumps({"subtaskId": "0", "result": RESULT})})


def gateway_model(http, *, summary=False, recovery=None, model=None):
    return OpenAICompatibleGatewayLlm(
        model=model or ("worker-summary-model" if summary else "worker-model"),
        client_handle=new_gateway_client(
            base_url=GATEWAY_URL,
            api_key=GATEWAY_CREDENTIAL,
            timeout_seconds=10,
            http_client=http,
            recovery=recovery,
        ),
    )


def assert_structured_payload(payload):
    assert "tools" not in payload and "tool_choice" not in payload
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"subtaskId", "result"}
    assert set(schema["required"]) == {"subtaskId", "result"}
    assert schema["additionalProperties"] is False


def model_span_attributes(exports):
    spans = [
        span
        for payload in exports
        for resource in ExportTraceServiceRequest.FromString(payload).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
        if span.name == "contractor.worker.model"
    ]
    result = []
    for span in sorted(spans, key=lambda span: span.start_time_unix_nano):
        attributes = {attribute.key: attribute.value for attribute in span.attributes}
        assert len(attributes) == len(span.attributes)
        result.append(attributes)
    return result


def assert_effective_input(attributes, payload):
    messages = []
    instructions = []
    for message in payload["messages"]:
        role = message["role"]
        if role == "system":
            instructions.append({"type": "text", "content": message["content"]})
            continue
        parts = []
        if role == "tool":
            parts.append(
                {
                    "type": "tool_call_response",
                    "id": message["tool_call_id"],
                    "response": json.loads(message["content"]),
                }
            )
        elif message["content"] is not None:
            parts.append({"type": "text", "content": message["content"]})
        parts.extend(
            {
                "type": "tool_call",
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for call in message.get("tool_calls", [])
        )
        messages.append({"role": role, "parts": parts})
    assert json.loads(attributes["gen_ai.input.messages"].string_value) == messages
    assert json.loads(attributes["gen_ai.system_instructions"].string_value) == instructions
    assert json.loads(attributes["gen_ai.tool.definitions"].string_value) == [
        {"type": "function", **tool["function"]} for tool in payload["tools"]
    ]


def finish_reasons(attributes):
    value = attributes.get("gen_ai.response.finish_reasons")
    return [] if value is None else [item.string_value for item in value.array_value.values]


@pytest.mark.parametrize("compact", [False, True], ids=["ordinary", "checkpoint"])
def test_main_gateway_trace_matches_effective_request(tmp_path, compact):
    main_requests, exports, lookups = [], [], []

    async def collector(request):
        exports.append(await request.aread())
        return httpx.Response(200, json={})

    async def gateway(request):
        payload = json.loads(request.content)
        if payload["model"] == COMPACTOR_MODEL:
            return httpx.Response(200, json=chat({"content": '{"summary":"Saved earlier work"}'}))
        if "response_format" in payload:
            return httpx.Response(200, json=structured_reply())
        main_requests.append(payload)
        ordinal = len(main_requests)
        if ordinal == 4:
            return httpx.Response(200, json=chat({"content": RESULT}))
        reply = tool_reply()
        call = reply["choices"][0]["message"]["tool_calls"][0]
        call["id"] = f"lookup-{ordinal}"
        call["function"].update(
            name="lookup", arguments=json.dumps({"key": f"lookup-key-{ordinal}"})
        )
        if ordinal == 2:
            reply["usage"].update(prompt_tokens=600, total_tokens=607)
        return httpx.Response(200, json=reply)

    async def lookup(key: str) -> dict:
        """Look up a fixture value."""
        lookups.append(key)
        return {"value": key}

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings()
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            state = WorkerState()
            if compact:
                context = replace(
                    compact_context(tmp_path, state, {"lookup": lookup}),
                    instruction=INSTRUCTION,
                    adapter_handles=adapter.handles,
                )
                runtime = await AdkWorkerRuntimeFactory(
                    lambda build: gateway_model(http, model=build.model_policy.model)
                ).create(context)
            else:
                runtime = await create_runtime(
                    tmp_path,
                    state,
                    {"lookup": lookup},
                    gateway_model(http),
                    instruction=INSTRUCTION,
                    instrumentation=adapter.handles.instrumentation,
                )
            try:
                completion = await runtime.invoke(stage_request())
                assert completion.failure is None, completion.failure
                assert completion.result.result == RESULT
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    assert lookups == ["lookup-key-1", "lookup-key-2", "lookup-key-3"]
    spans = model_span_attributes(exports)
    main_spans = [attributes for attributes in spans if "gen_ai.tool.definitions" in attributes]
    auxiliary = [attributes for attributes in spans if "gen_ai.tool.definitions" not in attributes]
    assert len(auxiliary) == 1 + int(compact)
    assert all(not finish_reasons(attributes) for attributes in auxiliary)
    assert len(main_requests) == len(main_spans) == 4
    for index, (attributes, request) in enumerate(zip(main_spans, main_requests, strict=True)):
        if compact and index >= 2:
            assert "Context checkpoint 1" in request["messages"][1]["content"]
            assert "lookup-key-1" not in json.dumps(request)
            assert "lookup-key-2" in json.dumps(request)
            assert "Context checkpoint 1" in attributes["gen_ai.input.messages"].string_value
    for attributes, request in zip(main_spans, main_requests, strict=True):
        assert_effective_input(attributes, request)
    for index, attributes in enumerate(main_spans):
        reason = "stop" if index == 3 else "tool_calls"
        assert finish_reasons(attributes) == [reason]
        output = json.loads(attributes["gen_ai.output.messages"].string_value)
        assert output[0]["finish_reason"] == reason


@pytest.mark.parametrize(
    "case", ["length", "malformed", "omitted", "transport", "missing", "unknown"]
)
def test_main_gateway_finish_reason_survives_output_failure_or_truncation(tmp_path, case):
    exports, lookups = [], []

    async def collector(request):
        exports.append(await request.aread())
        return httpx.Response(200, json={})

    async def gateway(request):
        if case == "transport":
            raise httpx.ReadTimeout("no response", request=request)
        if "response_format" in json.loads(request.content):
            return httpx.Response(200, json=structured_reply())
        reply = tool_reply()
        choice = reply["choices"][0]
        if case == "length":
            choice["finish_reason"] = "length"
            choice["message"]["content"] = "x" * (MAX_CONTENT_BYTES + 1)
        elif case == "malformed":
            choice["message"]["tool_calls"][0]["function"]["arguments"] = "{"
        elif case == "omitted":
            choice["finish_reason"] = "content_filter"
            choice["message"]["tool_calls"][0]["function"]["arguments"] = (
                '{"nested":' + "[" * 65 + '"value"' + "]" * 65 + "}"
            )
        elif case == "missing":
            choice = {"message": {"content": RESULT}}
            reply["choices"] = [choice]
        else:
            choice["finish_reason"] = f"vendor-{GATEWAY_CREDENTIAL}-" + "é" * 300
        return httpx.Response(200, json=reply)

    async def probe(required_text: str, optional_note: str | None) -> dict:
        """Read a fixture value."""
        lookups.append(required_text)
        return {}

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            replace(adapter_context(), known_credentials=(GATEWAY_CREDENTIAL, "tool_calls")),
            telemetry_settings(),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            runtime = await create_runtime(
                tmp_path,
                WorkerState(),
                {"probe": probe},
                gateway_model(http),
                instrumentation=adapter.handles.instrumentation,
            )
            try:
                completion = await runtime.invoke(stage_request())
                if case == "missing":
                    assert completion.failure is None
                else:
                    assert completion.failure is not None
                    if case == "length":
                        assert completion.failure.code == "worker_output_limit_exceeded"
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    assert lookups == []
    assert GATEWAY_CREDENTIAL.encode() not in b"".join(exports)
    attributes = model_span_attributes(exports)[0]
    reasons = finish_reasons(attributes)
    if case in {"malformed", "transport"}:
        assert "gen_ai.output.messages" not in attributes
        assert reasons == {"malformed": ["tool_calls"], "omitted": ["content_filter"]}.get(case, [])
    else:
        output = json.loads(attributes["gen_ai.output.messages"].string_value)[0]
        if case == "length":
            assert reasons == ["length"]
            assert output["parts"][0]["truncated"] is True
        elif case == "omitted":
            # Safety classification discards the malformed call before parsing.
            assert reasons == ["content_filter"]
        elif case == "missing":
            assert reasons == []
        else:
            expected = ("vendor-[REDACTED]-" + "é" * 300).encode()[:256].decode(errors="ignore")
            assert reasons == [expected]
        assert output["finish_reason"] == (reasons[0] if reasons else "unknown")
    if case != "transport":
        assert attributes["gen_ai.usage.input_tokens"].int_value == 11
        assert attributes["gen_ai.usage.output_tokens"].int_value == 7


@pytest.mark.parametrize("summary", [False, True], ids=["ordinary", "terminal"])
def test_gateway_error_with_tool_call_stops_ordinary_worker_before_side_effects(tmp_path, summary):
    effects, payloads = [], []

    async def gateway(request):
        payloads.append(json.loads(request.content))
        response = tool_reply()
        response["choices"][0]["finish_reason"] = "content_filter"
        return httpx.Response(200, json=response)

    async def probe(required_text: str, optional_note: str | None) -> dict:
        """Record an effect only when the model response is admitted."""
        effects.append((required_text, optional_note))
        return {"ok": True}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            state = WorkerState()
            runtime = await create_runtime(
                tmp_path,
                state,
                {"probe": probe},
                gateway_model(http),
                summary_model=gateway_model(http, summary=True) if summary else None,
                cumulative_budget=18 if summary else None,
            )
            try:
                outcome = await runtime.invoke(stage_request())
                assert outcome.result is None and outcome.failure.code == "worker_execution_failed"
                assert effects == [] and len(payloads) == 1
                report = state.metrics.build_report(report_id="error-tools", duration_ms=1)
                assert report.metrics.total_tokens == 18
                assert state.metrics.counters["llm_errors"] == 1
                assert report.metrics.worker_budget.observed_tool_calls == 0
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("summary", [False, True], ids=["finalizer", "summarizer"])
@pytest.mark.parametrize(("capture", "sensitive"), [(False, False), (True, False), (True, True)])
def test_real_gateway_roles_preserve_schemas_reasoning_and_serialized_capture_policy(
    tmp_path, caplog, summary, capture, sensitive
):
    payloads, exports, calls = [], [], []
    provider = trace.get_tracer_provider()

    async def collector(request):
        assert request.headers["authorization"] == f"Bearer {HEADER_SECRET}"
        exports.append(await request.aread())
        return httpx.Response(200, json={})

    async def gateway(request):
        assert request.headers["authorization"] == f"Bearer {GATEWAY_CREDENTIAL}"
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["model"] == "worker-summary-model" or "response_format" in payload:
            return httpx.Response(200, json=structured_reply())
        return httpx.Response(
            200, json=tool_reply() if len(payloads) == 1 else chat({"content": RESULT})
        )

    async def probe(
        required_text: str,
        optional_note: str | None,
        count: int = 3,
        tool_context: ToolContext = None,
    ) -> dict:
        """Read one bounded probe value without changing the workspace."""
        calls.append((required_text, optional_note, count, tool_context.invocation_id))
        return {"value": "tool-content-canary"}

    probe.contractor_sensitive_output = sensitive

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings().model_copy(update={})
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            normal = gateway_model(http)
            terminal = gateway_model(http, summary=True) if summary else None
            state = WorkerState()
            runtime = await create_runtime(
                tmp_path,
                state,
                {"probe": probe},
                normal,
                instruction=INSTRUCTION,
                instrumentation=adapter.handles.instrumentation if capture else None,
                summary_model=terminal,
                summary_instructions=(
                    "Preserve {unset_summary} literally. summary-instruction-canary"
                ),
                cumulative_budget=18 if summary else None,
            )
            try:
                completion = await runtime.invoke(stage_request())
                assert completion.failure is None
                assert completion.result.result == RESULT
                assert completion.result.summarized is summary
                assert len(calls) == 1
                assert calls[0] == ("argument-content-canary", None, 3, completion.invocation_id)
                assert_slot_released(runtime)
                report = state.metrics.build_report(report_id="conformance", duration_ms=1)
                assert report.metrics.model_calls == (1 if summary else 3)
                assert report.metrics.input_tokens == (11 if summary else 33)
                assert report.metrics.output_tokens == (7 if summary else 21)
                assert report.metrics.total_tokens == (18 if summary else 54)
                assert report.metrics.tools["probe"].calls == 1
                if summary:
                    assert report.metrics.summarizer.model_calls == 1
                    assert report.metrics.summarizer.total_tokens == 18
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
            assert normal.closed
            if terminal is not None:
                assert terminal.closed
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    assert trace.get_tracer_provider() is provider
    assert len(payloads) == (2 if summary else 3)
    first = payloads[0]
    assert "stream" not in first and first["model"] == "worker-model"
    assert first["messages"][0]["role"] == "system"
    # ADK appends the agent's own description after the literal instruction.
    assert first["messages"][0]["content"].startswith(INSTRUCTION + "\n\n")
    assert "response_format" not in first
    assert first["tool_choice"] == "auto"
    assert len(first["tools"]) == 1
    declaration = first["tools"][0]["function"]
    assert declaration["name"] == "probe"
    schema = declaration["parameters"]
    assert set(schema["properties"]) == {"required_text", "optional_note", "count"}
    assert set(schema["required"]) == {"required_text", "optional_note"}
    assert schema["properties"]["optional_note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert schema["properties"]["count"]["default"] == 3
    assert REASONING not in json.dumps(payloads[1:])
    assert_structured_payload(payloads[-1])
    if summary:
        assert "{unset_summary}" in payloads[-1]["messages"][0]["content"]
    assert all("stream" not in payload for payload in payloads)

    serialized = b"".join(exports)
    for forbidden in (
        GATEWAY_CREDENTIAL,
        GATEWAY_URL,
        HEADER_SECRET,
        "provider-extension-conformance-canary",
    ):
        assert forbidden.encode() not in serialized
        assert forbidden not in caplog.text
    visible = capture
    for canary in (
        REASONING,
        RESULT,
        "tool-content-canary",
        "argument-content-canary",
        "instruction-content-canary",
    ):
        assert (canary.encode() in serialized) is visible
    spans = [
        span
        for payload in exports
        for resource in ExportTraceServiceRequest.FromString(payload).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]
    if not capture:
        assert not spans
        return
    assert len({span.span_id for span in spans}) == len(spans)
    assert {span.name for span in spans} == {
        "contractor.worker.model",
        "contractor.worker.tool",
        "contractor.worker.a2a_task",
    }
    models = [span for span in spans if span.name == "contractor.worker.model"]
    assert len(models) == (2 if summary else 3)
    for span in models:
        attributes = {attribute.key: attribute.value for attribute in span.attributes}
        assert attributes["gen_ai.operation.name"].string_value == "chat"
        assert attributes["gen_ai.provider.name"].string_value == "contractor.gateway"
        assert attributes["gen_ai.usage.input_tokens"].int_value == 11
        assert attributes["gen_ai.usage.output_tokens"].int_value == 7
        assert "gen_ai.system" not in attributes


@pytest.mark.parametrize("role", ["main", "finalizer", "summarizer"])
@pytest.mark.parametrize("boundary", ["request", "recovery"])
def test_actual_gateway_role_cancellation_joins_request_or_recovery_without_resend(
    tmp_path, caplog, role, boundary
):
    async def scenario():
        waiting = asyncio.Event()
        settled, sent, calls = [], [], []
        authority = Authority(wait=True) if boundary == "recovery" else None
        if authority is not None:
            waiting = authority.observed

        async def gateway(request):
            payload = json.loads(request.content)
            sent.append(payload)
            blocked_role = (
                role == "main"
                or (role == "finalizer" and "response_format" in payload)
                or (role == "summarizer" and payload["model"] == "worker-summary-model")
            )
            if blocked_role:
                assert boundary == "request"
                waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    settled.append(role)
            return httpx.Response(
                200, json=tool_reply() if role == "summarizer" else chat({"content": RESULT})
            )

        async def probe(required_text: str, optional_note: str | None) -> dict:
            """Return one bounded observation before the isolated summary."""
            calls.append("probe")
            return {"value": "tool-content-canary"}

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            normal = gateway_model(http, recovery=authority if role == "main" else None)
            terminal = (
                gateway_model(http, summary=True, recovery=authority)
                if role == "summarizer"
                else None
            )
            state = WorkerState()
            runtime = await create_runtime(
                tmp_path,
                state,
                {"probe": probe} if role == "summarizer" else {},
                normal,
                summary_model=terminal,
                cumulative_budget=18 if terminal else None,
            )
            if role == "finalizer" and authority is not None:
                # Only the isolated serializer waits for recovery; the main
                # response is already spent and must remain charged once.
                original = runtime._pipeline.finalizer.run

                async def finalizer_with_recovery(**kwargs):
                    normal._client_handle.recovery = authority
                    return await original(**kwargs)

                runtime._pipeline.finalizer.run = finalizer_with_recovery
            task = asyncio.create_task(runtime.invoke(stage_request()))
            try:
                await asyncio.wait_for(waiting.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert_slot_released(runtime)
                assert state.metrics.final_outcome == "cancelled"
                assert len(sent) == ((1 if role != "main" else 0) + (boundary == "request"))
                assert calls == (["probe"] if role == "summarizer" else [])
                assert settled == ([role] if boundary == "request" else [])
                if authority is not None:
                    assert [event[2] for event in authority.events] == ["acquire"]
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
            for model in (normal, terminal):
                if model is not None:
                    assert model.closed
                    assert model._client_handle._api_key == ""
                    assert model._client_handle._completion_url is None
        assert GATEWAY_CREDENTIAL not in repr(state.metrics)
        assert GATEWAY_URL not in repr(state.metrics)

    asyncio.run(scenario())
    assert GATEWAY_CREDENTIAL not in caplog.text
    assert GATEWAY_URL not in caplog.text


@pytest.mark.parametrize("response_case", ["length", "malformed_arguments", "missing_usage"])
def test_actual_summary_gateway_keeps_strict_errors_and_spent_usage(tmp_path, response_case):
    async def scenario():
        sent, calls = [], []

        async def probe(required_text: str, optional_note: str | None) -> dict:
            """Return one bounded observation before the isolated summary."""
            calls.append("probe")
            return {"value": "tool-content-canary"}

        async def gateway(request):
            payload = json.loads(request.content)
            sent.append(payload)
            if payload["model"] == "worker-model":
                return httpx.Response(200, json=tool_reply())
            if response_case == "missing_usage":
                response = structured_reply()
                response.pop("usage")
            else:
                response = tool_reply()
                response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
                    '{"path":'
                )
                if response_case == "length":
                    response["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            state = WorkerState()
            runtime = await create_runtime(
                tmp_path,
                state,
                {"probe": probe},
                gateway_model(http),
                summary_model=gateway_model(http, summary=True),
                cumulative_budget=18,
            )
            try:
                completion = await runtime.invoke(stage_request())
                if response_case == "missing_usage":
                    assert completion.failure is None
                    assert completion.result.result == RESULT
                    assert completion.result.summarized
                else:
                    assert completion.result is None
                    assert completion.failure.code == "worker_summarization_failed"
                summary = state.metrics.build_report(
                    report_id="conformance", duration_ms=1
                ).metrics.summarizer
                assert summary.model_calls == 1
                assert summary.token_usage_unavailable == (response_case == "missing_usage")
                assert summary.total_tokens == (0 if response_case == "missing_usage" else 18)
                assert (
                    summary.failure_codes
                    == {
                        "length": {"output_limit_exceeded": 1},
                        "malformed_arguments": {"gateway_unavailable": 1},
                        "missing_usage": {},
                    }[response_case]
                )
                assert state.metrics.counters["total_tokens"] == 18
                assert calls == ["probe"]
                assert len(sent) == 2
                assert_structured_payload(sent[-1])
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("duplicate", [False, True])
def test_actual_gateway_rejects_duplicate_arguments_and_allows_envelope_extensions(
    tmp_path, duplicate
):
    async def scenario():
        calls, sent = [], []

        async def probe(required_text: str, optional_note: str | None) -> dict:
            """Read one bounded value to characterize the provider JSON parser."""
            calls.append(required_text)
            return {"ok": True}

        async def gateway(request):
            payload = json.loads(request.content)
            sent.append(payload)
            if len(sent) == 1:
                response = tool_reply()
                response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
                    '{"required_text":"first","required_text":"last","optional_note":null}'
                    if duplicate
                    else '{"required_text":"first","optional_note":null}'
                )
                response["provider_extension"] = {"valid": True}
            elif "response_format" in payload:
                response = structured_reply()
            else:
                response = chat({"content": RESULT})
            return httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            state = WorkerState()
            runtime = await create_runtime(tmp_path, state, {"probe": probe}, gateway_model(http))
            try:
                if duplicate:
                    completion = await runtime.invoke(stage_request())
                    assert completion.failure is not None
                    assert calls == [] and len(sent) == 1
                    assert state.metrics.counters["total_tokens"] == 18
                    return
                completion = await runtime.invoke(stage_request())
                assert completion.failure is None and completion.result.result == RESULT
                assert calls == ["first"]
                assert len(sent) == 3
                assert state.metrics.counters["total_tokens"] == 54
                assert (
                    state.metrics.build_report(report_id="conformance", duration_ms=1)
                    .metrics.tools["probe"]
                    .calls
                    == 1
                )
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_actual_parallel_dispatch_preserves_self_recorded_correlation_for_every_error_path(
    tmp_path, caplog
):
    async def scenario():
        state, instrumentation = WorkerState(), RecordingInstrumentation()
        barrier = asyncio.Barrier(2)
        executed, sent = [], []

        class ProbeFailure(RuntimeError):
            code = "probe_domain_failed"
            retryable = False

        async def probe(tag: str) -> dict:
            """Return a bounded observation or its classified domain error."""
            with RecordedToolCall(state.metrics, "probe", {"tag": tag}) as recorded:
                executed.append(tag)
                await barrier.wait()
                if tag == "domain-error":
                    raise ProbeFailure(GATEWAY_CREDENTIAL)
                result = {"ok": True}
                recorded.succeed(result)
                return result

        async def infrastructure_error() -> dict:
            """Expose a controlled infrastructure error for callback accounting."""
            executed.append("infrastructure-error")
            raise RuntimeError(GATEWAY_CREDENTIAL)

        async def rejected() -> dict:
            """Return no observation when raw admission rejects its call."""
            pytest.fail("raw rejected tool body executed")

        rejected.contractor_raw_argument_error = lambda _: ProbeFailure()

        async def gateway(request):
            payload = json.loads(request.content)
            sent.append(payload)
            if len(sent) == 1:
                selected = [
                    ("probe", {"tag": "success"}),
                    ("probe", {"tag": "domain-error"}),
                    ("infrastructure_error", {}),
                    ("rejected", {}),
                    ("arbitrary-unknown-tool", {}),
                ]
                return httpx.Response(
                    200,
                    json=chat(
                        {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"parallel-{index}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                                for index, (name, args) in enumerate(selected)
                            ],
                        },
                        finish_reason="tool_calls",
                    ),
                )
            return httpx.Response(
                200,
                json=structured_reply()
                if "response_format" in payload
                else chat({"content": RESULT}),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            runtime = await create_runtime(
                tmp_path,
                state,
                {
                    "probe": probe,
                    "infrastructure_error": infrastructure_error,
                    "rejected": rejected,
                },
                gateway_model(http),
                instrumentation=instrumentation,
            )
            try:
                completion = await runtime.invoke(stage_request())
                assert completion.failure is None and completion.result.result == RESULT
                assert sorted(executed) == ["domain-error", "infrastructure-error", "success"]
                report = state.metrics.build_report(report_id="conformance", duration_ms=1)
                assert len(report.tool_calls) == 5
                assert len({call.call_id for call in report.tool_calls}) == 5
                assert set(report.metrics.tools) == {
                    "probe",
                    "infrastructure_error",
                    "rejected",
                    "unknown_tool",
                }
                assert report.metrics.tools["probe"].calls == 2
                assert report.metrics.tools["probe"].succeeded == 1
                assert report.metrics.tools["probe"].failed == 1
                for name in ("infrastructure_error", "rejected", "unknown_tool"):
                    assert (
                        report.metrics.tools[name].calls == report.metrics.tools[name].failed == 1
                    )
                snapshot = await state.snapshot()
                assert snapshot["lastCompletedInvocation"]["metrics"]["toolCalls"] == 5
                tool_spans = [
                    span for span in instrumentation.spans if span.name == "contractor.worker.tool"
                ]
                assert len(tool_spans) == 5
                assert all(span.end_calls == 1 for span in instrumentation.spans)
                assert len(sent) == 3 and report.metrics.total_tokens == 54
                assert "arbitrary-unknown-tool" not in report.model_dump_json(by_alias=True)
                assert GATEWAY_CREDENTIAL not in report.model_dump_json(by_alias=True)
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())
    assert GATEWAY_CREDENTIAL not in caplog.text


@pytest.mark.parametrize(("capture", "sensitive"), [(False, False), (True, False), (True, True)])
def test_native_skill_content_reaches_only_the_authorized_serialized_sink(
    tmp_path, capture, sensitive
):
    exports, sent = [], []

    async def collector(request):
        exports.append(await request.aread())
        return httpx.Response(200, json={})

    async def gateway(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) <= 2:
            name, args = (
                ("load_skill", {"skill_name": "demo"})
                if len(sent) == 1
                else (
                    "load_skill_resource",
                    {"skill_name": "demo", "file_path": "references/guide.md"},
                )
            )
            return httpx.Response(
                200,
                json=chat(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"skill-{len(sent)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    },
                    finish_reason="tool_calls",
                ),
            )
        return httpx.Response(
            200,
            json=structured_reply() if "response_format" in payload else chat({"content": RESULT}),
        )

    async def selected_sensitive_tool() -> dict:
        """Keep sensitive output policy allocation-wide even before execution."""
        pytest.fail("unselected model call unexpectedly executed the sensitive tool")

    selected_sensitive_tool.contractor_sensitive_output = sensitive

    async def scenario():
        selected, value = resolved_value(skill_package())
        client, state = FakeSkillClient({"demo": value}), WorkerState()
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings().model_copy(update={})
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            context = replace(
                skill_context(tmp_path, state, selected),
                tools={"selected_sensitive_tool": selected_sensitive_tool},
                adapter_handles=AdapterHandles(
                    instrumentation=adapter.handles.instrumentation if capture else None
                ),
            )
            runtime = await AdkWorkerRuntimeFactory(
                lambda _: gateway_model(http), lambda _allocation, _settings: client
            ).create(context)
            try:
                completion = await runtime.invoke(stage_request())
                assert completion.failure is None and completion.result.result == RESULT
                assert len(sent) == 4
                assert [call.tool for call in state.metrics.tool_calls] == [
                    "load_skill",
                    "load_skill_resource",
                ]
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
            assert not (tmp_path / ".agent-skills").exists()
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    serialized = b"".join(exports)
    for canary in (b"Follow the exact demo procedure", b"Reference contents"):
        assert (canary in serialized) is capture
    assert HEADER_SECRET.encode() not in serialized
    assert GATEWAY_CREDENTIAL.encode() not in serialized
    assert GATEWAY_URL.encode() not in serialized
