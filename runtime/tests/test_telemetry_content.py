from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fakes.model import json_result, scripted_model
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from test_instrumentation import FakeContext, FakeOwner, FakeTool
from test_otlp_adapter import HEADER_SECRET, adapter_context, telemetry_settings
from test_result_finalizer import policy

from contractor_runtime.adapters.content import (
    MAX_CONTENT_BYTES,
    capture_span_content,
    encode_content,
)
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory, _accepted_response
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker.state import WorkerStateStore
from contractor_runtime.worker.summarizer import TerminalSummarizer


@pytest.mark.parametrize("enabled", [False, True])
def test_model_tool_and_finalizer_capture_opt_in(enabled: bool) -> None:
    payloads: list[bytes] = []

    async def collector(request: httpx.Request) -> httpx.Response:
        payloads.append(await request.aread())
        return httpx.Response(200, json={"name": "otel-ingestion-job", "id": "1"})

    async def scenario() -> None:
        settings = telemetry_settings().model_copy(update={"capture_content": enabled})
        # Validate the actual private wire representation too.
        type(settings).model_validate_json(settings.model_dump_json())
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings
        )
        plugin = WorkerInstrumentationPlugin(
            state=WorkerStateStore(),
            budget=lambda: None,
            observe_artifacts=lambda _owner, _cursor: None,
            instrumentation=adapter.handles.instrumentation,
            model_alias="test-model",
        )
        context = FakeContext("capture-test")
        plugin.prepare_invocation(invocation_id=context.invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=SimpleNamespace(
                contents=[{"text": "prompt-secret-canary"}],
                config=SimpleNamespace(
                    system_instruction="system-canary", tools=[{"name": "probe"}]
                ),
            ),
        )
        await plugin.after_model_callback(
            callback_context=context,
            llm_response=SimpleNamespace(content={"text": "response-canary"}, usage_metadata=None),
        )
        tool = FakeTool("probe", FakeOwner())
        await plugin.before_tool_callback(
            tool=tool, tool_args={"secret": "tool-input-canary"}, tool_context=context
        )
        await plugin.after_tool_callback(
            tool=tool, tool_args={}, tool_context=context, result={"secret": "tool-output-canary"}
        )
        await plugin.before_result_finalizer_call(invocation_id=context.invocation_id)
        await plugin.capture_result_finalizer_content(
            invocation_id=context.invocation_id,
            input="finalizer-input-canary",
            output="finalizer-output-canary",
        )
        await plugin.after_result_finalizer_call(invocation_id=context.invocation_id, usage=None)
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    payload = payloads[0]
    assert HEADER_SECRET.encode() not in payload
    for canary in (
        "prompt-secret-canary",
        "system-canary",
        "response-canary",
        "tool-input-canary",
        "tool-output-canary",
        "finalizer-input-canary",
        "finalizer-output-canary",
    ):
        assert (canary.encode() in payload) is enabled
    spans = ExportTraceServiceRequest.FromString(payload).resource_spans[0].scope_spans[0].spans
    for span in spans:
        attrs = {attr.key: attr.value.string_value for attr in span.attributes}
        if span.name == "contractor.worker.model":
            assert attrs["langfuse.observation.type"] == "generation"
        for key in ("langfuse.observation.input", "langfuse.observation.output"):
            if key in attrs:
                json.loads(attrs[key])


def test_content_truncation_is_bounded_valid_json() -> None:
    value = encode_content({"text": "界" * MAX_CONTENT_BYTES})
    assert len(value.encode()) <= MAX_CONTENT_BYTES
    assert json.loads(value)["truncated"] is True


def test_disabled_capture_never_evaluates_content() -> None:
    def forbidden() -> object:
        pytest.fail("disabled capture accessed content")

    capture_span_content(SimpleNamespace(capture_content=False), input=forbidden, output=forbidden)


def test_content_serialization_failure_is_optional() -> None:
    span = SimpleNamespace(capture_content=True)
    capture_span_content(span, input=lambda: object())


def test_terminal_summarizer_captures_actual_adk_request_and_response() -> None:
    payloads: list[bytes] = []

    async def collector(request: httpx.Request) -> httpx.Response:
        payloads.append(await request.aread())
        return httpx.Response(200)

    async def scenario() -> None:
        settings = telemetry_settings().model_copy(update={"capture_content": True})
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), settings
        )
        summarizer = TerminalSummarizer(
            model=scripted_model(
                [json_result({"subtaskId": "1", "result": "summary-output-canary"})],
                auto_result_finalizer=False,
            ),
            policy=policy(),
            instrumentation=adapter.handles.instrumentation,
        )
        await summarizer.run(prompt="summary-input-canary", invocation_id="summary-test")
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    assert b"summary-input-canary" in payloads[0]
    assert b"summary-output-canary" in payloads[0]
    assert b"systemInstruction" in payloads[0]


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ({}, True),
        ({"partialSuccess": {"rejectedSpans": "0"}}, True),
        ({"partialSuccess": {"rejectedSpans": "1"}}, False),
        ({"name": "otel-ingestion-job", "id": "1"}, True),
        ({"name": "otel-ingestion-job"}, False),
        ({"error": "failed"}, False),
        ([], False),
    ],
)
def test_json_collector_ack(value: object, accepted: bool) -> None:
    assert asyncio.run(_accepted_response(httpx.Response(200, json=value))) is accepted
