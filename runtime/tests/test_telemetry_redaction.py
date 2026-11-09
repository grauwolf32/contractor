from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from test_otlp_adapter import adapter_context, telemetry_settings

from contractor_runtime.adapters.content import MAX_CONTENT_BYTES, capture_span_content
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.telemetry.redaction import TelemetryRedactor

CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/telemetry-credential-redaction.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_shared_exact_replacement_shared_fixture(case):
    assert TelemetryRedactor(case["secrets"]).text(case["input"]) == case["expected"]


def test_shared_known_credentials_are_replaced_before_encoding_truncation_and_queueing():
    secret = 'pw"\\\nключ'
    values = (secret, "short")
    original = {"text": "payload " + secret + " short", "nested": [secret], "name": "ordinary"}
    snapshot = json.dumps(original)
    payloads = []

    async def collect(request):
        payloads.append(await request.aread())
        return httpx.Response(200)

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collect)).create(
            replace(adapter_context(), known_credentials=values), telemetry_settings()
        )
        instrumentation = adapter.handles.instrumentation
        span = instrumentation.start_span("contractor.worker.tool")
        capture_span_content(span, input=lambda: original)
        capture_span_content(span, output=lambda: "x" * (MAX_CONTENT_BYTES - 8) + secret)
        span.end(outcome="succeeded", attributes={"error.type": "short"})
        assert json.dumps(original) == snapshot
        for queued in instrumentation._queue:
            assert b"short" not in queued.encoded
            assert secret.encode() not in queued.encoded
        await adapter.flush()
        await adapter.close()
        assert instrumentation._redactor.text("short") == "short"

    asyncio.run(scenario())
    assert payloads
    message = ExportTraceServiceRequest.FromString(payloads[0])
    span = message.resource_spans[0].scope_spans[0].spans[0]
    attrs = {a.key: a.value.string_value for a in span.attributes}
    captured = json.loads(attrs["contractor.input"])
    assert captured == {
        "text": "payload [REDACTED] [REDACTED]",
        "nested": ["[REDACTED]"],
        "name": "ordinary",
    }
    assert json.loads(attrs["contractor.output"])["truncated"] is True


def test_shared_data_keys_are_preserved_without_collisions():
    redactor = TelemetryRedactor(["one", "two"])
    assert redactor.value({"one": "one", "two": "two"}) == {
        "one": "[REDACTED]",
        "two": "[REDACTED]",
    }
    assert repr(redactor) == "TelemetryRedactor([REDACTED])"


def test_shared_adapter_host_supplies_only_execution_scoped_known_credentials():
    from datetime import UTC, datetime, timedelta

    from test_adapter_host import configured_spec

    from contractor_runtime.adapters.host import AllocationAdapterHost
    from contractor_runtime.contracts.settings import HTTPIdentityMaterial

    exports = []

    async def collect(request):
        exports.append(await request.aread())
        return httpx.Response(200)

    async def scenario():
        spec = configured_spec(telemetry=True)
        spec.runtime_settings.http_identities = {
            "alice": HTTPIdentityMaterial(
                credential={"credentialId": "alice-secret", "kind": "target-bearer@1"},
                fields={"token": "target-identity-canary"},
            )
        }

        def now():
            return datetime.now(UTC)

        deadline = now() + timedelta(seconds=10)
        host = await AllocationAdapterHost.create(
            spec,
            {"otlp-http@1": OTLPHTTPAdapterFactory(httpx.MockTransport(collect))},
            deadline=deadline,
            now=now,
        )
        try:
            span = host.handles.instrumentation.start_span("contractor.worker.model")
            capture_span_content(
                span, output=lambda: {"text": "Observed target-identity-canary /invoices"}
            )
            span.end(outcome="succeeded")
        finally:
            await host.terminate(deadline=deadline, now=now)

    asyncio.run(scenario())
    assert exports
    assert b"target-identity-canary" not in b"".join(exports)
    assert b"/invoices" in b"".join(exports)
    assert b"[REDACTED]" in b"".join(exports)


@pytest.mark.parametrize("model", [False, True])
def test_shared_large_content_is_projected_and_masked_before_byte_bounds(model):
    secret = "credential-boundary-canary-" + "q" * 256
    text = "x" * (MAX_CONTENT_BYTES - 128) + secret + "x" * 1024
    original = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "tools": [{"functionDeclarations": [{"name": "lookup"}]}],
    }
    payloads = []

    async def collect(request):
        payloads.append(await request.aread())
        return httpx.Response(200)

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collect)).create(
            replace(adapter_context(), known_credentials=(secret,)), telemetry_settings()
        )
        instrumentation = adapter.handles.instrumentation
        span = instrumentation.start_span(
            "contractor.worker.model" if model else "contractor.worker.tool"
        )
        capture_span_content(span, input=lambda: original)
        span.end(outcome="succeeded")
        assert original["contents"][0]["parts"][0]["text"] == text
        for queued in instrumentation._queue:
            assert b"credential-boundary-canary-" not in queued.encoded
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    batch = ExportTraceServiceRequest.FromString(payloads[0])
    span = batch.resource_spans[0].scope_spans[0].spans[0]
    attrs = {a.key: a.value.string_value for a in span.attributes}
    if model:
        assert json.loads(attrs["gen_ai.tool.definitions"])[0]["name"] == "lookup"
        messages = json.loads(attrs["gen_ai.input.messages"])
        assert messages[0]["role"] == "user"
    for key, value in attrs.items():
        if (
            key.startswith("gen_ai.") and key.endswith(("messages", "definitions"))
        ) or key == "contractor.input":
            assert len(value.encode()) <= MAX_CONTENT_BYTES
            json.loads(value)


@pytest.mark.parametrize("failure", ["object", "depth", "values"])
def test_shared_capture_failure_omits_only_affected_content(failure):
    value = object()
    if failure == "depth":
        value = "deep"
        for _ in range(66):
            value = [value]
    elif failure == "values":
        value = [None] * 100001
    payloads = []

    async def collect(request):
        payloads.append(await request.aread())
        return httpx.Response(200)

    async def scenario():
        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collect)).create(
            adapter_context(), telemetry_settings()
        )
        span = adapter.handles.instrumentation.start_span("contractor.worker.model")
        capture_span_content(span, input=lambda: value, output=lambda: "useful output")
        span.end(outcome="succeeded")
        await adapter.flush()
        await adapter.close()

    asyncio.run(scenario())
    batch = ExportTraceServiceRequest.FromString(payloads[0])
    attrs = {
        a.key: a.value.string_value
        for a in batch.resource_spans[0].scope_spans[0].spans[0].attributes
    }
    assert "gen_ai.input.messages" not in attrs
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == "useful output"
