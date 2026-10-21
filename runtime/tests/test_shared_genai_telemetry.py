from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from test_otlp_adapter import adapter_context, telemetry_settings

from contractor_runtime.adapters.content import capture_span_content
from contractor_runtime.adapters.genai import usage_attributes
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.adapters.otlp_retry import classify_response


def test_shared_genai_telemetry() -> None:
    corpus = json.loads(
        (
            Path(__file__).resolve().parents[2] / "api/testdata/telemetry-genai-cases.json"
        ).read_text()
    )

    async def scenario() -> None:
        for case in corpus["content"]:
            payloads = []

            async def collector(request: httpx.Request, payloads=payloads) -> httpx.Response:
                payloads.append(await request.aread())
                return httpx.Response(200)

            settings = telemetry_settings().model_copy(update={"capture_content": True})
            adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
                adapter_context(), settings
            )
            span = adapter.handles.instrumentation.start_span(
                "contractor.worker.model" if case["model"] else "contractor.worker.tool",
                attributes={"model.alias": "fixture-model"},
            )
            capture_span_content(span, **{case["field"]: lambda case=case: case["value"]})
            span.end(outcome="succeeded")
            await adapter.flush()
            await adapter.close()
            emitted = (
                ExportTraceServiceRequest.FromString(payloads[0])
                .resource_spans[0]
                .scope_spans[0]
                .spans[0]
            )
            actual = {}
            for attr in emitted.attributes:
                assert not attr.key.startswith(("langfuse.", "openinference."))
                if attr.key in {
                    "gen_ai.input.messages",
                    "gen_ai.output.messages",
                    "gen_ai.system_instructions",
                    "gen_ai.tool.definitions",
                    "contractor.input",
                    "contractor.output",
                }:
                    actual[attr.key] = json.loads(attr.value.string_value)
            assert actual == case["expected"], case["name"]
        for case in corpus["usage"]:
            usage = SimpleNamespace(
                prompt_token_count=case.get("input"),
                candidates_token_count=case.get("output"),
                cached_content_token_count=case.get("cached"),
            )
            assert usage_attributes(usage) == case["expected"]
        for case in corpus["acknowledgements"]:
            receipt = await classify_response(httpx.Response(200, json=case["body"]), span_count=1)
            assert receipt.accepted is case["accepted"]

    asyncio.run(scenario())
