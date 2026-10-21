"""Send a synthetic model span through the production Runtime OTLP exporter."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace


async def emit_runtime(endpoint, headers):
    from contractor_runtime.adapters.content import capture_span_content
    from contractor_runtime.adapters.genai import usage_attributes
    from contractor_runtime.adapters.host import RuntimeAdapterBuildContext
    from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
    from contractor_runtime.contracts import TelemetrySettings

    context = RuntimeAdapterBuildContext(
        allocation_id="runtime-allocation",
        run_id="runtime-smoke",
        stage_execution_id="runtime-stage",
        logical_agent_name="builder",
        request_timeout_seconds=10,
        runtime_config_refs=(),
        runtime_config_digests=(),
        run_labels=(),
        agent_labels=(),
        run_metadata_labels={},
        runtime_adapter_refs=("otlp-http@1",),
        private_bypass_hosts=(),
    )
    settings = TelemetrySettings(
        adapter="otlp-http@1",
        endpoint=endpoint,
        captureContent=True,
        headers=headers,
        flushTimeoutSeconds=10,
        export={
            "batchSizeBytes": 1048576,
            "maxAttempts": 1,
            "maxPendingSpans": 4,
            "maxPendingBytes": 1048576,
            "retry": {"initialBackoffMilliseconds": 1, "maxBackoffMilliseconds": 1},
        },
    )
    adapter = await OTLPHTTPAdapterFactory().create(context, settings)
    try:
        span = adapter.handles.instrumentation.start_span(
            "contractor.worker.model", attributes={"model.alias": "smoke-model"}
        )
        capture_span_content(
            span,
            input=lambda: {
                "systemInstruction": "Runtime system instruction",
                "contents": [{"role": "user", "parts": [{"text": "synthetic Runtime input"}]}],
            },
            output=lambda: {
                "role": "model",
                "parts": [{"text": "synthetic Runtime output"}],
            },
        )
        span.end(
            outcome="succeeded",
            attributes=usage_attributes(
                SimpleNamespace(
                    prompt_token_count=13,
                    candidates_token_count=17,
                    cached_content_token_count=3,
                )
            ),
        )
        await adapter.flush()
        if adapter.metrics.failed_operations:
            raise RuntimeError("Runtime exporter reported failed delivery")
    finally:
        await adapter.close()


if __name__ == "__main__":
    headers = json.loads(os.environ.get("CONTRACTOR_TEST_OTLP_HEADERS", "{}"))
    asyncio.run(emit_runtime(sys.argv[1], headers))
