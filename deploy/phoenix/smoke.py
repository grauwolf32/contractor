"""Real-exporter OTLP check against the disposable Phoenix test container."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen


async def emit_runtime(endpoint):
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
        headers={},
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
                "contents": [
                    {"role": "user", "parts": [{"text": "synthetic Runtime input"}]}
                ],
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
    finally:
        await adapter.close()


def check():
    expected = {
        "contractor.planner.model": ("Go", 2, 3, 1),
        "contractor.worker.model": ("Runtime", 13, 17, 3),
    }
    for _ in range(30):
        try:
            with urlopen(
                "http://127.0.0.1:6006/v1/projects/default/spans?limit=10", timeout=5
            ) as response:
                spans = {span["name"]: span for span in json.load(response)["data"]}
        except HTTPError as error:
            if error.code != 404:
                raise
            spans = {}
        if set(expected) <= set(spans):
            for name, (source, inputs, outputs, cached) in expected.items():
                span = spans[name]
                values = span["attributes"]
                checks = {
                    "llm.token_count.prompt": inputs,
                    "llm.token_count.completion": outputs,
                    "llm.token_count.total": inputs + outputs,
                    "llm.token_count.prompt_details.cache_read": cached,
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": f"{source} system instruction",
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": f"synthetic {source} input",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": f"synthetic {source} output",
                }
                if span["span_kind"] != "LLM":
                    raise RuntimeError(f"{name} was not classified as an LLM span")
                for key, value in checks.items():
                    if values.get(key) != value:
                        raise RuntimeError(
                            f"{name}: native Phoenix projection mismatch for {key}: {values.get(key)!r}"
                        )
                if any(key.startswith("langfuse.") for key in values):
                    raise RuntimeError("Collector received vendor-specific attributes")
            print(
                "Real Go/Runtime exporters: native LLM prompts, responses and token projections verified"
            )
            return
        time.sleep(1)
    raise RuntimeError("The ingested spans did not appear within the probe budget")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "runtime":
        asyncio.run(emit_runtime(sys.argv[2]))
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        raise SystemExit("Usage: smoke.py runtime ENDPOINT | check")
