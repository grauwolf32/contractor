"""Synthetic OTLP check, executed inside the disposable Phoenix test container."""

import json
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

TRACE_ID = bytes.fromhex("1234567890abcdef1234567890abcdef")
INPUT = '{"messages":[{"role":"user","content":"synthetic smoke input"}]}'
OUTPUT = '{"text":"synthetic smoke output"}'


def send():
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    resource.resource.attributes.add(key="service.name").value.string_value = "contractor.runtime"
    scope = resource.scope_spans.add()
    scope.scope.name = "contractor.runtime.worker"
    span = scope.spans.add(
        trace_id=TRACE_ID,
        span_id=bytes.fromhex("1234567890abcdef"),
        name="contractor.worker.model",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=time.time_ns() - 1_000_000,
        end_time_unix_nano=time.time_ns(),
        status=Status(code=Status.STATUS_CODE_OK),
    )
    for key, value in {
        "outcome": "succeeded",
        "model.alias": "smoke-model",
        "gen_ai.request.model": "smoke-model",
        "langfuse.observation.type": "generation",
        "langfuse.observation.input": INPUT,
        "langfuse.observation.output": OUTPUT,
    }.items():
        span.attributes.add(key=key).value.string_value = value
    span.attributes.add(key="tokens.input").value.int_value = 2
    span.attributes.add(key="tokens.output").value.int_value = 3
    with urlopen(
        Request(
            "http://127.0.0.1:6006/v1/traces",
            data=request.SerializeToString(),
            headers={"Content-Type": "application/x-protobuf"},
        ),
        timeout=10,
    ) as response:
        receipt = ExportTraceServiceResponse.FromString(response.read())
        if response.status != 200 or receipt.partial_success.rejected_spans:
            raise RuntimeError("Phoenix rejected the OTLP spans")


def check():
    for _ in range(30):
        try:
            with urlopen(
                "http://127.0.0.1:6006/v1/projects/default/spans?limit=10", timeout=5
            ) as response:
                spans = json.load(response)["data"]
        except HTTPError as error:
            if error.code != 404:
                raise
            spans = []
        for span in spans:
            if span["context"]["trace_id"] != TRACE_ID.hex():
                continue
            values = span["attributes"]
            if (
                values.get("langfuse.observation.input") != INPUT
                or values.get("langfuse.observation.output") != OUTPUT
            ):
                raise RuntimeError("Phoenix did not preserve the raw content attributes")
            if values.get("tokens.input") != 2 or span["name"] != "contractor.worker.model":
                raise RuntimeError("Phoenix did not preserve the span metadata")
            print("OTLP span and raw Contractor attributes verified")
            return
        time.sleep(1)
    raise RuntimeError("The ingested OTLP span did not appear within the probe budget")


if __name__ == "__main__":
    if sys.argv[1:] == ["send"]:
        send()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        raise SystemExit("Usage: smoke.py send|check")
