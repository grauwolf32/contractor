"""Bounded, content-free OTLP/HTTP acknowledgement and retry policy."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime

import httpx
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

from contractor_runtime.contracts import TelemetryRetrySettings

# Protocol classification, independent of an operator's retry count/backoff.
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
MAX_RESPONSE_BYTES = 64 * 1024
MAX_RETRY_AFTER_BYTES = 128


@dataclass(frozen=True, slots=True)
class DeliveryResponse:
    accepted: bool = False
    retryable: bool = False
    rejected_spans: int = 0
    retry_after: float | None = None


def retry_after_seconds(value: str | None, *, now: float | None = None) -> float | None:
    if value is None or len(value) > MAX_RETRY_AFTER_BYTES:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(int(value))
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, when.timestamp() - (time.time() if now is None else now))
    except (TypeError, ValueError, OverflowError):
        return None


def backoff_seconds(settings: TelemetryRetrySettings, retry_index: int) -> float:
    ceiling = (
        min(
            settings.max_backoff_milliseconds,
            settings.initial_backoff_milliseconds * 2**retry_index,
        )
        / 1000.0
    )
    # Equal jitter keeps a positive delay even at the low end of randomness.
    return random.uniform(ceiling / 2, ceiling)


async def classify_response(response: httpx.Response, *, span_count: int) -> DeliveryResponse:
    if response.status_code in RETRYABLE_STATUS_CODES:
        return DeliveryResponse(
            retryable=True, retry_after=retry_after_seconds(response.headers.get("retry-after"))
        )
    if not 200 <= response.status_code < 300:
        return DeliveryResponse()
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                return DeliveryResponse()
            body.extend(chunk)
        if body and "application/json" in response.headers.get("content-type", ""):
            value = json.loads(body)
            if not isinstance(value, dict):
                return DeliveryResponse()
            # Retain the explicit Langfuse durable-ingestion acknowledgement.
            if value.get("name") == "otel-ingestion-job":
                return DeliveryResponse(
                    accepted=isinstance(value.get("id"), str) and bool(value["id"])
                )
            if response.status_code != 200 or set(value) - {"partialSuccess"}:
                return DeliveryResponse()
            partial = value.get("partialSuccess", {})
            if not isinstance(partial, dict) or set(partial) - {"rejectedSpans", "errorMessage"}:
                return DeliveryResponse()
            rejected = partial.get("rejectedSpans", 0)
            if isinstance(rejected, str) and rejected.isascii() and rejected.isdigit():
                # Reject excessive integer strings without converting unbounded input.
                if len(rejected) > 20:
                    return DeliveryResponse()
                rejected = int(rejected)
            if type(rejected) is not int or not isinstance(partial.get("errorMessage", ""), str):
                return DeliveryResponse()
        else:
            if response.status_code != 200:
                return DeliveryResponse()
            decoded = ExportTraceServiceResponse.FromString(bytes(body))
            rejected = decoded.partial_success.rejected_spans
    except (DecodeError, ValueError, RecursionError):
        return DeliveryResponse()
    if not 0 <= rejected <= span_count:
        return DeliveryResponse()
    # Partial acceptance is terminal, including warnings with zero rejected spans.
    # Its diagnostic message is deliberately neither retained nor exposed.
    return DeliveryResponse(accepted=True, rejected_spans=rejected)
