from __future__ import annotations

import asyncio
import base64
import copy
import json
from pathlib import Path

import pytest
from private_schemas import private_validator
from pydantic import ValidationError

from contractor_runtime.finding_proposals import MAX_FINDING_SUBMISSION_BYTES
from contractor_runtime.toolsets.common import http_capture
from contractor_runtime.toolsets.common.http_capture import CapturedHTTPExchange, RecentHTTPCaptures
from contractor_runtime.toolsets.common.http_limits import (
    MAX_EXCHANGE_ATTEMPTS,
    MAX_HEADER_BYTES,
    MAX_HISTORY,
    MAX_REQUEST_BODY_BYTES,
    MAX_RESPONSE_BODY_BYTES,
)


def test_shared_http_finding_capture():
    fixtures = json.loads(
        (Path(__file__).parents[2] / "api/testdata/http-capture-cases.json").read_text()
    )
    assert fixtures["limits"] == {
        "bufferEntries": MAX_HISTORY,
        "bufferBytes": http_capture.MAX_CAPTURE_BUFFER_BYTES,
        "requestBodyBytes": MAX_REQUEST_BODY_BYTES,
        "responseBodyBytes": MAX_RESPONSE_BODY_BYTES,
        "headerBytes": MAX_HEADER_BYTES,
        "attempts": MAX_EXCHANGE_ATTEMPTS,
        "submissionBytes": MAX_FINDING_SUBMISSION_BYTES,
    }
    for case in fixtures["cases"]:
        if case["valid"]:
            CapturedHTTPExchange.model_validate(case["value"])
            private_validator("http-exchange.schema.json").validate(case["value"])
            private_validator("http-exchange.schema.json#/$defs/artifact").validate(
                {
                    "schema": http_capture.HTTP_EXCHANGE_SCHEMA,
                    "capture": case["value"],
                    "finding": {"proposal_id": "proposal-test", "receipt_id": "receipt-test"},
                }
            )
        else:
            with pytest.raises(ValidationError):
                CapturedHTTPExchange.model_validate(case["value"])


def capture(request_id="1"):
    return {
        "source": "http",
        "request_id": request_id,
        "request_tag": f"r-test-h{request_id}",
        "attempts": [
            {
                "request": {
                    "method": "GET",
                    "url": "https://target.example/",
                    "message": {"kind": "structured", "headers": [], "body_base64": ""},
                },
                "error": "timeout",
            }
        ],
    }


def test_capture_buffer_count_ownership_replacement_and_clear():
    async def scenario():
        buffer = RecentHTTPCaptures()
        for number in range(1, 131):
            await buffer.put(capture(str(number)), "invocation-1")
        assert len(buffer._entries) == 128
        with pytest.raises(ValueError, match="unavailable"):
            await buffer.get("http", "1", invocation_id="invocation-1")
        with pytest.raises(ValueError, match="unavailable"):
            await buffer.get("http", "3", invocation_id="invocation-2")
        selected = await buffer.get("http", "3", invocation_id="invocation-1")
        assert selected["attempts"][0]["error"] == "timeout"
        selected["attempts"].clear()
        assert (await buffer.get("http", "3"))["attempts"]
        size = buffer._bytes
        await buffer.put(capture("130"), "invocation-1")
        assert buffer._bytes == size and len(buffer._entries) == 128
        buffer.clear("caido")
        assert buffer.contains("http", "130")
        buffer.clear("http")
        assert buffer._bytes == 0 and not buffer._entries

    asyncio.run(scenario())


def test_capture_buffer_real_response_bound_evicts_by_bytes():
    async def scenario():
        buffer = RecentHTTPCaptures()
        body = base64.b64encode(b"x" * MAX_RESPONSE_BODY_BYTES).decode("ascii")
        for number in range(1, 5):
            value = capture(str(number))
            attempt = value["attempts"][0]
            attempt.pop("error")
            attempt["response"] = {
                "status": 200,
                "complete": True,
                "message": {"kind": "structured", "headers": [], "body_base64": body},
            }
            await buffer.put(value, "invocation-1")
            assert buffer._bytes <= 64 * 1024 * 1024
        assert len(buffer._entries) == 2
        assert not buffer.contains("http", "2")
        assert buffer.contains("http", "3") and buffer.contains("http", "4")
        buffer.clear("http")
        assert buffer._bytes == 0

    asyncio.run(scenario())


def test_capture_buffer_byte_eviction_is_independent_of_count(monkeypatch):
    async def scenario():
        buffer = RecentHTTPCaptures()
        first = capture()
        encoded = (
            CapturedHTTPExchange.model_validate(first).model_dump_json(exclude_none=True).encode()
        )
        monkeypatch.setattr(http_capture, "MAX_CAPTURE_BUFFER_BYTES", len(encoded) * 2)
        for number in range(1, 5):
            await buffer.put(capture(str(number)), "invocation-1")
        assert len(buffer._entries) == 2
        assert not buffer.contains("http", "2") and buffer.contains("http", "4")
        assert buffer._bytes <= http_capture.MAX_CAPTURE_BUFFER_BYTES
        oversized = copy.deepcopy(first)
        oversized["attempts"][0]["request"]["message"]["headers"] = [
            {"name": "X-Fill", "value": "x" * 1000}
        ]
        with pytest.raises(ValueError, match="budget"):
            await buffer.put(oversized, "invocation-1")
        assert len(buffer._entries) == 2

    asyncio.run(scenario())
