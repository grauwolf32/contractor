"""Structured JSON and exact text through actual ADK and three socket routes."""

import asyncio
import base64
import json

import pytest
from fakes.tool_arguments import http_receiver, received, run_calls

from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.toolsets.http.tools import HTTPRequestTool
from contractor_runtime.worker.function_tools import ContractorFunctionTool


@pytest.mark.parametrize("route", ["direct", "proxy", "controlled"])
@pytest.mark.parametrize(
    "value",
    [
        {"amount": "8100", "valid": True, "nested": [None, 1.5, "{}"]},
        ["01", False, {"note": "雪\n"}],
    ],
)
def test_structured_json_is_serialized_once_and_capture_matches_wire(tmp_path, route, value):
    async def scenario():
        async with http_receiver(tmp_path, route) as (tool, state, url, requests):

            async def inspect(results):
                for result, request in zip(results, requests, strict=True):
                    capture = await tool._session._captures.get("http", str(result["request_id"]))
                    body = capture["attempts"][0]["request"]["message"]["body_base64"]
                    assert base64.b64decode(body) == received(request)[2]

            results = await run_calls(
                tmp_path,
                tool,
                state,
                [
                    {"url": url, "method": "POST", "body_type": "json", "body": body}
                    for body in (value, json.dumps(value))
                ],
                inspect_results=inspect,
            )
            assert len(requests) == 2
            for result, request in zip(results, requests, strict=True):
                assert result["status"] == 200
                _, headers, body = received(request)
                assert headers["content-type"] == "application/json"
                assert json.loads(body) == value
            assert received(requests[0])[2] == received(requests[1])[2]
            assert results[1]["normalization"]["fields"][0]["field"] == "body"

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "proxy", "controlled"])
@pytest.mark.parametrize(
    "value",
    [
        ' "value" ',
        "42",
        "true",
        "null",
        ' {"x":1,"x":2} ',
        '\r\n{"broken": 雪\\n\t',
        '{"space": "雪\\t"}\r\n',
    ],
)
def test_text_retains_every_utf8_byte_even_for_invalid_json(tmp_path, route, value):
    async def scenario():
        async with http_receiver(tmp_path, route, status=400) as (tool, state, url, requests):

            async def inspect(results):
                capture = await tool._session._captures.get("http", str(results[0]["request_id"]))
                encoded = capture["attempts"][0]["request"]["message"]["body_base64"]
                assert base64.b64decode(encoded) == value.encode()

            (result,) = await run_calls(
                tmp_path,
                tool,
                state,
                [
                    {
                        "url": url,
                        "method": "POST",
                        "body_type": "text",
                        "body": value,
                        "headers": {"content-type": "application/json"},
                    }
                ],
                inspect_results=inspect,
            )
            assert result["status"] == 400 and "normalization" not in result
            assert len(requests) == 1  # A target 400 never triggers a semantic resend.
            _, headers, body = received(requests[0])
            assert body == value.encode() and headers["content-type"] == "application/json"
            assert state.metrics.counters.get("tool_errors", 0) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["direct", "proxy", "controlled"])
@pytest.mark.parametrize(
    "mode,value",
    [
        ("json", None),
        ("json", 1),
        ("json", True),
        ("json", "null"),
        ("json", '"value"'),
        ("json", '"{}"'),
        ("json", '{"x":1,"x":2}'),
        ("json", '{"x":1e999}'),
        ("json", '{"bad":'),
        ("text", {}),
        ("text", "\ud800"),
        ("none", ""),
        ("form", None),
    ],
)
def test_invalid_body_has_precise_error_before_any_send(tmp_path, route, mode, value):
    async def scenario():
        async with http_receiver(tmp_path, route) as (tool, state, url, requests):
            (result,) = await run_calls(
                tmp_path,
                tool,
                state,
                [
                    {
                        "url": url,
                        "method": "POST",
                        "body_type": mode,
                        "body": value,
                    }
                ],
            )
            assert result["error"]["code"] == "http_request_invalid"
            assert result["error"]["field"] == "body"
            assert not result["error"]["retryable"] and not requests
            if mode == "json":
                assert "body_type=text" in result["error"]["message"]

    asyncio.run(scenario())


def test_effective_body_bound_applies_to_representation_and_encoded_bytes(tmp_path):
    async def scenario():
        async with http_receiver(
            tmp_path, "controlled", limits=HTTPLimits(requestBodyBytes=16)
        ) as (tool, state, url, requests):
            bodies = [
                ("json", {"x": "éééé"}),  # Exactly 16 encoded bytes.
                ("json", {"x": "ééééx"}),
                ("json", ' {"x": "éééé"} '),  # Small decoded result, oversized representation.
                ("json", {"x": "\n" * 5}),
                ("text", "é" * 8),
                ("text", "é" * 8 + "x"),
                ("form", {"x": " " * 16}),
            ]
            results = await run_calls(
                tmp_path,
                tool,
                state,
                [
                    {"url": url, "method": "POST", "body_type": mode, "body": body}
                    for mode, body in bodies
                ],
            )
            assert [r.get("status") for r in results] == [200, None, None, None, 200, None, None]
            assert [len(received(r)[2]) for r in requests] == [16, 16]
            assert all(r["error"]["field"] == "body" for r in results if "error" in r)

    asyncio.run(scenario())


def test_body_declaration_has_explicit_native_and_serialized_shapes():
    tool = ContractorFunctionTool(
        HTTPRequestTool(None, None), logical_invocation_id=lambda _: "test"
    )
    declaration = tool._get_declaration()
    variants = declaration.parameters_json_schema["properties"]["body"]["anyOf"]
    assert {variant["type"] for variant in variants} == {"object", "array", "string", "null"}
    assert "duplicate keys" in declaration.description
    assert "exact UTF-8" in declaration.description
