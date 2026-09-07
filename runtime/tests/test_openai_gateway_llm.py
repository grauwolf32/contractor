from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from openai import AsyncOpenAI
from pydantic import BaseModel

from contractor_runtime.model_client import GatewayClientHandle, new_gateway_client
from contractor_runtime.openai_gateway_llm import (
    GatewayModelError,
    OpenAICompatibleGatewayLlm,
    _to_llm_response,
)

SECRET = "recognizable-openai-gateway-secret"


class StructuredResult(BaseModel):
    result: str
    optional_note: str | None = None


@pytest.mark.parametrize("arguments", [None, '{"path":', '{"path":"partial.txt"}'])
def test_output_limit_retains_usage_without_executable_tool_calls(arguments: str | None) -> None:
    calls = (
        []
        if arguments is None
        else [
            SimpleNamespace(
                id="call-truncated",
                function=SimpleNamespace(name="write_file", arguments=arguments),
            )
        ]
    )
    response = _to_llm_response(
        SimpleNamespace(
            model="worker-model",
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content=None, tool_calls=calls),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=32, total_tokens=44),
        )
    )
    assert response.finish_reason == types.FinishReason.MAX_TOKENS
    assert response.error_code == "MAX_TOKENS"
    assert response.usage_metadata.total_token_count == 44
    assert not response.content.parts


def test_adk_request_and_gateway_response_round_trip() -> None:
    async def scenario() -> None:
        captured: list[dict[str, Any]] = []

        async def gateway(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            assert request.headers["authorization"] == f"Bearer {SECRET}"
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": "chatcmpl-contract",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "resolved-worker-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "bounded thought",
                                "content": "checking",
                                "tool_calls": [
                                    {
                                        "id": "call-next",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"path":"next.py"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 31,
                        "completion_tokens": 7,
                        "total_tokens": 38,
                        "prompt_tokens_details": {"cached_tokens": 11},
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                },
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(gateway),
            trust_env=False,
        )
        model = OpenAICompatibleGatewayLlm(
            model="worker-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=120,
                http_client=http_client,
            ),
        )
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text="first"), types.Part(text="second")],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(text="prior thought", thought=True),
                        types.Part(text="prior answer"),
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call-prior", name="lookup", args={"path": "old.py"}
                            )
                        ),
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="call-prior",
                                name="lookup",
                                response={"contents": "old"},
                            )
                        )
                    ],
                ),
                types.Content(role="user", parts=[types.Part(text="continue")]),
            ],
            config=types.GenerateContentConfig(
                system_instruction="system contract",
                max_output_tokens=321,
                temperature=0,
                top_p=0.8,
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name="lookup",
                                description="Read one path",
                                parameters_json_schema={
                                    "type": "OBJECT",
                                    "properties": {"path": {"type": "STRING"}},
                                    "required": ["path"],
                                },
                            )
                        ]
                    )
                ],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["lookup"],
                    )
                ),
                response_schema=StructuredResult,
            ),
        )

        responses = [item async for item in model.generate_content_async(request)]
        assert len(responses) == 1
        response = responses[0]
        assert response.model_version == "resolved-worker-model"
        assert response.finish_reason == types.FinishReason.STOP
        assert [(part.text, part.thought) for part in response.content.parts[:2]] == [
            ("bounded thought", True),
            ("checking", None),
        ]
        function_call = response.content.parts[2].function_call
        assert function_call.id == "call-next"
        assert function_call.name == "lookup"
        assert function_call.args == {"path": "next.py"}
        assert response.usage_metadata.prompt_token_count == 31
        assert response.usage_metadata.candidates_token_count == 7
        assert response.usage_metadata.total_token_count == 38
        assert response.usage_metadata.cached_content_token_count == 11
        assert response.usage_metadata.thoughts_token_count == 3

        payload = captured[0]
        assert payload["model"] == "worker-model"
        assert payload["max_tokens"] == 321
        assert payload["temperature"] == 0
        assert payload["top_p"] == 0.8
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert payload["messages"][1]["content"] == "first\nsecond"
        assert payload["messages"][2]["reasoning_content"] == "prior thought"
        assert payload["messages"][2]["tool_calls"][0]["id"] == "call-prior"
        assert json.loads(payload["messages"][3]["content"]) == {"contents": "old"}
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "lookup"},
        }
        assert payload["tools"][0]["function"]["parameters"]["type"] == "object"
        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["optional_note", "result"]

        await model.close()
        assert model.closed
        assert not http_client.is_closed
        await http_client.aclose()

    asyncio.run(scenario())


def test_sdk_retries_three_connection_failures_with_one_logical_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        attempts = 0

        async def no_delay(_seconds: float) -> None:
            return None

        async def flaky_gateway(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                raise httpx.ConnectError("connection was lost", request=request)
            return completion_response(request, content="recovered")

        monkeypatch.setattr("openai._base_client.anyio.sleep", no_delay)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(flaky_gateway),
            trust_env=False,
        )
        handle = new_gateway_client(
            base_url="https://gateway.example/v1",
            api_key=SECRET,
            timeout_seconds=120,
            http_client=http_client,
        )
        model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=handle)

        assert SECRET not in repr(handle)

        responses = [
            item
            async for item in model.generate_content_async(
                LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="go")])])
            )
        ]

        assert attempts == 4
        assert len(responses) == 1
        assert responses[0].content.parts[0].text == "recovered"
        assert handle.client.max_retries == 3
        assert handle.operation_timeout_seconds == 180
        await model.close()
        await http_client.aclose()

    asyncio.run(scenario())


def test_unsupported_inline_content_and_part_metadata_fail_closed() -> None:
    async def scenario() -> None:
        async def unreachable(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("invalid content must not reach the Gateway")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(unreachable),
            trust_env=False,
        )
        model = OpenAICompatibleGatewayLlm(
            model="worker-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=120,
                http_client=http_client,
            ),
        )
        requests = (
            LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(inline_data=types.Blob(mime_type="image/png", data=b"x"))
                        ],
                    )
                ]
            ),
            LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text="signed", thought_signature=b"opaque")],
                    )
                ]
            ),
            LlmRequest(
                config=types.GenerateContentConfig(
                    system_instruction=types.Content(
                        role="system",
                        parts=[
                            types.Part(text="visible"),
                            types.Part(inline_data=types.Blob(mime_type="image/png", data=b"x")),
                        ],
                    )
                )
            ),
        )
        for request in requests:
            with pytest.raises(GatewayModelError) as captured:
                async for _ in model.generate_content_async(request):
                    pass
            assert captured.value.provider_error_type in {
                "UnsupportedContentPart",
                "UnsupportedSystemInstruction",
            }
        await model.close()
        await http_client.aclose()

    asyncio.run(scenario())


def test_gateway_errors_are_secret_free_and_external_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        async def no_delay(_seconds: float) -> None:
            return None

        async def leaking_gateway(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError(f"provider rejected Bearer {SECRET}")

        monkeypatch.setattr("openai._base_client.anyio.sleep", no_delay)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(leaking_gateway),
            trust_env=False,
        )
        model = OpenAICompatibleGatewayLlm(
            model="worker-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=120,
                http_client=http_client,
            ),
        )
        with pytest.raises(GatewayModelError) as captured:
            async for _ in model.generate_content_async(LlmRequest()):
                pass
        assert captured.value.provider_error_type == "APIConnectionError"
        assert captured.value.__context__ is None
        assert SECRET not in repr(captured.value)
        await model.close()
        await http_client.aclose()

        async def cancelled_gateway(_request: httpx.Request) -> httpx.Response:
            raise asyncio.CancelledError

        cancelled_http = httpx.AsyncClient(
            transport=httpx.MockTransport(cancelled_gateway),
            trust_env=False,
        )
        cancelled_model = OpenAICompatibleGatewayLlm(
            model="worker-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=120,
                http_client=cancelled_http,
            ),
        )
        with pytest.raises(asyncio.CancelledError):
            async for _ in cancelled_model.generate_content_async(LlmRequest()):
                pass
        await cancelled_model.close()
        await cancelled_http.aclose()

    asyncio.run(scenario())


def test_owned_client_cleanup_is_idempotent_and_aggregate_deadline_is_safe() -> None:
    async def scenario() -> None:
        async def blocked_gateway(_request: httpx.Request) -> httpx.Response:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(blocked_gateway),
            trust_env=False,
        )
        openai_client = AsyncOpenAI(
            api_key=SECRET,
            base_url="https://gateway.example/v1",
            max_retries=3,
            http_client=http_client,
        )
        handle = GatewayClientHandle(openai_client, True, 0.01)
        model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=handle)

        with pytest.raises(GatewayModelError) as captured:
            async for _ in model.generate_content_async(LlmRequest()):
                pass
        assert captured.value.provider_error_type == "TimeoutError"
        await model.close()
        await model.close()
        assert model.closed
        assert http_client.is_closed

        with pytest.raises(GatewayModelError) as closed:
            async for _ in model.generate_content_async(LlmRequest()):
                pass
        assert closed.value.provider_error_type == "GatewayClientClosedError"

    asyncio.run(scenario())


def test_json_object_fallback_and_non_stop_finish_reason() -> None:
    async def scenario() -> None:
        captured: list[dict[str, Any]] = []

        async def gateway(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return completion_response(request, content='{"partial":true}', finish_reason="length")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(gateway),
            trust_env=False,
        )
        model = OpenAICompatibleGatewayLlm(
            model="worker-model",
            client_handle=new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=None,
                timeout_seconds=120,
                http_client=http_client,
            ),
        )
        [response] = [
            item
            async for item in model.generate_content_async(
                LlmRequest(
                    contents=[types.Content(role="user", parts=[types.Part(text="json")])],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            )
        ]
        assert captured[0]["response_format"] == {"type": "json_object"}
        assert response.finish_reason == types.FinishReason.MAX_TOKENS
        assert response.error_code == "MAX_TOKENS"
        await model.close()
        await http_client.aclose()

    asyncio.run(scenario())


def completion_response(
    request: httpx.Request,
    *,
    content: str,
    finish_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "worker-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (400, "context_length_exceeded", False),
        (401, "invalid_api_key", False),
        (403, None, False),
        (404, None, False),
        (413, None, False),
        (422, None, False),
        (408, None, True),
        (409, None, True),
        (429, "rate_limit_exceeded", True),
        (429, "insufficient_quota", False),
        (429, "budget_exceeded", False),
        (500, None, True),
        (502, None, True),
        (503, None, True),
        (504, None, True),
    ],
)
def test_gateway_http_failures_preserve_safe_retryability(status, code, retryable) -> None:
    async def scenario() -> None:
        async def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status, json={"error": {"message": SECRET, "code": code}}, request=request
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        handle = new_gateway_client(
            base_url="https://gateway.example/v1",
            api_key=SECRET,
            timeout_seconds=120,
            http_client=http_client,
        )
        handle.client.max_retries = 0
        model = OpenAICompatibleGatewayLlm(model="worker-model", client_handle=handle)
        try:
            with pytest.raises(GatewayModelError) as captured:
                async for _ in model.generate_content_async(LlmRequest()):
                    pass
            assert captured.value.retryable is retryable
            assert captured.value.__context__ is None
            assert captured.value.__cause__ is None
            assert SECRET not in repr(captured.value.__dict__)
        finally:
            await model.close()
            await http_client.aclose()

    asyncio.run(scenario())
