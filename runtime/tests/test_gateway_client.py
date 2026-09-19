from __future__ import annotations

import asyncio
from email.utils import formatdate

import httpx
import pytest

from contractor_runtime.llm.client import (
    GatewayClientClosedError,
    GatewayRequestError,
    new_gateway_client,
)

SECRET = "gateway-client-secret-canary"
PAYLOAD = {"model": "worker-model", "messages": [{"role": "user", "content": "go"}]}


@pytest.mark.parametrize(
    "status,headers,expected_delay",
    [
        (408, {}, 0.5),
        (409, {}, 0.5),
        (429, {"retry-after-ms": "1250", "retry-after": "3"}, 1.25),
        (429, {"retry-after-ms": "invalid", "retry-after": "2"}, 2.0),
        (503, {"retry-after": "0.25"}, 0.25),
        (503, {"retry-after": formatdate(1_700_000_002, usegmt=True)}, 2.0),
        (503, {"retry-after": "120"}, 120.0),
        (500, {"retry-after": "NaN"}, 0.5),
        (502, {"retry-after": "inf"}, 0.5),
        (504, {"retry-after": "-5"}, 0.5),
        (503, {"retry-after": "invalid"}, 0.5),
        (400, {"x-should-retry": "true"}, 0.5),
    ],
)
def test_gateway_retries_preserve_route_payload_and_server_delay(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    expected_delay: float,
) -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []
        responses: list[httpx.Response] = []
        delays: list[float] = []

        async def gateway(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            response = (
                httpx.Response(status, headers=headers, json={"error": {"message": SECRET}})
                if len(requests) == 1
                else httpx.Response(200, json={"ok": True})
            )
            responses.append(response)
            return response

        async def wait(seconds: float) -> None:
            assert responses[-1].is_closed
            delays.append(seconds)

        monkeypatch.setattr("contractor_runtime.llm.client.sleep", wait)
        monkeypatch.setattr("contractor_runtime.llm.client.random", lambda: 0)
        monkeypatch.setattr("contractor_runtime.llm.client.time", lambda: 1_700_000_000)
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.example/custom%2Froute/v1/",
                api_key=SECRET,
                timeout_seconds=7,
                http_client=http,
            )
            assert await handle.complete(PAYLOAD) == {"ok": True}
            assert delays == [expected_delay]
            assert len(requests) == 2
            assert requests[0].content == requests[1].content
            for index, request in enumerate(requests):
                assert request.method == "POST"
                assert request.url.raw_path == b"/custom%2Froute/v1/chat/completions"
                assert request.headers["authorization"] == f"Bearer {SECRET}"
                assert request.headers["x-stainless-retry-count"] == str(index)
                assert request.headers["content-type"] == "application/json"
                assert set(request.extensions["timeout"].values()) == {7}
            await handle.close()
            assert not http.is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,headers,retryable",
    [
        (401, {}, False),
        (503, {"x-should-retry": "false"}, True),
        (429, {"retry-after": "121", "x-should-retry": "true"}, True),
    ],
)
def test_transport_retry_suppression_keeps_final_failure_classification(
    status: int, headers: dict[str, str], retryable: bool
) -> None:
    async def scenario() -> None:
        calls = 0

        async def gateway(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status, headers=headers, text=SECRET)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=1,
                http_client=http,
            )
            with pytest.raises(GatewayRequestError) as captured:
                await handle.complete(PAYLOAD)
            assert calls == 1
            assert captured.value.retryable is retryable
            assert captured.value.__cause__ is None
            assert captured.value.__context__ is None
            assert SECRET not in repr(captured.value.__dict__)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["connect", "timeout", "http"])
def test_gateway_exhausts_bounded_attempts_without_retaining_provider_data(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    async def scenario() -> None:
        calls = 0
        delays: list[float] = []

        async def gateway(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if failure == "connect":
                raise httpx.ConnectError(SECRET, request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout(SECRET, request=request)
            return httpx.Response(503, json={"error": {"message": SECRET}})

        async def wait(seconds: float) -> None:
            delays.append(seconds)

        monkeypatch.setattr("contractor_runtime.llm.client.sleep", wait)
        monkeypatch.setattr("contractor_runtime.llm.client.random", lambda: 0)
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=1,
                http_client=http,
            )
            with pytest.raises(GatewayRequestError) as captured:
                await handle.complete(PAYLOAD)
            assert calls == 4
            assert delays == [0.5, 1.0, 2.0]
            assert (
                captured.value.provider_error_type
                == {
                    "connect": "APIConnectionError",
                    "timeout": "APITimeoutError",
                    "http": "InternalServerError",
                }[failure]
            )
            assert captured.value.retryable
            assert captured.value.__cause__ is None
            assert captured.value.__context__ is None
            assert SECRET not in repr(captured.value.__dict__)
            await handle.close()

    asyncio.run(scenario())


def test_closing_one_client_during_backoff_preserves_other_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        authorizations: list[str] = []
        waiting = asyncio.Event()
        resume = asyncio.Event()

        async def gateway(request: httpx.Request) -> httpx.Response:
            authorization = request.headers["authorization"]
            authorizations.append(authorization)
            return httpx.Response(503 if authorization == f"Bearer {SECRET}" else 200, json={})

        async def wait(_seconds: float) -> None:
            waiting.set()
            await resume.wait()

        monkeypatch.setattr("contractor_runtime.llm.client.sleep", wait)
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            first = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key=SECRET,
                timeout_seconds=1,
                http_client=http,
            )
            second = new_gateway_client(
                base_url="https://gateway.example/v1",
                api_key="second-allocation",
                timeout_seconds=1,
                http_client=http,
            )
            pending = asyncio.create_task(first.complete(PAYLOAD))
            await asyncio.wait_for(waiting.wait(), 1)
            await first.close()
            resume.set()
            with pytest.raises(GatewayClientClosedError):
                await asyncio.wait_for(pending, 1)
            assert await second.complete(PAYLOAD) == {}
            assert authorizations == [f"Bearer {SECRET}", "Bearer second-allocation"]
            assert "authorization" not in http.headers
            assert SECRET not in repr(first)
            await second.close()

    asyncio.run(scenario())
