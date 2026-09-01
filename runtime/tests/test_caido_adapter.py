from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from contractor_runtime.adapters.caido_graphql import (
    MAX_CAIDO_RESPONSE_BYTES,
    CaidoClientError,
    CaidoCloseError,
    CaidoGraphQLAdapter,
    CaidoGraphQLAdapterFactory,
    CaidoGraphQLClient,
)
from contractor_runtime.adapters.host import RuntimeAdapterBuildContext
from contractor_runtime.contracts import CaidoSettingsV2

CAIDO_TOKEN = "recognizable-caido-bearer-secret"
CAIDO_ENDPOINT = "https://caido.internal/prefix"
SERVER_DETAIL = "recognizable-caido-server-error"


def test_static_authenticated_and_guest_operations_are_exact_and_detach() -> None:
    async def scenario() -> None:
        authenticated_requests: list[httpx.Request] = []

        async def authenticated_response(request: httpx.Request) -> httpx.Response:
            authenticated_requests.append(request)
            return httpx.Response(
                200,
                json={"data": {"scopes": [{"id": "1", "name": "target"}]}},
            )

        adapter = CaidoGraphQLAdapter(
            adapter_context(),
            caido_settings(token=CAIDO_TOKEN),
            transport=httpx.MockTransport(authenticated_response),
        )
        handle = adapter.handles.caido_graphql
        assert isinstance(handle, CaidoGraphQLClient)
        result = await handle.execute("scopes")
        assert result == {"scopes": [{"id": "1", "name": "target"}]}
        assert len(authenticated_requests) == 1
        request = authenticated_requests[0]
        assert str(request.url) == f"{CAIDO_ENDPOINT}/graphql"
        assert request.headers["authorization"] == f"Bearer {CAIDO_TOKEN}"
        payload = json.loads(request.content)
        assert payload == {
            "operationName": "Scopes",
            "query": "query Scopes { scopes { id name allowlist denylist } }",
            "variables": {},
        }
        with pytest.raises(CaidoClientError) as unknown:
            await handle.execute("arbitrary-query", {"query": "{ secrets }"})
        assert unknown.value.code == "caido_request_invalid"
        assert len(authenticated_requests) == 1
        assert adapter.metrics.operations == 1

        rendered = f"{adapter!r} {handle!r} {unknown.value!r}"
        assert CAIDO_TOKEN not in rendered
        assert CAIDO_ENDPOINT not in rendered
        await adapter.close()
        assert handle.closed
        assert adapter.handles.enabled_channels == ()
        with pytest.raises(CaidoClientError):
            await handle.execute("scopes")

        guest_requests: list[httpx.Request] = []

        async def guest_response(request: httpx.Request) -> httpx.Response:
            guest_requests.append(request)
            return httpx.Response(200, json={"data": {"scopes": []}})

        guest = CaidoGraphQLAdapter(
            adapter_context(),
            caido_settings(token=None),
            transport=httpx.MockTransport(guest_response),
        )
        await guest.handles.caido_graphql.execute("scopes")
        assert "authorization" not in guest_requests[0].headers
        await guest.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "code", "retryable"),
    [
        (lambda: httpx.Response(503, text=SERVER_DETAIL), "caido_request_failed", True),
        (lambda: httpx.Response(401, text=SERVER_DETAIL), "caido_request_failed", False),
        (lambda: httpx.Response(200, content=b"not-json"), "caido_response_invalid", False),
        (
            lambda: httpx.Response(200, json={"errors": [{"message": SERVER_DETAIL}]}),
            "caido_request_failed",
            False,
        ),
        (
            lambda: httpx.Response(
                200,
                headers={"content-length": str(MAX_CAIDO_RESPONSE_BYTES + 1)},
                content=b"{}",
            ),
            "caido_response_too_large",
            False,
        ),
    ],
)
def test_failures_are_classified_bounded_and_content_free(
    response: Callable[[], httpx.Response],
    code: str,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        adapter = CaidoGraphQLAdapter(
            adapter_context(),
            caido_settings(token=CAIDO_TOKEN),
            transport=httpx.MockTransport(lambda _request: response()),
        )
        handle = adapter.handles.caido_graphql
        with pytest.raises(CaidoClientError) as failure:
            await handle.execute("scopes")
        assert failure.value.code == code
        assert failure.value.retryable is retryable
        rendered = f"{failure.value!s} {failure.value!r} {adapter!r} {adapter.metrics!r}"
        for forbidden in (CAIDO_TOKEN, CAIDO_ENDPOINT, SERVER_DETAIL):
            assert forbidden not in rendered
        assert adapter.metrics.operations == 1
        assert adapter.metrics.failed_operations == 1
        assert adapter.metrics.last_error_code == "request_failed"
        await adapter.close()

    asyncio.run(scenario())


def test_variable_bounds_and_close_failure_detach_before_error() -> None:
    async def scenario() -> None:
        transport = FailingCloseTransport()
        adapter = CaidoGraphQLAdapter(
            adapter_context(),
            caido_settings(token=CAIDO_TOKEN),
            transport=transport,
        )
        handle = adapter.handles.caido_graphql
        with pytest.raises(CaidoClientError) as oversized:
            await handle.execute("scopes", {"filter": "x" * (1024 * 1024 + 1)})
        assert oversized.value.code == "caido_request_invalid"
        assert transport.requests == 0
        with pytest.raises(CaidoClientError):
            await handle.execute("scopes", {"bad": float("nan")})
        assert transport.requests == 0
        with pytest.raises(CaidoCloseError):
            await adapter.close()
        assert handle.closed
        assert adapter.handles.enabled_channels == ()
        rendered = f"{adapter!r} {handle!r}"
        assert CAIDO_TOKEN not in rendered
        assert CAIDO_ENDPOINT not in rendered

    asyncio.run(scenario())


def test_factory_probe_and_wrong_settings_are_safe() -> None:
    async def scenario() -> None:
        factory = CaidoGraphQLAdapterFactory()
        assert await factory.probe()
        with pytest.raises(Exception) as failure:
            await factory.create(adapter_context(), object())  # type: ignore[arg-type]
        assert CAIDO_TOKEN not in repr(failure.value)

    asyncio.run(scenario())


class FailingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(200, json={"data": {"scopes": []}}, request=request)

    async def aclose(self) -> None:
        raise RuntimeError(f"close failed: {CAIDO_TOKEN}")


def adapter_context() -> RuntimeAdapterBuildContext:
    return RuntimeAdapterBuildContext(
        allocation_id="allocation-caido",
        run_id="run-caido",
        stage_execution_id="stage-caido",
        logical_agent_name="analyst",
        request_timeout_seconds=5,
        runtime_config_refs=("caido-lab@1",),
        runtime_config_digests=("sha256:" + "a" * 64,),
        run_labels=("caido",),
        agent_labels=(),
        runtime_adapter_refs=("caido-graphql@1",),
        private_bypass_hosts=("127.0.0.1", "localhost"),
    )


def caido_settings(*, token: str | None) -> CaidoSettingsV2:
    return CaidoSettingsV2(
        adapter="caido-graphql@1",
        endpoint=CAIDO_ENDPOINT,
        bearerToken=token,
        requestTimeoutSeconds=4,
    )
