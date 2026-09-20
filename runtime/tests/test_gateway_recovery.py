"""Recovery retries one model call; it never reruns the surrounding invocation."""

import asyncio
import json

import httpx
import pytest

from contractor_runtime.llm.client import GatewayRequestError, new_gateway_client
from contractor_runtime.llm.recovery import RecoveryDecision


class Authority:
    def __init__(self, *, wait=False):
        self.events = []
        self.wait = wait
        self.observed = asyncio.Event()

    async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
        self.events.append((model, request_id, action, code, retry_after_seconds))
        self.observed.set()
        return RecoveryDecision(
            allowed=not self.wait,
            retry_after_seconds=60,
            request_timeout_seconds=1,
            requires_retry=self.wait,
        )


def test_model_unload_retries_only_identical_model_request():
    async def scenario():
        authority = Authority()
        requests = []

        async def gateway(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(400, json={"error": "Model is unloaded."})
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key="secret",
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            payload = {"model": "worker", "messages": [{"role": "tool", "content": "already done"}]}
            assert await handle.complete(payload) == {"ok": True}
        assert len(requests) == 2
        assert (
            requests[0].content
            == requests[1].content
            == json.dumps(payload, separators=(",", ":")).encode()
        )
        assert [event[2] for event in authority.events] == [
            "acquire",
            "failed",
            "acquire",
            "succeeded",
        ]
        assert authority.events[1][3] == "model_unavailable"
        assert authority.events[0][1] == authority.events[1][1]
        assert authority.events[2][1] == authority.events[3][1] != authority.events[0][1]
        assert "secret" not in repr(authority.events)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,body",
    [
        (400, {"error": "invalid messages"}),
        (401, {"error": "bad credential"}),
        (429, {"error": {"code": "insufficient_quota"}}),
    ],
)
def test_permanent_error_does_not_block_route_or_retry(status, body):
    async def scenario():
        authority = Authority()
        requests = []

        async def gateway(request):
            requests.append(request)
            return httpx.Response(status, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            with pytest.raises(GatewayRequestError) as caught:
                await handle.complete({"model": "worker"})
        assert not caught.value.retryable
        assert len(requests) == 1
        assert [event[2] for event in authority.events] == ["acquire", "finished"]

    asyncio.run(scenario())


def test_cancellation_interrupts_manual_recovery_wait_without_model_call():
    async def scenario():
        authority = Authority(wait=True)

        async def gateway(_request):
            pytest.fail("waiting invocation sent a model request")

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            task = asyncio.create_task(handle.complete({"model": "worker"}))
            await asyncio.wait_for(authority.observed.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        assert [event[2] for event in authority.events] == ["acquire"]

    asyncio.run(scenario())
