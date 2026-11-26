from __future__ import annotations

import asyncio
import gzip
import json
import threading
import zlib
from pathlib import Path

import httpx
import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

import contractor_runtime.llm.client as gateway
import contractor_runtime.llm.openai as adapter


class Stream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self.data = data
        self.closed = False
        self.received = 0

    async def __aiter__(self):
        for start in range(0, len(self.data), 8192):
            chunk = self.data[start : start + 8192]
            self.received += len(chunk)
            yield chunk

    async def aclose(self):
        self.closed = True


def client(http):
    handle = gateway.new_gateway_client(
        base_url="https://gateway.example/v1",
        api_key="private-canary",
        timeout_seconds=5,
        http_client=http,
    )
    handle.max_retries = 0
    return handle


def test_shared_gateway_response_byte_ceiling():
    fixture = Path(__file__).parents[2] / "api/testdata/v1alpha1/gateway-response-limits.json"
    maximum = json.loads(fixture.read_bytes())["max_response_bytes"]
    assert maximum == gateway.MAX_GATEWAY_RESPONSE_BYTES

    async def scenario():
        body = b'"' + b"x" * (maximum - 2) + b'"'
        stream = Stream(body)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ) as http:
            assert len(await client(http).complete({"max_tokens": 1})) == maximum - 2
        assert stream.closed and stream.received == maximum

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("encoding", ["identity", "gzip", "deflate"])
@pytest.mark.parametrize("length", [None, "1"])
def test_stream_overflow_is_terminal_before_json_parsing(monkeypatch, status, encoding, length):
    async def scenario():
        monkeypatch.setattr(gateway, "MAX_GATEWAY_RESPONSE_BYTES", 1024)
        body = b'"' + b"x" * (4 * 1024 * 1024) + b'"'
        encoded = {"identity": lambda b: b, "gzip": gzip.compress, "deflate": zlib.compress}
        stream = Stream(encoded[encoding](body))
        calls = []

        def respond(request):
            calls.append(request)
            headers = {"content-encoding": encoding, "x-should-retry": "true"}
            if length is not None:
                headers["content-length"] = length
            return httpx.Response(status, headers=headers, stream=stream)

        def forbidden_parse(*args):
            raise AssertionError("oversized body reached JSON parsing")

        monkeypatch.setattr(gateway, "strict_json_loads", forbidden_parse)
        monkeypatch.setattr(json, "loads", forbidden_parse)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            handle = client(http)
            handle.max_retries = 3
            with pytest.raises(gateway.GatewayRequestError) as caught:
                await handle.complete({"max_tokens": 1})
            assert caught.value.provider_error_type == "InvalidGatewayResponse"
            assert not caught.value.retryable
            assert caught.value.__context__ is None
        assert stream.closed and len(calls) == 1
        assert stream.received < len(body)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [b"{", b'"\xff"', b"[" * 20_000 + b"]" * 20_000],
    ids=["malformed", "invalid-utf8", "deep"],
)
def test_malformed_or_deep_success_is_a_permanent_protocol_failure(body):
    async def scenario():
        stream = Stream(body)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ) as http:
            with pytest.raises(gateway.GatewayRequestError) as caught:
                await client(http).complete({})
            assert caught.value.provider_error_type == "InvalidGatewayResponse"
            assert not caught.value.retryable
        assert stream.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["success-json", "error-json", "adk-projection"])
def test_response_cpu_work_runs_off_loop_and_joins_repeated_cancellation(monkeypatch, phase):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        original = (
            adapter._to_llm_response
            if phase == "adk-projection"
            else gateway.strict_json_loads
            if phase == "success-json"
            else json.loads
        )

        def blocked(*args):
            assert threading.get_ident() != loop_thread
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(5), "test did not release the CPU operation"
                return original(*args)
            finally:
                finished.set()

        if phase == "adk-projection":
            monkeypatch.setattr(adapter, "_to_llm_response", blocked)
        else:
            if phase == "success-json":
                monkeypatch.setattr(gateway, "strict_json_loads", blocked)
            else:
                monkeypatch.setattr(json, "loads", blocked)
        stream = Stream(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}')
        status = 400 if phase == "error-json" else 200
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status, stream=stream))
        ) as http:
            handle = client(http)

            async def complete():
                if phase != "adk-projection":
                    return await handle.complete({})
                model = adapter.OpenAICompatibleGatewayLlm(model="worker", client_handle=handle)
                return [
                    response
                    async for response in model.generate_content_async(
                        LlmRequest(
                            contents=[types.Content(role="user", parts=[types.Part(text="go")])]
                        )
                    )
                ]

            pending = asyncio.create_task(complete())
            ready = asyncio.create_task(started.wait())
            try:
                await asyncio.wait({pending, ready}, return_when=asyncio.FIRST_COMPLETED)
                if pending.done():
                    pending.result()
                await ready
                for _ in range(3):
                    pending.cancel()
                    await asyncio.sleep(0)
                    assert not pending.done()
                assert not finished.is_set()
            finally:
                release.set()
                ready.cancel()
                await asyncio.gather(ready, return_exceptions=True)
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert finished.is_set() and stream.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["read", "eof-close"])
def test_cancelled_stream_read_joins_close_through_further_cancellation(phase):
    async def scenario():
        started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Blocked(Stream):
            close_calls = 0

            async def __aiter__(self):
                if phase == "read":
                    started.set()
                    await asyncio.Event().wait()
                yield b"{}"

            async def aclose(self):
                self.close_calls += 1
                if phase == "eof-close" and self.close_calls == 1:
                    started.set()
                else:
                    closing.set()
                await release.wait()
                self.closed = True

        stream = Blocked(b"")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ) as http:
            pending = asyncio.create_task(client(http).complete({}))
            await started.wait()
            pending.cancel()
            await closing.wait()
            for _ in range(3):
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done() and not stream.closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert stream.closed

    asyncio.run(scenario())


def test_oversized_response_settles_recovery_as_finished_without_replaying():
    from test_gateway_recovery import Authority

    async def scenario():
        authority = Authority()
        stream = Stream(b"x" * (gateway.MAX_GATEWAY_RESPONSE_BYTES + 1))
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(503, stream=stream))
        ) as http:
            handle = client(http)
            handle.recovery = authority
            with pytest.raises(gateway.GatewayRequestError) as caught:
                await handle.complete({"model": "worker"})
            assert not caught.value.retryable
        assert [event[2] for event in authority.events] == ["acquire", "finished"]
        assert authority.events[0][1] == authority.events[1][1]
        assert stream.closed

    asyncio.run(scenario())
