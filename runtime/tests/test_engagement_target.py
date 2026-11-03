"""Domain admission, limits and retry decisions with an engine double."""

import asyncio
import ipaddress
import ssl

import httpcore
import pytest
from fakes.engagement import preflight_spec
from fakes.engagement_engine import transport as HTTPTransport

from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.target_policy import TargetPolicyConfig, TargetUnresolved


def owner(network, **options):
    spec = preflight_spec()
    spec.engagement.http.limits = options.pop("limits", None)
    resource = HTTPTransport(spec, resolver=policy_config().resolver, backend=network, **options)
    resource.configure(authority_deadline=lambda: resource._clock() + 60)
    return resource


def start(resource, url="http://target.example/start", **options):
    resource.configure(authority_deadline=lambda: resource._clock() + 60)

    async def execute():
        outcome, _ = await resource.request(url, **options)
        return outcome

    return asyncio.create_task(execute())


class Stream(httpcore.AsyncNetworkStream):
    def __init__(
        self,
        events,
        response,
        *,
        fail_write=False,
        hang=False,
        tls_fail=False,
        wrong_peer=False,
        close_fail=False,
    ):
        self.events, self.response = events, response
        self.fail_write, self.hang, self.tls_fail = fail_write, hang, tls_fail
        self.entered_read = asyncio.Event()
        self.closed = False
        self.written = bytearray()
        self.wrong_peer, self.close_fail = wrong_peer, close_fail

    async def read(self, max_bytes, timeout=None):
        self.entered_read.set()
        if self.fail_write:
            raise httpcore.ReadError("connection closed after partial write")
        if self.hang:
            await asyncio.Event().wait()
        data, self.response = self.response[:max_bytes], self.response[max_bytes:]
        return data

    async def write(self, buffer, timeout=None):
        self.events.append(buffer)
        self.written.extend(buffer)
        if self.fail_write:
            raise httpcore.WriteError("partial application write")

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.events.append("tls")
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        if self.tls_fail:
            raise httpcore.ConnectError("TLS verification failed") from ssl.SSLError()
        return self

    async def aclose(self):
        if self.close_fail:
            raise RuntimeError("transport close failed")
        self.closed = True

    def get_extra_info(self, name):
        return (
            ("10.20.0.9" if self.wrong_peer else "10.20.0.8", 80) if name == "server_addr" else None
        )


class Network(httpcore.AsyncNetworkBackend):
    def __init__(self, events, responses=None, *, fail_connect=False, **stream_options):
        self.events = events
        self.responses = responses or [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]
        self.options, self.streams, self.fail_connect = stream_options, [], fail_connect

    async def connect_tcp(self, host, port, **kwargs):
        assert host == "10.20.0.8"
        self.events.append("connect")
        if self.fail_connect:
            raise httpcore.ConnectError("connect failed")
        stream = Stream(
            self.events,
            self.responses[min(len(self.streams), len(self.responses) - 1)],
            **self.options,
        )
        self.streams.append(stream)
        return stream


def policy_config(target=("10.20.0.8",), *, extra=None):
    async def resolver(host, port):
        if extra is not None and host in extra:
            values = extra[host]
        elif host == "target.example":
            values = target
        else:
            values = ("10.30.0.1",)
        if values is None:
            raise TargetUnresolved
        return tuple(ipaddress.ip_address(value) for value in values)

    return TargetPolicyConfig(resolver=resolver)


def test_local_http_sends_complete_request_and_bounds_response():
    async def scenario():
        network = Network([], [b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nsecretxx"])
        resource = owner(network, limits=HTTPLimits(responseBodyBytes=3))
        outcome, response = await resource.request("http://target.example/start")
        assert outcome.kind == "completed"
        assert response.body == b"sec" and response.truncated
        assert network.streams[0].written.startswith(b"GET /start HTTP/1.1\r\n")
        assert network.streams[0].closed
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,attempts",
    [("GET", 2), ("HEAD", 2), ("OPTIONS", 2), ("POST", 1), ("PUT", 1), ("PATCH", 1), ("DELETE", 1)],
)
def test_only_safe_methods_retry_a_connection_failure_before_write(method, attempts):
    async def scenario():
        events = []
        resource = owner(Network(events, fail_connect=True))
        outcome, _ = await resource.request("http://target.example/start", method=method)
        assert outcome.kind == "failed"
        assert events.count("connect") == attempts
        assert resource._request_count == attempts
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_partial_application_write_never_retries(method):
    async def scenario():
        network = Network([], fail_write=True)
        resource = owner(network)
        outcome, _ = await resource.request("http://target.example/start", method=method)
        assert outcome.kind == "failed"
        assert len(network.streams) == resource._request_count == 1
        assert network.streams[0].written and network.streams[0].closed
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_http_status_does_not_trigger_automatic_retry(status):
    async def scenario():
        resource = owner(
            Network([], [f"HTTP/1.1 {status} Status\r\nContent-Length: 0\r\n\r\n".encode()])
        )
        outcome, response = await resource.request("http://target.example/start")
        assert outcome.kind == "completed" and response.status == status
        assert resource._request_count == 1
        await resource.close()

    asyncio.run(scenario())


def test_slot_budget_and_rate_reset_with_each_new_owner():
    class Clock:
        def __init__(self):
            self.now = 100.0
            self.waits = []

        def __call__(self):
            return self.now

        async def sleep(self, delay):
            self.waits.append(delay)
            self.now += delay

    async def scenario():
        clock = Clock()
        limits = HTTPLimits(targetRequests=3, requestsPerSecond=2)
        for _ in range(2):
            resource = owner(Network([]), limits=limits, monotonic=clock, sleep=clock.sleep)
            for _ in range(3):
                assert (await resource.request("http://target.example/start"))[
                    0
                ].kind == "completed"
            with pytest.raises(HTTPTransportError, match="http_limit_exceeded"):
                await resource.request("http://target.example/start")
            assert resource._request_count == 3
            await resource.close()
        assert clock.waits == [0.5] * 4

    asyncio.run(scenario())


def test_oversized_request_is_rejected_before_connecting():
    async def scenario():
        network = Network([])
        resource = owner(network, limits=HTTPLimits(requestBodyBytes=3))
        with pytest.raises(ValueError):
            await resource.request("http://target.example/start", method="POST", body=b"1234")
        assert resource._request_count == 0 and not network.streams
        await resource.close()

    asyncio.run(scenario())


def test_omitted_http_limits_remain_unconfigured():
    async def scenario():
        resource = owner(Network([]))
        assert resource._http_limits.model_dump(exclude_none=True) == {}
        for _ in range(3):
            assert (await resource.request("http://target.example/start"))[0].kind == "completed"
        assert resource._next_request_at == 0.0
        await resource.close()

    asyncio.run(scenario())
