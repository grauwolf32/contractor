"""Domain admission, limits and retry decisions with an engine double."""

import asyncio
import ipaddress

import pytest
from fakes.engagement import preflight_spec
from fakes.engagement_engine import Results, reply
from fakes.engagement_engine import transport as HTTPTransport

from contractor_runtime.adapters.bridge import BridgeFailure
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.target_policy import TargetPolicyConfig, TargetUnresolved


def owner(results, **options):
    spec = preflight_spec()
    spec.engagement.http.limits = options.pop("limits", None)
    resource = HTTPTransport(spec, resolver=policy_config().resolver, engine=results, **options)
    resource.configure(authority_deadline=lambda: resource._clock() + 60)
    return resource


def start(resource, url="http://target.example/start", **options):
    resource.configure(authority_deadline=lambda: resource._clock() + 60)

    async def execute():
        outcome, _ = await resource.request(url, **options)
        return outcome

    return asyncio.create_task(execute())


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


def test_domain_preserves_given_bounded_engine_result():
    async def scenario():
        results = Results([reply(body=b"sec", truncated=True)])
        resource = owner(results, limits=HTTPLimits(responseBodyBytes=3))
        outcome, response = await resource.request("http://target.example/start")
        assert outcome.kind == "completed"
        assert response.body == b"sec" and response.truncated
        assert results.attempts[0].request.method == "GET"
        assert results.attempts[0].finished
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,attempts",
    [("GET", 2), ("HEAD", 2), ("OPTIONS", 2), ("POST", 1), ("PUT", 1), ("PATCH", 1), ("DELETE", 1)],
)
def test_only_safe_methods_retry_a_connection_failure_before_write(method, attempts):
    async def scenario():
        results = Results([BridgeFailure("connect_failed")])
        resource = owner(results)
        outcome, _ = await resource.request("http://target.example/start", method=method)
        assert outcome.kind == "failed"
        assert len(results.attempts) == attempts
        assert resource._request_count == attempts
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_domain_does_not_retry_given_possibly_sent_failure(method):
    async def scenario():
        results = Results([BridgeFailure("transport_failed")])
        resource = owner(results)
        outcome, _ = await resource.request("http://target.example/start", method=method)
        assert outcome.kind == "failed"
        assert len(results.attempts) == resource._request_count == 1
        assert results.attempts[0].finished
        await resource.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_http_status_does_not_trigger_automatic_retry(status):
    async def scenario():
        resource = owner(Results([reply(status)]))
        outcome, response = await resource.request("http://target.example/start")
        assert outcome.kind == "completed" and response.status == status
        assert resource._request_count == 1
        await resource.close()

    asyncio.run(scenario())


def test_rate_resets_with_each_new_owner_without_total_quota():
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
        limits = HTTPLimits(requestsPerSecond=2)
        for _ in range(2):
            resource = owner(Results(), limits=limits, monotonic=clock, sleep=clock.sleep)
            for _ in range(45):
                assert (await resource.request("http://target.example/start"))[
                    0
                ].kind == "completed"
            assert resource._request_count == 45
            await resource.close()
        assert clock.waits == [0.5] * 88

    asyncio.run(scenario())


def test_oversized_request_is_rejected_before_connecting():
    async def scenario():
        results = Results()
        resource = owner(results, limits=HTTPLimits(requestBodyBytes=3))
        with pytest.raises(ValueError):
            await resource.request("http://target.example/start", method="POST", body=b"1234")
        assert resource._request_count == 0 and not results.attempts
        await resource.close()

    asyncio.run(scenario())


def test_omitted_http_limits_remain_unconfigured():
    async def scenario():
        resource = owner(Results())
        assert resource._http_limits.model_dump(exclude_none=True) == {}
        for _ in range(3):
            assert (await resource.request("http://target.example/start"))[0].kind == "completed"
        assert resource._next_request_at == 0.0
        await resource.close()

    asyncio.run(scenario())
