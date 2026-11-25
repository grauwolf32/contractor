"""Process fixture: add one verified target CA, preserving the production adapter."""

import contextvars
import ipaddress
import json
import os
import ssl
from dataclasses import replace

from contractor_runtime import cli
from contractor_runtime.adapters.http_transport import (
    HTTPTransport,
    HTTPTransportFactory,
)
from contractor_runtime.core.target_policy import system_resolver

original_factories = cli.built_in_factories
fixture_dns = json.loads(os.environ.get("CONTRACTOR_TEST_TARGET_DNS", "{}"))
fixture_dns_counts = {}


async def fixture_resolver(host, port):
    if host not in fixture_dns:
        return await system_resolver(host, port)
    responses = fixture_dns[host]
    index = fixture_dns_counts.get(host, 0)
    fixture_dns_counts[host] = index + 1
    return tuple(
        ipaddress.ip_address(value)
        for value in responses[min(index, len(responses) - 1)]
    )


class ObservedHTTPTransport(HTTPTransport):
    _invocation = contextvars.ContextVar("http_test_invocation", default=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_index = 0
        self._test_observed = {}

    def _observe(self, outcome, response=None):
        index = self._invocation.get()
        self._test_observed[index] = outcome
        data = {
            "step": index,
            "kind": outcome.kind,
            "reason": outcome.reason,
            "statusCode": response.status if response else 0,
            "responseBytes": len(response.body) if response else 0,
            "truncated": bool(response and response.truncated),
        }
        with open(
            os.environ["CONTRACTOR_TEST_HTTP_OBSERVATIONS"], "a", encoding="utf-8"
        ) as output:
            output.write(json.dumps(data) + "\n")

    async def _request_chain(self, *args, **kwargs):
        index = self._test_index
        self._test_index += 1
        token = self._invocation.set(index)
        try:
            result = await super()._request_chain(*args, **kwargs)
            if index not in self._test_observed:
                self._observe(*result)
            return result
        finally:
            self._invocation.reset(token)

    async def _request(self, *args, **kwargs):
        outcome, next_url, response = await super()._request(*args, **kwargs)
        self._observe(outcome, response)
        return outcome, next_url, response

    def _failure(self, boundary, reason):
        result = super()._failure(boundary, reason)
        self._observe(result)
        return result


class VerifiedFixtureTargetFactory(HTTPTransportFactory):
    def create(self, spec):
        context = ssl.create_default_context(
            cafile=os.environ["CONTRACTOR_TEST_TARGET_CA"]
        )
        return ObservedHTTPTransport(
            spec,
            native_engine=self._native_engine(),
            protected_urls=self._protected_urls,
            resolver=self._resolver,
            ssl_context=context,
        )


def fixture_factories(*args, **kwargs):
    factories = original_factories(*args, **kwargs)
    original = factories.http_transport_factory
    if original is None:
        raise RuntimeError("production engagement target adapter is absent")
    return replace(
        factories,
        http_transport_factory=VerifiedFixtureTargetFactory(
            resolver=fixture_resolver,
            protected_urls=original._protected_urls
            + tuple(json.loads(os.environ.get("CONTRACTOR_TEST_PROTECTED_URLS", "[]"))),
        ),
    )


cli.built_in_factories = fixture_factories
cli.main()
