"""Predetermined domain outcomes; no sockets, HTTP parser or wire encoder."""

import asyncio
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

from contractor_runtime.adapters.bridge import BridgeFailure
from contractor_runtime.adapters.http_session import PrivateRequest, PrivateResponse
from fakes.http_transport import HTTPTransport


def reply(status=200, headers=(), body=b"ok", *, truncated=False):
    return PrivateResponse(
        "",
        status,
        [
            (k.encode() if isinstance(k, str) else k, v.encode() if isinstance(v, str) else v)
            for k, v in headers
        ],
        body,
        truncated,
    )


@dataclass
class Attempt:
    url: str
    request: PrivateRequest
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False


class Results:
    def __init__(self, responses=None, *, hold=False):
        self.responses = responses or [reply()]
        self.hold = hold
        self.attempts: list[Attempt] = []

    async def request(self, url, boundary, request, policy, identity, body_limit):
        parsed = urlsplit(url)
        await policy.resolve(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
        boundary.require_time()
        result = self.responses[min(len(self.attempts), len(self.responses) - 1)]
        attempt = Attempt(url, replace(request, headers=list(request.headers)))
        self.attempts.append(attempt)
        try:
            if not isinstance(result, BridgeFailure) or result.reason not in {
                "connect_failed",
                "tls_failed",
            }:
                boundary.request_started = True
            attempt.entered.set()
            if self.hold:
                await asyncio.Event().wait()
            if isinstance(result, BridgeFailure):
                raise BridgeFailure(result.reason)
            return replace(result, url=url, headers=list(result.headers))
        finally:
            attempt.finished = True

    async def close(self):
        if any(not attempt.finished for attempt in self.attempts):
            raise RuntimeError("test owner did not join its requests")


def transport(spec, *, engine=None, **options):
    resource = HTTPTransport(spec, **options)
    if engine is not None:
        resource._engine = engine
    return resource
