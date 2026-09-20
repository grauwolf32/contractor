"""Private, bounded outgoing HTTP capture; never serialized into model state."""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.http.limits import MAX_HISTORY


def capture_headers(headers: httpx.Headers) -> list[dict[str, str]]:
    """Preserve duplicates and header bytes using the HTTP Latin-1 mapping."""
    return [
        {"name": name.decode("ascii"), "value": value.decode("latin-1")}
        for name, value in headers.raw
    ]


@dataclass(slots=True)
class CapturedAttempt:
    method: str
    url: str
    headers: list[dict[str, str]] = field(repr=False)
    body: bytes = field(repr=False)
    status: int | None = None
    response_headers: list[dict[str, str]] = field(default_factory=list, repr=False)
    error: Literal["transport_error", "cancelled"] | None = None

    @classmethod
    def from_request(cls, request: httpx.Request) -> CapturedAttempt:
        return cls(
            request.method, str(request.url), capture_headers(request.headers), request.content
        )

    def receive(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.response_headers = capture_headers(response.headers)

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "url": self.url,
            "headers": [dict(header) for header in self.headers],
            "body_base64": base64.b64encode(self.body).decode("ascii"),
        }
        if self.status is not None:
            result["status"] = self.status
            if self.response_headers:
                result["response_headers"] = [dict(header) for header in self.response_headers]
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(slots=True)
class CapturedExchange:
    request_id: int
    request_tag: str
    invocation_id: str | None
    attempts: list[CapturedAttempt] = field(default_factory=list, repr=False)
    complete: bool = False
    response_body: ArtifactRef | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_tag": self.request_tag,
            "attempts": [attempt.snapshot() for attempt in self.attempts],
        }


class HTTPExchangeHistory:
    """Access is serialized by the owning HTTP session's lock."""

    def __init__(self) -> None:
        self._entries: OrderedDict[int, CapturedExchange] = OrderedDict()

    def begin(self, request_id: int, tag: str, invocation_id: str | None) -> CapturedExchange:
        exchange = CapturedExchange(request_id, tag, invocation_id)
        self._entries[request_id] = exchange
        while len(self._entries) > MAX_HISTORY:
            self._entries.popitem(last=False)
        return exchange

    def resolve(self, request_id: int, invocation_id: str) -> CapturedExchange:
        if type(request_id) is not int:
            raise ValueError("request_id must be an integer returned by http_request")
        exchange = self._entries.get(request_id)
        if exchange is None or not exchange.complete or exchange.invocation_id != invocation_id:
            raise ValueError(
                "request_id is unavailable in this invocation; omit it or use a recent ID"
            )
        return exchange

    def clear(self) -> None:
        self._entries.clear()

    def contains(self, request_id: int) -> bool:
        return request_id in self._entries
