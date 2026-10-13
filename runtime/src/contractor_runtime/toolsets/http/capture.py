"""Private, bounded outgoing HTTP capture; never serialized into model state."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from contractor_runtime.toolsets.common.credentials import RUNTIME_TARGET_CREDENTIAL


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
    def from_request(
        cls, request: httpx.Request, *, target_credential: bool = False
    ) -> CapturedAttempt:
        """Capture the request as sent.

        With ``target_credential`` the Authorization header is the project
        target credential Runtime injected. The model never sees it and finding
        evidence is model-readable, so a marker is retained instead.
        """

        headers = capture_headers(request.headers)
        if target_credential:
            for row in headers:
                if row["name"].lower() == "authorization":
                    row["value"] = RUNTIME_TARGET_CREDENTIAL
        return cls(request.method, str(request.url), headers, request.content)

    def header_pairs(self) -> list[tuple[str, str]]:
        return [(row["name"], row["value"]) for row in self.headers]

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
    attempts: list[CapturedAttempt] = field(default_factory=list, repr=False)
    complete: bool = False
    response_body: bytes = field(default=b"", repr=False)
    error: str | None = None

    def evidence(self) -> dict[str, Any]:
        attempts = []
        for index, attempt in enumerate(self.attempts):
            last = index == len(self.attempts) - 1
            value: dict[str, Any] = {
                "request": {
                    "method": attempt.method,
                    "url": attempt.url,
                    "message": {
                        "kind": "structured",
                        "headers": [dict(header) for header in attempt.headers],
                        "body_base64": base64.b64encode(attempt.body).decode("ascii"),
                    },
                }
            }
            if attempt.status is not None and not (last and self.error == "response_invalid"):
                value["response"] = {
                    "status": attempt.status,
                    "message": {
                        "kind": "structured",
                        "headers": [dict(header) for header in attempt.response_headers],
                        "body_base64": base64.b64encode(self.response_body if last else b"").decode(
                            "ascii"
                        ),
                    },
                    "complete": last and self.complete,
                }
            error = self.error if last and self.error else attempt.error
            if error is not None:
                value["error"] = error
            if "response" not in value and "error" not in value:
                value["error"] = "transport_error"
            attempts.append(value)
        return {
            "source": "http",
            "request_id": str(self.request_id),
            "request_tag": self.request_tag,
            "attempts": attempts,
        }
