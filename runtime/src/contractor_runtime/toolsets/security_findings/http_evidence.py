"""Validate retained HTTP evidence independently of transport and tool schemas."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import Field, field_validator, model_validator

from contractor_runtime.toolsets.http.limits import (
    MAX_EXCHANGE_ATTEMPTS,
    MAX_HEADER_BYTES,
    MAX_REQUEST_BODY_BYTES,
    header_block_bytes,
)
from contractor_runtime.toolsets.security_findings.locations import (
    HTTP_TOKEN,
    ClosedModel,
    LineNumber,
    WebLocation,
)


class HTTPHeader(ClosedModel):
    name: str
    value: str

    @model_validator(mode="after")
    def check_header(self) -> HTTPHeader:
        if not HTTP_TOKEN.fullmatch(self.name) or any(char in self.value for char in "\r\n\0"):
            raise ValueError("invalid HTTP header")
        return self


class HTTPAttempt(WebLocation):
    method: str
    headers: list[HTTPHeader]
    body_base64: str
    # RFC 9110 HTTP status range.
    status: int | None = Field(default=None, ge=100, le=599)
    response_headers: list[HTTPHeader] = Field(default_factory=list)
    error: Literal["transport_error", "cancelled"] | None = None

    @field_validator("status", "error", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None:
            raise ValueError("omit an unknown outcome field instead of passing null")
        return value

    @field_validator("body_base64")
    @classmethod
    def check_body(cls, value: str) -> str:
        body = base64.b64decode(value, validate=True)
        if len(body) > MAX_REQUEST_BODY_BYTES or base64.b64encode(body).decode("ascii") != value:
            raise ValueError("invalid or oversized request body encoding")
        return value

    @field_validator("headers", "response_headers")
    @classmethod
    def check_headers(cls, values: list[HTTPHeader]) -> list[HTTPHeader]:
        if header_block_bytes((row.name, row.value) for row in values) > MAX_HEADER_BYTES:
            raise ValueError("HTTP headers exceed the byte limit")
        return values

    @model_validator(mode="after")
    def check_outcome(self) -> HTTPAttempt:
        if (self.status is None) == (self.error is None):
            raise ValueError("an attempt requires either response status or transport error")
        return self


class HTTPExchange(ClosedModel):
    request_id: LineNumber
    request_tag: str
    attempts: list[HTTPAttempt] = Field(min_length=1, max_length=MAX_EXCHANGE_ATTEMPTS)
    response_body_evidence_id: str | None = None

    @field_validator("response_body_evidence_id")
    @classmethod
    def check_evidence_id(cls, value: str | None) -> str:
        if not value:
            raise ValueError("omit an absent response body reference")
        return value
