"""Bounded allocation-owned HTTP evidence shared by transport and finding tools."""

from __future__ import annotations

import base64
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from contractor_runtime import threads
from contractor_runtime.core.http_limits import (
    MAX_EXCHANGE_ATTEMPTS,
    MAX_HEADER_BYTES,
    MAX_HISTORY,
    MAX_REQUEST_BODY_BYTES,
    MAX_RESPONSE_BODY_BYTES,
    header_block_bytes,
)
from contractor_runtime.toolsets.common.finding_locations import ClosedModel, WebLocation
from contractor_runtime.toolsets.common.http_evidence import HTTPHeader

HTTP_EXCHANGE_MEDIA_TYPE = "application/vnd.contractor.http-exchange+json"
HTTP_EXCHANGE_SCHEMA = "contractor.http.exchange.v1"
MAX_CAPTURE_BUFFER_BYTES = 64 * 1024 * 1024


def decode_bytes(value: str, maximum: int) -> bytes:
    if len(value) > 4 * ((maximum + 2) // 3):
        raise ValueError("HTTP evidence body exceeds its byte limit")
    data = base64.b64decode(value, validate=True)
    if len(data) > maximum or base64.b64encode(data).decode("ascii") != value:
        raise ValueError("invalid HTTP evidence body encoding")
    return data


class StructuredHTTPMessage(ClosedModel):
    kind: Literal["structured"]
    headers: list[HTTPHeader]
    body_base64: str

    @field_validator("headers")
    @classmethod
    def bounded_headers(cls, value: list[HTTPHeader]) -> list[HTTPHeader]:
        if header_block_bytes((row.name, row.value) for row in value) > MAX_HEADER_BYTES:
            raise ValueError("HTTP evidence headers exceed their byte limit")
        return value


class RawHTTPMessage(ClosedModel):
    kind: Literal["raw"]
    data_base64: str


HTTPMessage = Annotated[StructuredHTTPMessage | RawHTTPMessage, Field(discriminator="kind")]


class CapturedHTTPRequest(WebLocation):
    method: str
    message: HTTPMessage

    @model_validator(mode="after")
    def bounded_message(self) -> CapturedHTTPRequest:
        value = self.message
        if isinstance(value, RawHTTPMessage) and not value.data_base64:
            raise ValueError("a captured raw request cannot be empty")
        decode_bytes(
            value.body_base64 if isinstance(value, StructuredHTTPMessage) else value.data_base64,
            MAX_REQUEST_BODY_BYTES,
        )
        return self


class CapturedHTTPResponse(ClosedModel):
    status: int = Field(ge=100, le=599)
    message: HTTPMessage
    complete: bool

    @model_validator(mode="after")
    def bounded_message(self) -> CapturedHTTPResponse:
        value = self.message
        decode_bytes(
            value.body_base64 if isinstance(value, StructuredHTTPMessage) else value.data_base64,
            MAX_RESPONSE_BODY_BYTES,
        )
        return self


class CapturedHTTPAttempt(ClosedModel):
    request: CapturedHTTPRequest
    response: CapturedHTTPResponse | None = None
    error: (
        Literal[
            "transport_error",
            "timeout",
            "cancelled",
            "response_invalid",
            "response_too_large",
            "response_unavailable",
            "target_denied",
            "request_invalid",
        ]
        | None
    ) = None

    @field_validator("response", "error", mode="before")
    @classmethod
    def omit_unknown(cls, value):
        if value is None:
            raise ValueError("omit an absent HTTP outcome")
        return value

    @model_validator(mode="after")
    def outcome(self) -> CapturedHTTPAttempt:
        if self.response is None and self.error is None:
            raise ValueError("HTTP evidence requires a response or explicit failure")
        if self.response is not None and self.response.complete and self.error is not None:
            raise ValueError("a complete response cannot also have an error")
        return self


class CapturedHTTPExchange(ClosedModel):
    source: Literal["http", "caido"]
    request_id: str = Field(min_length=1, max_length=256)
    request_tag: str | None = Field(default=None, min_length=1, max_length=160)
    attempts: list[CapturedHTTPAttempt] = Field(min_length=1, max_length=MAX_EXCHANGE_ATTEMPTS)

    @field_validator("request_tag", mode="before")
    @classmethod
    def omit_unknown_tag(cls, value):
        if value is None:
            raise ValueError("omit an absent request tag")
        return value

    @model_validator(mode="after")
    def source_identity(self) -> CapturedHTTPExchange:
        if len(self.request_id.encode()) > 256 or any(ord(char) < 32 for char in self.request_id):
            raise ValueError("invalid HTTP evidence identity")
        if self.request_tag is not None and (
            not self.request_tag.isascii()
            or any(not (char.isalnum() or char in "._:-") for char in self.request_tag)
            or not self.request_tag[0].isalnum()
        ):
            raise ValueError("invalid HTTP request tag")
        expected_kind = "structured" if self.source == "http" else "raw"
        for attempt in self.attempts:
            if attempt.request.message.kind != expected_kind or (
                attempt.response is not None and attempt.response.message.kind != expected_kind
            ):
                raise ValueError("HTTP evidence message does not match its source")
        if self.source == "http" and (
            not self.request_id.isascii()
            or not self.request_id.isdecimal()
            or str(int(self.request_id)) != self.request_id
            or not 1 <= int(self.request_id) <= 9007199254740991
            or self.request_tag is None
        ):
            raise ValueError("invalid Runtime HTTP request identity")
        return self


@dataclass(frozen=True, slots=True)
class HTTPObservation:
    """Private capture metadata; only the transport supplies observation facts."""

    observed_at: str
    deployment_marker: str | None = None

    def __post_init__(self):
        if not isinstance(self.observed_at, str) or len(self.observed_at) > 32:
            raise ValueError("invalid HTTP observation time")
        parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        shape = re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{0,8}[1-9])?Z",
            self.observed_at,
        )
        if parsed.tzinfo is None or shape is None:
            raise ValueError("HTTP observation requires a UTC timestamp")
        if self.deployment_marker is not None and (
            not self.deployment_marker
            or len(self.deployment_marker.encode()) > 256
            or any(ord(char) < 32 for char in self.deployment_marker)
        ):
            raise ValueError("invalid HTTP deployment marker")

    @property
    def size_bytes(self) -> int:
        return len(self.observed_at.encode()) + len((self.deployment_marker or "").encode())


class RecentHTTPCaptures:
    """Only canonical JSON bytes survive a call; object graphs are not retained.

    Access and mutation happen on the allocation event loop. Encoding and
    parsing run in joined threads. The encoded payload budget and record count
    are independent; replacing or evicting a capture releases its bytes.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, str], tuple[str | None, bytes]] = OrderedDict()
        self._bytes = 0
        self._observations: dict[tuple[str, str], HTTPObservation] = {}

    async def put(
        self,
        capture: dict[str, Any],
        invocation_id: str | None,
        *,
        observation: HTTPObservation | None = None,
    ) -> None:
        def encode() -> tuple[tuple[str, str], bytes]:
            value = CapturedHTTPExchange.model_validate(capture)
            return (value.source, value.request_id), value.model_dump_json(
                exclude_none=True
            ).encode()

        key, encoded = await threads.to_thread_until_done(encode, name="http-capture-encode")
        metadata_bytes = observation.size_bytes if observation is not None else 0
        if len(encoded) + metadata_bytes > MAX_CAPTURE_BUFFER_BYTES:
            raise ValueError("HTTP evidence exceeds the allocation capture budget")
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= len(previous[1])
        self._entries[key] = (invocation_id, encoded)
        old_observation = self._observations.pop(key, None)
        if old_observation is not None:
            self._bytes -= old_observation.size_bytes
        if observation is not None:
            self._observations[key] = observation
        self._bytes += len(encoded) + metadata_bytes
        while len(self._entries) > MAX_HISTORY or self._bytes > MAX_CAPTURE_BUFFER_BYTES:
            expired_key, (_, expired) = self._entries.popitem(last=False)
            metadata = self._observations.pop(expired_key, None)
            self._bytes -= len(expired) + (metadata.size_bytes if metadata is not None else 0)

    async def get(
        self, source: str, request_id: str, *, invocation_id: str | None = None
    ) -> dict[str, Any]:
        entry = self._entries.get((source, request_id))
        if entry is None or (invocation_id is not None and entry[0] != invocation_id):
            raise ValueError("HTTP evidence is unavailable in this invocation")
        return await threads.to_thread_until_done(
            lambda: CapturedHTTPExchange.model_validate_json(entry[1]).model_dump(
                exclude_none=True
            ),
            name="http-capture-read",
        )

    def select_observation(
        self,
        request_id: str,
        invocation_id: str,
    ) -> tuple[bytes, HTTPObservation]:
        """Select an exact Runtime HTTP record without parsing/copying its body.

        Immutable bytes are shared; the collector charges them against its own
        prospective package bound. Caido history has no trusted observation time.
        """
        key = ("http", request_id)
        entry, observation = self._entries.get(key), self._observations.get(key)
        if entry is None or entry[0] != invocation_id or observation is None:
            raise ValueError("HTTP observation is unavailable in this invocation")
        return entry[1], observation

    def contains(self, source: str, request_id: str) -> bool:
        return (source, request_id) in self._entries

    def clear(self, source: str) -> None:
        for key in tuple(self._entries):
            if key[0] == source:
                _, data = self._entries.pop(key)
                metadata = self._observations.pop(key, None)
                self._bytes -= len(data) + (metadata.size_bytes if metadata is not None else 0)


def allocation_http_captures(state: Any) -> RecentHTTPCaptures:
    value = getattr(state, "http_captures", None)
    if value is None:
        value = RecentHTTPCaptures()
        state.http_captures = value
    if not isinstance(value, RecentHTTPCaptures):
        raise TypeError("allocation HTTP capture buffer is invalid")
    return value
