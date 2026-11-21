"""Bounded allocation-owned HTTP evidence shared by transport, results and finding tools."""

from __future__ import annotations

import base64
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from contractor_runtime import threads
from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.core.http_limits import (
    MAX_EXCHANGE_ATTEMPTS,
    MAX_HISTORY,
    header_block_bytes,
)
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.toolsets.common.finding_locations import ClosedModel, WebLocation
from contractor_runtime.toolsets.common.http_evidence import HTTPHeader

if TYPE_CHECKING:
    from contractor_runtime.toolsets.common.resources import ToolResources

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
        if (
            header_block_bytes((row.name, row.value) for row in value)
            > get_shared_config().http_header_block_max_bytes
        ):
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
            get_shared_config().http_request_body_max_bytes,
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
            get_shared_config().http_response_body_max_bytes,
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
    observed_at: str | None = None
    deployment_marker: str | None = None
    attempts: list[CapturedHTTPAttempt] = Field(min_length=1, max_length=MAX_EXCHANGE_ATTEMPTS)

    @field_validator("observed_at", "deployment_marker", mode="before")
    @classmethod
    def observation_field(cls, value):
        if not isinstance(value, str) or not value:
            raise ValueError("omit unavailable observation metadata")
        return value

    @field_validator("request_tag", mode="before")
    @classmethod
    def omit_unknown_tag(cls, value):
        if value is None:
            raise ValueError("omit an absent request tag")
        return value

    @model_validator(mode="after")
    def source_identity(self) -> CapturedHTTPExchange:
        if self.observed_at is not None:
            HTTPObservation(self.observed_at, self.deployment_marker)
        elif self.deployment_marker is not None:
            HTTPObservation("2000-01-01T00:00:00Z", self.deployment_marker)
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
            or not 1 <= int(self.request_id) <= MAX_EXACT_INTEGER
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
        self._next_local_id = 1
        self._locators: OrderedDict[int, tuple[str, str]] = OrderedDict()
        self._local_ids: dict[tuple[str, str], int] = {}
        self._observations: dict[tuple[str, str], HTTPObservation] = {}

    def _issue_id(self) -> int:
        if self._next_local_id > MAX_EXACT_INTEGER:
            raise ValueError("local HTTP request identity exhausted")
        result = self._next_local_id
        self._next_local_id += 1
        return result

    def new_http_request(self) -> int:
        local_id = self._issue_id()
        self._retain_locator(local_id, ("http", str(local_id)))
        return local_id

    def local_id(self, source: str, provider_id: str) -> int:
        if source not in {"http", "caido"} or not isinstance(provider_id, str):
            raise ValueError("invalid HTTP request locator")
        if (
            not provider_id
            or len(provider_id.encode()) > 256
            or any(ord(char) < 32 for char in provider_id)
        ):
            raise ValueError("invalid HTTP request locator")
        key = (source, provider_id)
        existing = self._local_ids.get(key)
        if existing is not None:
            self._locators.move_to_end(existing)
            return existing
        # HTTP's native identity is already allocated here. A request that was
        # in flight during discovery eviction keeps that original identity.
        local_id = int(provider_id) if source == "http" else self._issue_id()
        if source == "http" and (
            str(local_id) != provider_id or not 0 < local_id < self._next_local_id
        ):
            raise ValueError("HTTP request was not allocated locally")
        self._retain_locator(local_id, key)
        return local_id

    def _retain_locator(self, local_id: int, key: tuple[str, str]) -> None:
        self._locators[local_id] = key
        self._local_ids[key] = local_id
        while len(self._locators) > MAX_HISTORY:
            _, expired_key = self._locators.popitem(last=False)
            self._local_ids.pop(expired_key)
            self._remove_capture(expired_key)

    def resolve(self, local_id: int, *, source: str | None = None) -> tuple[str, str]:
        if type(local_id) is not int or not 1 <= local_id <= MAX_EXACT_INTEGER:
            raise ValueError("request_id must be a local positive integer")
        key = self._locators.get(local_id)
        if key is None or (source is not None and key[0] != source):
            raise ValueError("request_id is unavailable for this tool; inspect recent history")
        return key

    def _remove_capture(self, key: tuple[str, str]) -> None:
        previous = self._entries.pop(key, None)
        observation = self._observations.pop(key, None)
        if previous is not None:
            self._bytes -= len(previous[1])
        if observation is not None:
            self._bytes -= observation.size_bytes

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

        if observation is not None:
            capture = {
                **capture,
                "observed_at": observation.observed_at,
                **(
                    {"deployment_marker": observation.deployment_marker}
                    if observation.deployment_marker
                    else {}
                ),
            }
        key, encoded = await threads.to_thread_until_done(encode, name="http-capture-encode")
        metadata_bytes = observation.size_bytes if observation is not None else 0
        if len(encoded) + metadata_bytes > MAX_CAPTURE_BUFFER_BYTES:
            raise ValueError("HTTP evidence exceeds the allocation capture budget")
        self.local_id(*key)
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
        for local_id, key in tuple(self._locators.items()):
            if key[0] == source:
                self._locators.pop(local_id)
                self._local_ids.pop(key)
        for key in tuple(self._entries):
            if key[0] == source:
                _, data = self._entries.pop(key)
                metadata = self._observations.pop(key, None)
                self._bytes -= len(data) + (metadata.size_bytes if metadata is not None else 0)


def allocation_http_captures(resources: ToolResources) -> RecentHTTPCaptures:
    return resources.http_captures


class HTTPExchangeOrigin(ClosedModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
    invocation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class HTTPExchangeArtifact(ClosedModel):
    schema_: Literal["contractor.http.exchange.v1"] = Field(alias="schema")
    origin: HTTPExchangeOrigin
    capture: CapturedHTTPExchange

    @model_validator(mode="after")
    def observed_http(self):
        if self.capture.source == "http" and self.capture.observed_at is None:
            raise ValueError("Runtime HTTP evidence requires its original observation time")
        return self
