"""Allocation-bound client for the Control Plane private Artifact API.

The module also holds what every client of that API shares: its transport,
error model and response checks. Finding proposals
(:mod:`contractor_runtime.finding_proposals`) and model recovery
(:mod:`contractor_runtime.llm.recovery`) have clients of their own.
"""

from __future__ import annotations

import asyncio
import json
import re
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote, urlsplit

from pydantic import Field, ValidationError

from contractor_runtime import _https
from contractor_runtime.contracts import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    ArtifactListResult,
    ArtifactReadResult,
    ArtifactRef,
    ArtifactWriteResult,
    validate_artifact_name,
    validate_media_type,
)
from contractor_runtime.contracts.base import is_blank, trim_whitespace
from contractor_runtime.mtls import verify_control_plane_peer

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 1 << 20
# artifacts.MaxBindingListLimit (LIM-ART-3), pinned by
# api/testdata/v1alpha1/artifact-list-cases.json.
MAX_BINDING_LIST_LIMIT = 256
PATH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
RUNTIME_INSTANCE_HEADER = "X-Contractor-Runtime-Instance-ID"
BINDING_CREATED_AT_HEADER = "x-contractor-binding-created-at"
REVISION_CREATED_AT_HEADER = "x-contractor-revision-created-at"
RFC3339_UTC_PATTERN = re.compile(
    r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?Z$"
)


class ArtifactClientError(Exception):
    """Bounded error safe to record in allocation metrics."""


class ArtifactTransportError(ArtifactClientError):
    pass


class ArtifactResponseLimitError(ArtifactTransportError):
    """The response exceeded the caller's byte limit."""


class ArtifactAPIError(ArtifactClientError):
    def __init__(
        self, status_code: int, code: str, retryable: bool, request_id: str | None = None
    ) -> None:
        super().__init__(f"Artifact API returned HTTP {status_code} ({code})")
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class ArtifactHTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


class ArtifactTransport(Protocol):
    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse: ...


@dataclass(frozen=True, slots=True)
class ArtifactValue:
    artifact: ArtifactRef
    media_type: str
    data: bytes = field(repr=False)
    binding_created_at: datetime
    revision_created_at: datetime


class ArtifactWriteValue(ArtifactWriteResult):
    """Trusted response metadata plus the unchanged public wire projection."""

    binding_created_at: datetime = Field(exclude=True)
    revision_created_at: datetime = Field(exclude=True)


def allocation_path(allocation_id: str, resource: str) -> str:
    """Return the private API path of one resource of an allocation."""

    if PATH_ID_PATTERN.fullmatch(allocation_id) is None:
        raise ValueError("allocation ID is not a safe URL path segment")
    return f"/allocations/{quote(allocation_id, safe='')}/{resource}"


class ArtifactClient:
    """Exposes only the RunScope implied by one active allocation ID."""

    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        self._root = allocation_path(allocation_id, "artifacts")
        self._allocation_id = allocation_id
        self._transport = transport
        self._known_exact_refs: dict[tuple[str, str], ArtifactRef] = {}
        self._observed_exact_refs: list[ArtifactRef] = []

    @property
    def allocation_id(self) -> str:
        """Return the immutable allocation scope used by every request."""
        return self._allocation_id

    @property
    def transport(self) -> ArtifactTransport:
        """Return the allocation transport, which the other private API clients share."""
        return self._transport

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known_exact_refs.values())

    @property
    def observation_cursor(self) -> int:
        """Return an allocation-local cursor for invocation-scoped provenance."""

        return len(self._observed_exact_refs)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        if type(cursor) is not int or cursor < 0 or cursor > len(self._observed_exact_refs):
            raise ValueError("artifact observation cursor is invalid")
        return tuple(self._observed_exact_refs[cursor:])

    def clear_observations(self) -> None:
        """Discard a completed invocation journal without losing latest refs."""

        self._observed_exact_refs.clear()

    def observe_cached_read(self, ref: ArtifactRef) -> None:
        """Record an exact revision served from an allocation-owned read cache."""

        self._remember(ref.require_exact())

    async def list_artifact_page(
        self,
        namespace: str | None = None,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> ArtifactListResult:
        """Return one page of model-visible versionless refs (ART-R47).

        The Server excludes Run system and model-hidden bindings before the
        limit, so every page except the last holds exactly ``limit`` refs.
        Pass the returned ``next_cursor`` with the same namespace for the next
        page; it is ``None`` on the last page.
        """

        query: list[tuple[str, str]] = []
        if namespace is not None:
            validate_artifact_name("namespace", namespace)
            query.append(("namespace", namespace))
        _validate_list_limit(limit)
        query.append(("limit", str(limit)))
        if cursor is not None:
            if type(cursor) is not str or ARTIFACT_LIST_CURSOR_PATTERN.fullmatch(cursor) is None:
                raise ValueError("artifact list cursor is invalid")
            query.append(("cursor", cursor))
        result = await self._list(query)
        if len(result.artifacts) > limit or any(
            namespace is not None and ref.namespace != namespace for ref in result.artifacts
        ):
            raise ArtifactTransportError("Artifact API returned an invalid list page")
        if result.next_cursor is not None and len(result.artifacts) != limit:
            raise ArtifactTransportError("Artifact API returned a short list page with a cursor")
        return result

    async def list_artifact_prefix(
        self, namespace: str, name_prefix: str, *, limit: int
    ) -> list[ArtifactRef]:
        """Return at most ``limit`` refs of one namespace whose names start
        with ``name_prefix``, for trusted adapters such as Memory. Callers
        checking a quota request one extra ref to detect overflow."""

        validate_artifact_name("namespace", namespace)
        validate_artifact_name("name prefix", name_prefix)
        _validate_list_limit(limit)
        result = await self._list(
            [("namespace", namespace), ("namePrefix", name_prefix), ("limit", str(limit))]
        )
        if (
            result.next_cursor is not None
            or len(result.artifacts) > limit
            or any(
                ref.namespace != namespace or not ref.name.startswith(name_prefix)
                for ref in result.artifacts
            )
        ):
            raise ArtifactTransportError("Artifact API returned an invalid filtered list")
        return result.artifacts

    async def _list(self, query: list[tuple[str, str]]) -> ArtifactListResult:
        path = self._root + "?" + "&".join(f"{key}={quote(value, safe='')}" for key, value in query)
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "application/json"},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        raise_for_status(response)
        require_json(response.headers)
        try:
            return ArtifactListResult.model_validate_json(response.body)
        except ValidationError as error:
            raise ArtifactTransportError(
                "Artifact API returned an invalid list response"
            ) from error

    async def read_artifact_metadata(self, ref: ArtifactRef) -> ArtifactReadResult:
        """Return the exact revision, media type and size ``read_artifact`` of
        ``ref`` would return, without downloading the payload (ART-R116).

        The result is not a content observation: only a payload read records
        the exact revision for invocation provenance.
        """

        path = f"{self._root}/{quote(ref.namespace, safe='')}/{quote(ref.name, safe='')}/metadata"
        if ref.revision is not None:
            path += f"?revision={quote(ref.revision, safe='')}"
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "application/json"},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        raise_for_status(response)
        require_json(response.headers)
        try:
            result = ArtifactReadResult.model_validate_json(response.body)
        except ValidationError as error:
            raise ArtifactTransportError(
                "Artifact API returned invalid artifact metadata"
            ) from error
        exact = result.artifact
        if (
            exact.namespace != ref.namespace
            or exact.name != ref.name
            or (ref.revision is not None and exact.revision != ref.revision)
            or not 0 <= result.size <= MAX_ARTIFACT_BYTES
        ):
            raise ArtifactTransportError("Artifact API metadata does not match the requested ref")
        return result

    async def read_artifact(
        self, ref: ArtifactRef, *, max_bytes: int = MAX_ARTIFACT_BYTES
    ) -> ArtifactValue:
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_ARTIFACT_BYTES:
            raise ValueError("artifact read byte limit is invalid")
        path = self._ref_path(ref)
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "*/*"},
            body=b"",
            max_response_bytes=max_bytes,
        )
        raise_for_status(response)
        if len(response.body) > max_bytes:
            raise ArtifactResponseLimitError("Artifact API response exceeds the read byte limit")
        media_type = _response_media_type(response.headers)
        revision = _strong_etag(response.headers)
        if ref.revision is not None and revision != ref.revision:
            raise ArtifactTransportError(
                "Artifact API exact-read ETag does not match the requested revision"
            )
        binding_created_at, revision_created_at = _artifact_timestamps(response.headers)
        exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision=revision)
        try:
            ArtifactReadResult(
                apiVersion="contractor/v1alpha1",
                artifact=exact,
                mediaType=media_type,
                size=len(response.body),
            )
        except ValidationError as error:
            raise ArtifactTransportError(
                "Artifact API returned invalid artifact metadata"
            ) from error
        self._remember(exact)
        return ArtifactValue(
            artifact=exact,
            media_type=media_type,
            data=response.body,
            binding_created_at=binding_created_at,
            revision_created_at=revision_created_at,
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteValue:
        if target.revision is not None:
            raise ValueError("artifact write target must be versionless")
        if not isinstance(data, bytes):
            raise TypeError("artifact payload must be bytes")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact payload exceeds the 64 MiB limit")
        validate_media_type(media_type)
        headers = {"Accept": "application/json", "Content-Type": media_type}
        expected_status = 201
        if expected_revision is None:
            headers["If-None-Match"] = "*"
        else:
            if is_blank(expected_revision) or any(char in expected_revision for char in '\r\n",'):
                raise ValueError("expected revision is not a valid strong ETag value")
            headers["If-Match"] = json.dumps(expected_revision)
            expected_status = 200
        response = await self._transport.request(
            "PUT",
            self._ref_path(target),
            headers=headers,
            body=data,
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        raise_for_status(response)
        if response.status_code != expected_status:
            raise ArtifactTransportError("Artifact API returned an invalid write status")
        require_json(response.headers)
        try:
            result = ArtifactWriteResult.model_validate_json(response.body)
        except ValidationError as error:
            raise ArtifactTransportError(
                "Artifact API returned an invalid write response"
            ) from error
        revision = _strong_etag(response.headers)
        binding_created_at, revision_created_at = _artifact_timestamps(response.headers)
        if (
            result.artifact.namespace != target.namespace
            or result.artifact.name != target.name
            or result.artifact.revision != revision
            or result.media_type != media_type
            or result.size != len(data)
        ):
            raise ArtifactTransportError("Artifact API write response does not match the request")
        self._remember(result.artifact)
        return ArtifactWriteValue(
            apiVersion="contractor/v1alpha1",
            artifact=result.artifact,
            mediaType=result.media_type,
            size=result.size,
            binding_created_at=binding_created_at,
            revision_created_at=revision_created_at,
        )

    def _remember(self, ref: ArtifactRef) -> None:
        revision = ref.require_exact().revision
        if not (revision is not None):
            raise RuntimeError("Expected revision is not None")
        self._known_exact_refs[(ref.namespace, ref.name)] = ref
        self._observed_exact_refs.append(ref)

    def _ref_path(self, ref: ArtifactRef) -> str:
        path = f"{self._root}/{quote(ref.namespace, safe='')}/{quote(ref.name, safe='')}"
        if ref.revision is not None:
            path += f"?revision={quote(ref.revision, safe='')}"
        return path


def raise_for_status(response: ArtifactHTTPResponse) -> None:
    """Raise the private API error a non-2xx response carries."""

    if 200 <= response.status_code < 300:
        return
    error = _https.decode_private_error(response.body)
    if error is None:
        raise ArtifactAPIError(response.status_code, "invalid_error_response", False)
    raise ArtifactAPIError(response.status_code, error.code, error.retryable, error.request_id)


class MTLSArtifactTransport:
    """Minimal HTTPS/1.1 transport with pre-request Control Plane role check."""

    def __init__(
        self,
        base_url: str,
        context: ssl.SSLContext,
        timeout_seconds: float,
        runtime_instance_id: str,
    ) -> None:
        parsed = urlsplit(base_url)
        base_path = parsed.path.rstrip("/")
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not base_path.endswith("/private/v1")
        ):
            raise ValueError(
                "Artifact API URL must be an HTTPS private/v1 root without credentials"
            )
        if timeout_seconds <= 0:
            raise ValueError("Artifact API timeout must be positive")
        if (
            is_blank(runtime_instance_id)
            or runtime_instance_id != trim_whitespace(runtime_instance_id)
            or len(runtime_instance_id) > 256
        ):
            raise ValueError("Runtime Agent instance ID is invalid")
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._base_path = base_path
        self._context = context
        self._timeout = timeout_seconds
        self._runtime_instance_id = runtime_instance_id

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        if method not in {"GET", "PUT", "POST"} or not path.startswith("/") or "#" in path:
            raise ValueError("invalid private Artifact API request target")
        if not 0 <= max_response_bytes <= MAX_ARTIFACT_BYTES:
            raise ValueError("invalid Artifact API response limit")
        try:
            request_headers = {
                "Host": _https.host_header(self._host, self._port),
                "Content-Length": str(len(body)),
                "Connection": "close",
                **headers,
                "X-Request-ID": _https.new_request_id(),
                RUNTIME_INSTANCE_HEADER: self._runtime_instance_id,
            }
            request = _https.encode_request(method, self._base_path + path, request_headers, body)
            response = await _https.exchange(
                self._host,
                self._port,
                self._context,
                request,
                timeout_seconds=self._timeout,
                max_response_bytes=max_response_bytes,
                verify_peer=verify_control_plane_peer,
            )
        except _https.HTTPResponseTooLargeError:
            raise ArtifactResponseLimitError(
                "Artifact API response exceeds its size limit"
            ) from None
        except _https.HTTPResponseError as error:
            raise ArtifactTransportError(f"invalid Artifact API response ({error})") from None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ArtifactTransportError(
                f"Artifact API transport failed ({type(error).__name__})"
            ) from None
        return ArtifactHTTPResponse(response.status_code, response.headers, response.body)


def _validate_list_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= MAX_BINDING_LIST_LIMIT:
        raise ValueError("artifact list limit is invalid")


def _response_media_type(headers: Mapping[str, str]) -> str:
    return validate_media_type(headers.get("content-type", ""))


def require_json(headers: Mapping[str, str]) -> None:
    """Refuse a private API response that is not JSON."""

    if headers.get("content-type") != "application/json":
        raise ArtifactTransportError("Artifact API response is not application/json")


def _strong_etag(headers: Mapping[str, str]) -> str:
    value = headers.get("etag", "")
    if not value or value.startswith("W/") or "," in value:
        raise ArtifactTransportError("Artifact API response has no strong revision ETag")
    try:
        revision = json.loads(value)
    except json.JSONDecodeError as error:
        raise ArtifactTransportError(
            "Artifact API response has an invalid revision ETag"
        ) from error
    if not isinstance(revision, str) or not revision:
        raise ArtifactTransportError("Artifact API response has an invalid revision ETag")
    return revision


def _artifact_timestamps(headers: Mapping[str, str]) -> tuple[datetime, datetime]:
    return (
        _artifact_timestamp(headers, BINDING_CREATED_AT_HEADER),
        _artifact_timestamp(headers, REVISION_CREATED_AT_HEADER),
    )


def _artifact_timestamp(headers: Mapping[str, str], name: str) -> datetime:
    value = headers.get(name, "")
    match = RFC3339_UTC_PATTERN.fullmatch(value)
    if match is None:
        raise ArtifactTransportError("Artifact API response has invalid timestamp metadata")
    fraction = match.group("fraction") or ""
    # PostgreSQL stores microseconds. Accept the full RFC3339Nano syntax so a
    # future repository may emit nine digits; datetime intentionally projects
    # it at Python's microsecond precision.
    normalized_fraction = (fraction + "000000")[:6]
    suffix = f".{normalized_fraction}" if fraction else ""
    try:
        parsed = datetime.fromisoformat(f"{match.group('date')}{suffix}+00:00")
    except ValueError as error:
        raise ArtifactTransportError(
            "Artifact API response has invalid timestamp metadata"
        ) from error
    if parsed.tzinfo != UTC:
        raise ArtifactTransportError("Artifact API response has invalid timestamp metadata")
    return parsed
