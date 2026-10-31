"""Allocation-bound client for the Control Plane private Artifact API.

The module also holds what every client of that API shares: its transport,
error model and response checks. Finding proposals
(:mod:`contractor_runtime.finding_proposals`) and model recovery
(:mod:`contractor_runtime.llm.recovery`) have clients of their own.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import quote, urlsplit

from pydantic import Field, ValidationError

from contractor_runtime import _https
from contractor_runtime.contracts import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    ArtifactListResult,
    ArtifactMetadataResult,
    ArtifactReadResult,
    ArtifactRef,
    ArtifactWriteResult,
    PrivateProtocolDecodeError,
    decode_private,
    validate_artifact_name,
    validate_media_type,
)
from contractor_runtime.contracts.base import NAME_PATTERN, is_blank, trim_whitespace
from contractor_runtime.shared_config import artifact_payload_max_bytes

if TYPE_CHECKING:
    from contractor_runtime.archive_cache import ImmutableArchiveCache

MAX_ARTIFACT_JSON_BYTES = 1 << 20
# artifacts.MaxBindingListLimit (LIM-ART-3), pinned by
# api/testdata/v1alpha1/artifact-list-cases.json.
MAX_BINDING_LIST_LIMIT = 256
PATH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# The revision a strong entity tag carries verbatim between its double
# quotes: visible ASCII other than a double quote, a backslash or a comma.
# An entity tag has no escapes, so If-Match and ETag are never JSON-encoded;
# the Server reads the same grammar (API-R70,
# api/testdata/v1alpha1/strong-etag-cases.json).
STRONG_ETAG_REVISION_PATTERN = re.compile(r"[!#-+\--\[\]-~]+")
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


# A private Artifact API answer to a CAS write whose binding's retained
# history needs more maintenance than one Server transaction may do. The write
# stored nothing and the Server committed one maintenance pass before it
# answered, so the same write may be sent again (ART-R127).
ARTIFACT_HISTORY_MAINTENANCE_PENDING = "artifact_history_maintenance_pending"


class ArtifactAPIError(ArtifactClientError):
    def __init__(
        self, status_code: int, code: str, retryable: bool, request_id: str | None = None
    ) -> None:
        super().__init__(f"Artifact API returned HTTP {status_code} ({code})")
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        self.retry_after_seconds: float | None = None


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


@dataclass(frozen=True, slots=True)
class ArtifactReadTarget:
    """An explicitly selected Project operation; its scope ID stays on the Server."""

    scope: Literal["project"]
    operation: str

    def __post_init__(self) -> None:
        if self.scope != "project" or (
            type(self.operation) is not str or NAME_PATTERN.fullmatch(self.operation) is None
        ):
            raise ValueError("artifact read target is invalid")


def allocation_path(allocation_id: str, resource: str) -> str:
    """Return the private API path of one resource of an allocation."""

    if PATH_ID_PATTERN.fullmatch(allocation_id) is None:
        raise ValueError("allocation ID is not a safe URL path segment")
    return f"/allocations/{quote(allocation_id, safe='')}/{resource}"


class ArtifactClient:
    """Read the allocation's Run or an explicitly granted Project binding.

    Listing and writing always use the current Run. Project reads do not enter
    the Run's exact-reference provenance journal.
    """

    def __init__(
        self,
        allocation_id: str,
        transport: ArtifactTransport,
        *,
        archive_cache: ImmutableArchiveCache | None = None,
        archive_trust_domain: str = "",
    ) -> None:
        self.archive_cache = archive_cache
        self.archive_trust_domain = archive_trust_domain
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
        """Record a verified cache hit as this allocation's exact content observation."""

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

    async def read_artifact_metadata(
        self, ref: ArtifactRef, *, target: ArtifactReadTarget | None = None
    ) -> ArtifactMetadataResult:
        """Return the exact revision, media type and size ``read_artifact`` of
        ``ref`` would return, without downloading the payload (ART-R116).

        The result is not a content observation: only a payload read records
        the exact revision for invocation provenance.
        """

        path = self._ref_path(ref, target=target, suffix="/metadata")
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
            result = decode_private(ArtifactMetadataResult, response.body)
        except (PrivateProtocolDecodeError, RecursionError):
            raise ArtifactTransportError(
                "Artifact API returned invalid artifact metadata"
            ) from None
        exact = result.artifact
        if (
            exact.namespace != ref.namespace
            or exact.name != ref.name
            or (ref.revision is not None and exact.revision != ref.revision)
            or not 0 <= result.size <= artifact_payload_max_bytes()
        ):
            raise ArtifactTransportError("Artifact API metadata does not match the requested ref")
        return result

    async def read_artifact(
        self,
        ref: ArtifactRef,
        *,
        max_bytes: int | None = None,
        target: ArtifactReadTarget | None = None,
    ) -> ArtifactValue:
        if max_bytes is None:
            max_bytes = artifact_payload_max_bytes()
        if type(max_bytes) is not int or not 0 < max_bytes <= artifact_payload_max_bytes():
            raise ValueError("artifact read byte limit is invalid")
        path = self._ref_path(ref, target=target)
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
        if target is None:
            self._remember(exact)
        return ArtifactValue(
            artifact=exact,
            media_type=media_type,
            data=response.body,
            binding_created_at=binding_created_at,
            revision_created_at=revision_created_at,
        )

    async def check_artifact_access(self, ref: ArtifactRef, *, target: ArtifactReadTarget) -> None:
        """Recheck current operation authority without resolving content/history."""

        if not isinstance(target, ArtifactReadTarget):
            raise ValueError("artifact read target is invalid")
        if ref.revision is not None:
            raise ValueError("artifact access target must be versionless")
        response = await self._transport.request(
            "GET",
            self._ref_path(ref, target=target, suffix="/access"),
            headers={"Accept": "application/json"},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        raise_for_status(response)
        if response.status_code != 204 or response.body:
            raise ArtifactTransportError("Artifact API returned an invalid access response")

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
        if len(data) > artifact_payload_max_bytes():
            raise ValueError("artifact payload exceeds the configured Artifact limit")
        validate_media_type(media_type)
        headers = {"Accept": "application/json", "Content-Type": media_type}
        expected_status = 201
        if expected_revision is None:
            headers["If-None-Match"] = "*"
        else:
            if STRONG_ETAG_REVISION_PATTERN.fullmatch(expected_revision) is None:
                raise ValueError("expected revision is not a valid strong ETag value")
            headers["If-Match"] = f'"{expected_revision}"'
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

    def _ref_path(
        self,
        ref: ArtifactRef,
        *,
        target: ArtifactReadTarget | None = None,
        suffix: str = "",
    ) -> str:
        path = f"{self._root}/{quote(ref.namespace, safe='')}/{quote(ref.name, safe='')}{suffix}"
        query = []
        if target is not None:
            if not isinstance(target, ArtifactReadTarget):
                raise ValueError("artifact read target is invalid")
            query.extend((("scope", target.scope), ("operation", target.operation)))
        if ref.revision is not None:
            query.append(("revision", ref.revision))
        if query:
            path += "?" + "&".join(f"{key}={quote(value, safe='')}" for key, value in query)
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
    """Minimal HTTPS/1.1 transport with pre-request Control Plane role check.

    Requests reuse the process-owned private connections, which verify the
    Control Plane role on every new connection before its first request byte
    (CTL-R163). Only a GET is ever sent again, and only when a reused
    connection returned no response byte; a write is never repeated.
    """

    def __init__(
        self,
        base_url: str,
        connections: _https.PrivateConnections,
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
        self._connections = connections
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
        # Metadata/list/write replies have their own JSON budget. A smaller
        # payload policy must not make those protocol responses impossible.
        response_ceiling = max(artifact_payload_max_bytes(), MAX_ARTIFACT_JSON_BYTES)
        if type(max_response_bytes) is not int or not 0 <= max_response_bytes <= response_ceiling:
            raise ValueError("invalid Artifact API response limit")
        try:
            request_headers = {
                "Host": _https.host_header(self._host, self._port),
                "Content-Length": str(len(body)),
                **headers,
                "X-Request-ID": _https.new_request_id(),
                RUNTIME_INSTANCE_HEADER: self._runtime_instance_id,
            }
            request = _https.encode_request(method, self._base_path + path, request_headers, body)
            response = await self._connections.exchange(
                self._host,
                self._port,
                request,
                idempotent=method == "GET",
                timeout_seconds=self._timeout,
                max_response_bytes=max_response_bytes,
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
    revision = value[1:-1]
    if (
        len(value) < 3
        or value[0] != '"'
        or value[-1] != '"'
        or STRONG_ETAG_REVISION_PATTERN.fullmatch(revision) is None
    ):
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
