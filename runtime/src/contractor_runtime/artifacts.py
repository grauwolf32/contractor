"""Allocation-bound client for the Control Plane private Artifact API."""

from __future__ import annotations

import asyncio
import json
import re
import ssl
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote, urlsplit

from pydantic import Field, ValidationError

from contractor_runtime.contracts import (
    ARTIFACT_NAME_PATTERN,
    ArtifactListResult,
    ArtifactReadResult,
    ArtifactRef,
    ArtifactWriteResult,
)
from contractor_runtime.mtls import verify_control_plane_peer

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 1 << 20
MAX_BINDING_LIST_LIMIT = 256
MAX_RESPONSE_HEADERS = 64
PATH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*$"
)
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


class ArtifactClient:
    """Exposes only the RunScope implied by one active allocation ID."""

    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        if PATH_ID_PATTERN.fullmatch(allocation_id) is None:
            raise ValueError("allocation ID is not a safe URL path segment")
        self._allocation_id = allocation_id
        self._transport = transport
        self._root = f"/allocations/{quote(allocation_id, safe='')}/artifacts"
        self._finding_root = f"/allocations/{quote(allocation_id, safe='')}/finding-proposals"
        self._known_exact_refs: dict[tuple[str, str], ArtifactRef] = {}
        self._observed_exact_refs: list[ArtifactRef] = []

    @property
    def allocation_id(self) -> str:
        """Return the immutable allocation scope used by every request."""
        return self._allocation_id

    async def submit_finding_proposal(self, request: Mapping[str, object]) -> dict[str, object]:
        """Submit one stable allocation-bound finding request.

        A transport loss is retried once with byte-identical content. The
        Server receipt is the idempotency authority; this client never invents
        a second submission identity after an ambiguous commit.
        """

        body = json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(body) > MAX_ARTIFACT_JSON_BYTES:
            raise ValueError("finding proposal request exceeds its size limit")
        response: ArtifactHTTPResponse | None = None
        for attempt in range(2):
            try:
                response = await self._transport.request(
                    "POST",
                    self._finding_root,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
                )
                break
            except ArtifactTransportError:
                if attempt != 0:
                    raise
        assert response is not None
        self._raise_for_status(response)
        if response.status_code not in {200, 201}:
            raise ArtifactTransportError("finding proposal API returned an invalid status")
        _require_json(response.headers)
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactTransportError("finding proposal API returned invalid JSON") from error
        expected = {"apiVersion", "proposalId", "receiptId", "proposal", "replayed"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ArtifactTransportError("finding proposal API returned an invalid receipt")
        if value.get("apiVersion") != "contractor/v1alpha1":
            raise ArtifactTransportError("finding proposal API returned an invalid version")
        for field_name in ("proposalId", "receiptId"):
            candidate = value.get(field_name)
            if not isinstance(candidate, str) or REQUEST_ID_PATTERN.fullmatch(candidate) is None:
                raise ArtifactTransportError("finding proposal API returned an invalid identity")
        if not isinstance(value.get("replayed"), bool):
            raise ArtifactTransportError("finding proposal API returned an invalid replay marker")
        proposal = value.get("proposal")
        if not isinstance(proposal, dict) or set(proposal) != {
            "ref",
            "digest",
            "mediaType",
            "sizeBytes",
        }:
            raise ArtifactTransportError("finding proposal API returned invalid artifact metadata")
        try:
            ref = ArtifactRef.model_validate(proposal.get("ref"))
            ref.require_exact()
        except (ValidationError, ValueError, AttributeError) as error:
            raise ArtifactTransportError(
                "finding proposal API returned invalid artifact metadata"
            ) from error
        digest = proposal.get("digest")
        size = proposal.get("sizeBytes")
        if (
            ref.namespace != "finding-proposals"
            or not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
            or proposal.get("mediaType") != "application/json"
            or type(size) is not int
            or not 0 <= size <= 8 * 1024 * 1024
        ):
            raise ArtifactTransportError("finding proposal API returned invalid artifact metadata")
        return value

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

    async def list_artifacts(
        self,
        namespace: str | None = None,
        *,
        name_prefix: str | None = None,
        limit: int | None = None,
    ) -> list[ArtifactRef]:
        path = self._root
        if namespace is not None:
            _validate_component("namespace", namespace)
            path += f"?namespace={quote(namespace, safe='')}"
        if name_prefix is not None or limit is not None:
            if namespace is None or type(name_prefix) is not str:
                raise ValueError("filtered artifact list requires namespace, name_prefix and limit")
            _validate_component("name prefix", name_prefix)
            if type(limit) is not int or not 1 <= limit <= MAX_BINDING_LIST_LIMIT:
                raise ValueError("artifact list limit is invalid")
            path += f"&namePrefix={quote(name_prefix, safe='')}&limit={limit}"
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "application/json"},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        self._raise_for_status(response)
        _require_json(response.headers)
        try:
            result = ArtifactListResult.model_validate_json(response.body)
        except ValidationError as error:
            raise ArtifactTransportError(
                "Artifact API returned an invalid list response"
            ) from error
        if name_prefix is not None and (
            len(result.artifacts) > limit
            or any(
                ref.namespace != namespace
                or not ref.name.startswith(name_prefix)
                or ref.revision is not None
                for ref in result.artifacts
            )
        ):
            raise ArtifactTransportError("Artifact API returned an invalid filtered list")
        return result.artifacts

    async def read_artifact(
        self, ref: ArtifactRef, *, max_bytes: int = MAX_ARTIFACT_BYTES
    ) -> ArtifactValue:
        _validate_ref(ref)
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
        self._raise_for_status(response)
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
        _validate_ref(target)
        if target.revision is not None:
            raise ValueError("artifact write target must be versionless")
        if not isinstance(data, bytes):
            raise TypeError("artifact payload must be bytes")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact payload exceeds the 64 MiB limit")
        _validate_media_type(media_type)
        headers = {"Accept": "application/json", "Content-Type": media_type}
        expected_status = 201
        if expected_revision is None:
            headers["If-None-Match"] = "*"
        else:
            if not expected_revision.strip() or any(char in expected_revision for char in '\r\n",'):
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
        self._raise_for_status(response)
        if response.status_code != expected_status:
            raise ArtifactTransportError("Artifact API returned an invalid write status")
        _require_json(response.headers)
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
        assert revision is not None
        self._known_exact_refs[(ref.namespace, ref.name)] = ref
        self._observed_exact_refs.append(ref)

    def _ref_path(self, ref: ArtifactRef) -> str:
        path = f"{self._root}/{quote(ref.namespace, safe='')}/{quote(ref.name, safe='')}"
        if ref.revision is not None:
            path += f"?revision={quote(ref.revision, safe='')}"
        return path

    @staticmethod
    def _raise_for_status(response: ArtifactHTTPResponse) -> None:
        if 200 <= response.status_code < 300:
            return
        code = "invalid_error_response"
        retryable = False
        request_id: str | None = None
        if len(response.body) <= MAX_ARTIFACT_JSON_BYTES:
            try:
                value = json.loads(response.body)
                if (
                    isinstance(value, dict)
                    and set(value) == {"code", "message", "retryable", "requestId"}
                    and isinstance(value["code"], str)
                    and value["code"].strip()
                    and isinstance(value["message"], str)
                    and value["message"].strip()
                    and isinstance(value["retryable"], bool)
                    and isinstance(value["requestId"], str)
                    and REQUEST_ID_PATTERN.fullmatch(value["requestId"]) is not None
                ):
                    code = value["code"]
                    retryable = value["retryable"]
                    request_id = value["requestId"]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        raise ArtifactAPIError(response.status_code, code, retryable, request_id)


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
            not runtime_instance_id.strip()
            or runtime_instance_id != runtime_instance_id.strip()
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
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                self._host,
                self._port,
                ssl=self._context,
                server_hostname=self._host,
                limit=64 * 1024,
            ),
            timeout=self._timeout,
        )
        try:
            ssl_object = writer.get_extra_info("ssl_object")
            if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                raise ssl.SSLCertVerificationError("private connection has no TLS peer")
            verify_control_plane_peer(ssl_object)
            target = self._base_path + path
            host_name = f"[{self._host}]" if ":" in self._host else self._host
            host = host_name if self._port == 443 else f"{host_name}:{self._port}"
            request_headers = {
                "Host": host,
                "Content-Length": str(len(body)),
                "Connection": "close",
                **headers,
                "X-Request-ID": f"request_{uuid.uuid4().hex}",
                RUNTIME_INSTANCE_HEADER: self._runtime_instance_id,
            }
            encoded_headers = _encode_headers(request_headers)
            request = f"{method} {target} HTTP/1.1\r\n".encode("ascii")
            writer.write(request + encoded_headers + b"\r\n" + body)
            await asyncio.wait_for(writer.drain(), timeout=self._timeout)
            status, response_headers = await asyncio.wait_for(
                _read_response_head(reader), timeout=self._timeout
            )
            response_body = await asyncio.wait_for(
                _read_response_body(reader, response_headers, max_response_bytes),
                timeout=self._timeout,
            )
            return ArtifactHTTPResponse(status, response_headers, response_body)
        except ArtifactClientError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ArtifactTransportError(
                f"Artifact API transport failed ({type(error).__name__})"
            ) from None
        finally:
            writer.close()
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                writer.transport.abort()
            else:
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=min(0.5, self._timeout))
                except Exception:
                    writer.transport.abort()


def _validate_ref(ref: ArtifactRef) -> None:
    _validate_component("namespace", ref.namespace)
    _validate_component("name", ref.name)
    if ref.revision is not None and not ref.revision.strip():
        raise ValueError("artifact revision must not be empty")


def _validate_component(field_name: str, value: str) -> None:
    if not ARTIFACT_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"artifact {field_name} is invalid")


def _validate_media_type(value: str) -> None:
    if value == "*/*" or MEDIA_TYPE_PATTERN.fullmatch(value) is None:
        raise ValueError("artifact media type is invalid")


def _response_media_type(headers: Mapping[str, str]) -> str:
    value = headers.get("content-type", "")
    _validate_media_type(value)
    return value


def _require_json(headers: Mapping[str, str]) -> None:
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


def _encode_headers(headers: Mapping[str, str]) -> bytes:
    result = bytearray()
    for name, value in headers.items():
        if (
            not name
            or any(char in name for char in "\r\n:")
            or any(char in value for char in "\r\n")
        ):
            raise ValueError("invalid private HTTP header")
        result.extend(f"{name}: {value}\r\n".encode("ascii"))
    return bytes(result)


async def _read_response_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    status_line = await reader.readline()
    if not status_line.endswith(b"\r\n") or len(status_line) > 8192:
        raise ArtifactTransportError("invalid Artifact API status line")
    parts = status_line.decode("ascii", errors="strict").strip().split(" ", 2)
    if len(parts) < 2 or parts[0] not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ArtifactTransportError("invalid Artifact API status line")
    try:
        status = int(parts[1])
    except ValueError as error:
        raise ArtifactTransportError("invalid Artifact API status code") from error
    headers: dict[str, str] = {}
    total = len(status_line)
    for _ in range(MAX_RESPONSE_HEADERS):
        line = await reader.readline()
        total += len(line)
        if total > 64 * 1024 or not line.endswith(b"\r\n"):
            raise ArtifactTransportError("invalid or oversized Artifact API headers")
        if line == b"\r\n":
            return status, headers
        name, separator, value = line.partition(b":")
        key = name.decode("ascii", errors="strict").strip().lower()
        if not separator or not key or key in headers:
            raise ArtifactTransportError("invalid or duplicate Artifact API response header")
        headers[key] = value.decode("ascii", errors="strict").strip()
    raise ArtifactTransportError("too many Artifact API response headers")


async def _read_response_body(
    reader: asyncio.StreamReader,
    headers: Mapping[str, str],
    maximum: int,
) -> bytes:
    transfer_encoding = headers.get("transfer-encoding", "").lower()
    if transfer_encoding:
        if transfer_encoding != "chunked":
            raise ArtifactTransportError("unsupported Artifact API transfer encoding")
        return await _read_chunked_body(reader, maximum)
    if "content-length" in headers:
        try:
            length = int(headers["content-length"])
        except ValueError as error:
            raise ArtifactTransportError("invalid Artifact API content length") from error
        if not 0 <= length <= maximum:
            raise ArtifactResponseLimitError("Artifact API response exceeds its size limit")
        return await reader.readexactly(length)
    result = bytearray()
    while True:
        chunk = await reader.read(min(64 * 1024, maximum + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum:
            raise ArtifactResponseLimitError("Artifact API response exceeds its size limit")


async def _read_chunked_body(reader: asyncio.StreamReader, maximum: int) -> bytes:
    result = bytearray()
    while True:
        size_line = await reader.readline()
        raw_size = size_line.partition(b";")[0].strip()
        try:
            size = int(raw_size, 16)
        except ValueError as error:
            raise ArtifactTransportError("invalid Artifact API chunk size") from error
        if size < 0 or len(result) + size > maximum:
            raise ArtifactResponseLimitError("Artifact API response exceeds its size limit")
        if size == 0:
            if await reader.readline() != b"\r\n":
                raise ArtifactTransportError("Artifact API chunked trailers are not supported")
            return bytes(result)
        result.extend(await reader.readexactly(size))
        if await reader.readexactly(2) != b"\r\n":
            raise ArtifactTransportError("invalid Artifact API chunk delimiter")
