"""Allocation-bound client for the Control Plane private Artifact API."""

from __future__ import annotations

import asyncio
import json
import re
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote, urlsplit

from pydantic import ValidationError

from contractor_runtime.contracts import (
    ArtifactListResult,
    ArtifactReadResult,
    ArtifactRef,
    ArtifactWriteResult,
)
from contractor_runtime.mtls import verify_control_plane_peer

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 1 << 20
MAX_RESPONSE_HEADERS = 64
PATH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&'+.^_`|~-]*$"
)


class ArtifactClientError(Exception):
    """Bounded error safe to record in allocation metrics."""


class ArtifactTransportError(ArtifactClientError):
    pass


class ArtifactAPIError(ArtifactClientError):
    def __init__(self, status_code: int, code: str, retryable: bool) -> None:
        super().__init__(f"Artifact API returned HTTP {status_code} ({code})")
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


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


class ArtifactClient:
    """Exposes only the RunScope implied by one active allocation ID."""

    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        if PATH_ID_PATTERN.fullmatch(allocation_id) is None:
            raise ValueError("allocation ID is not a safe URL path segment")
        self._allocation_id = allocation_id
        self._transport = transport
        self._root = f"/allocations/{quote(allocation_id, safe='')}/artifacts"

    async def list_artifacts(self, namespace: str | None = None) -> list[ArtifactRef]:
        path = self._root
        if namespace is not None:
            _validate_component("namespace", namespace)
            path += f"?namespace={quote(namespace, safe='')}"
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
        return result.artifacts

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        _validate_ref(ref)
        path = self._ref_path(ref)
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "*/*"},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_BYTES,
        )
        self._raise_for_status(response)
        media_type = _response_media_type(response.headers)
        revision = _strong_etag(response.headers)
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
        return ArtifactValue(artifact=exact, media_type=media_type, data=response.body)

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        _validate_ref(target)
        if target.revision is not None:
            raise ValueError("artifact write target must be versionless")
        if not isinstance(data, bytes):
            raise TypeError("artifact payload must be bytes")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact payload exceeds the 16 MiB limit")
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
        if (
            result.artifact.namespace != target.namespace
            or result.artifact.name != target.name
            or result.artifact.revision != revision
            or result.media_type != media_type
            or result.size != len(data)
        ):
            raise ArtifactTransportError("Artifact API write response does not match the request")
        return result

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
        if len(response.body) <= MAX_ARTIFACT_JSON_BYTES:
            try:
                value = json.loads(response.body)
                if (
                    isinstance(value, dict)
                    and set(value) == {"code", "message", "retryable"}
                    and isinstance(value["code"], str)
                    and value["code"].strip()
                    and isinstance(value["message"], str)
                    and value["message"].strip()
                    and isinstance(value["retryable"], bool)
                ):
                    code = value["code"]
                    retryable = value["retryable"]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        raise ArtifactAPIError(response.status_code, code, retryable)


class MTLSArtifactTransport:
    """Minimal HTTPS/1.1 transport with pre-request Control Plane role check."""

    def __init__(self, base_url: str, context: ssl.SSLContext, timeout_seconds: float) -> None:
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
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._base_path = base_path
        self._context = context
        self._timeout = timeout_seconds

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        if method not in {"GET", "PUT"} or not path.startswith("/") or "#" in path:
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
    if not value.strip() or "/" in value or "\x00" in value:
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
            raise ArtifactTransportError("Artifact API response exceeds its size limit")
        return await reader.readexactly(length)
    result = bytearray()
    while True:
        chunk = await reader.read(min(64 * 1024, maximum + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum:
            raise ArtifactTransportError("Artifact API response exceeds its size limit")


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
            raise ArtifactTransportError("Artifact API response exceeds its size limit")
        if size == 0:
            if await reader.readline() != b"\r\n":
                raise ArtifactTransportError("Artifact API chunked trailers are not supported")
            return bytes(result)
        result.extend(await reader.readexactly(size))
        if await reader.readexactly(2) != b"\r\n":
            raise ArtifactTransportError("invalid Artifact API chunk delimiter")
