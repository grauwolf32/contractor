"""Allocation-bound client for the Control Plane private Artifact API.

The module also holds what every client of that API shares: its transport,
error model and response checks. Finding proposals
(:mod:`contractor_runtime.finding_proposals`) and model recovery
(:mod:`contractor_runtime.llm.recovery`) have clients of their own.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import quote

from pydantic import Field, ValidationError

from contractor_runtime import _https, threads
from contractor_runtime._allocation_errors import (
    ArtifactAPIError as ArtifactAPIError,
)
from contractor_runtime._allocation_errors import (
    ArtifactClientError as ArtifactClientError,
)
from contractor_runtime._allocation_errors import (
    ArtifactResponseLimitError as ArtifactResponseLimitError,
)
from contractor_runtime._allocation_errors import (
    ArtifactTransportError as ArtifactTransportError,
)
from contractor_runtime._bulk_rpc import PayloadDownload
from contractor_runtime.contracts import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    ArtifactListResult,
    ArtifactMetadataResult,
    ArtifactRef,
    ArtifactWriteResult,
    PrivateProtocolDecodeError,
    decode_private,
    validate_artifact_name,
    validate_media_type,
)
from contractor_runtime.contracts.base import (
    NAME_PATTERN,
    RESOURCE_ID_PATTERN,
    is_blank,
    parse_private_rpc_origin,
    trim_whitespace,
)
from contractor_runtime.contracts.grpc_wire.bulk import ArtifactPayloadMetadata
from contractor_runtime.shared_config import artifact_list_max_items, artifact_payload_max_bytes

if TYPE_CHECKING:
    from contractor_runtime._rpc_connections import RPCConnections
    from contractor_runtime.archive_cache import ImmutableArchiveCache

MAX_ARTIFACT_JSON_BYTES = 1 << 20
# artifacts.MaxBindingListLimit (LIM-ART-3), pinned by
# api/testdata/v1alpha1/artifact-list-cases.json.
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


# A private Artifact API answer to a CAS write whose binding's retained
# history needs more maintenance than one Server transaction may do. The write
# stored nothing and the Server committed one maintenance pass before it
# answered, so the same write may be sent again (ART-R127).
ARTIFACT_HISTORY_MAINTENANCE_PENDING = "artifact_history_maintenance_pending"


@dataclass(frozen=True, slots=True)
class ArtifactHTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


class ArtifactTransport(Protocol):
    async def planner_operation(
        self, allocation_id: str, value: Mapping, *, timeout_seconds: float
    ) -> Mapping: ...

    async def gateway_recovery(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def submit_audit_evidence(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def submit_audit_judgments(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def mutate_sitemap(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def snapshot_sitemap(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def collect_sitemap(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def list_artifacts(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def read_artifact_metadata(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def check_artifact_access(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def list_findings(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def read_finding(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def select_findings(self, allocation_id: str, value: Mapping) -> Mapping: ...

    async def read_artifact(
        self, allocation_id: str, value: Mapping, *, max_response_bytes: int
    ) -> PayloadDownload: ...

    async def write_artifact(self, allocation_id: str, value: Mapping, data: bytes) -> Mapping: ...

    async def submit_finding(self, allocation_id: str, submission) -> Mapping: ...


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
        if (
            not isinstance(allocation_id, str)
            or RESOURCE_ID_PATTERN.fullmatch(allocation_id) is None
        ):
            raise ValueError("allocation ID is invalid")
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
        value = {key: int(item) if key == "limit" else item for key, item in query}
        response = await self._transport.list_artifacts(self.allocation_id, value)
        try:
            return await threads.to_thread_until_done(
                lambda: decode_private(
                    ArtifactListResult, json.dumps(response, ensure_ascii=False, allow_nan=False)
                ),
                name="artifact-list-validation",
            )
        except PrivateProtocolDecodeError:
            raise ArtifactTransportError("Artifact API returned an invalid list response") from None

    async def read_artifact_metadata(
        self, ref: ArtifactRef, *, target: ArtifactReadTarget | None = None
    ) -> ArtifactMetadataResult:
        """Return the exact revision, media type and size ``read_artifact`` of
        ``ref`` would return, without downloading the payload (ART-R116).

        The result is not a content observation: only a payload read records
        the exact revision for invocation provenance.
        """

        value = self._target_value(ref, target=target)
        response = await self._transport.read_artifact_metadata(self.allocation_id, value)
        try:
            result = await threads.to_thread_until_done(
                lambda: decode_private(
                    ArtifactMetadataResult,
                    json.dumps(response, ensure_ascii=False, allow_nan=False),
                ),
                name="artifact-metadata-validation",
            )
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
        value = self._target_value(ref, target=target)
        response = await self._transport.read_artifact(
            self.allocation_id, value, max_response_bytes=max_bytes
        )
        if not isinstance(response, PayloadDownload) or not isinstance(response.data, bytes):
            raise ArtifactTransportError("Artifact RPC returned an invalid payload result")
        if len(response.data) > max_bytes:
            raise ArtifactResponseLimitError("Artifact payload exceeds the read limit")
        metadata = self._payload_metadata(response.metadata)
        if (
            metadata.artifact.namespace != ref.namespace
            or metadata.artifact.name != ref.name
            or (ref.revision is not None and metadata.artifact.revision != ref.revision)
            or metadata.size != len(response.data)
        ):
            raise ArtifactTransportError("Artifact RPC payload does not match the request")
        if target is None:
            self._remember(metadata.artifact)
        return ArtifactValue(
            artifact=metadata.artifact,
            media_type=metadata.media_type,
            data=response.data,
            binding_created_at=metadata.binding_created_at,
            revision_created_at=metadata.revision_created_at,
        )

    @staticmethod
    def _payload_metadata(value) -> ArtifactPayloadMetadata:
        try:
            # Wire decoding already ran off-loop in the transport. Keep this
            # small DTO check synchronous so a received receipt remains known
            # before the invocation owner's next cancellation boundary.
            return ArtifactPayloadMetadata.model_validate(value)
        except (ValidationError, ValueError, TypeError):
            raise ArtifactTransportError("Artifact RPC returned invalid payload metadata") from None

    async def check_artifact_access(self, ref: ArtifactRef, *, target: ArtifactReadTarget) -> None:
        """Recheck current operation authority without resolving content/history."""

        if not isinstance(target, ArtifactReadTarget):
            raise ValueError("artifact read target is invalid")
        if ref.revision is not None:
            raise ValueError("artifact access target must be versionless")
        self._target_value(ref, target=target)
        response = await self._transport.check_artifact_access(
            self.allocation_id,
            {
                "artifact": ref.model_dump(mode="json", by_alias=True, exclude_none=True),
                "scope": "project",
                "operation": target.operation,
            },
        )
        if response != {}:
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
        value = {**self._target_value(target), "mediaType": media_type, "size": len(data)}
        if expected_revision is None:
            value["createOnly"] = True
        else:
            if not isinstance(expected_revision, str) or is_blank(expected_revision):
                raise ValueError("expected revision is invalid")
            value["expectedRevision"] = expected_revision
        response = await self._transport.write_artifact(self.allocation_id, value, data)
        metadata = self._payload_metadata(response)
        if (
            metadata.artifact.namespace != target.namespace
            or metadata.artifact.name != target.name
            or metadata.media_type != media_type
            or metadata.size != len(data)
        ):
            raise ArtifactTransportError("Artifact RPC write response does not match the request")
        self._remember(metadata.artifact)
        return ArtifactWriteValue(
            apiVersion="contractor/v1alpha1",
            artifact=metadata.artifact,
            mediaType=metadata.media_type,
            size=metadata.size,
            binding_created_at=metadata.binding_created_at,
            revision_created_at=metadata.revision_created_at,
        )

    def _remember(self, ref: ArtifactRef) -> None:
        revision = ref.require_exact().revision
        if not (revision is not None):
            raise RuntimeError("Expected revision is not None")
        self._known_exact_refs[(ref.namespace, ref.name)] = ref
        self._observed_exact_refs.append(ref)

    @staticmethod
    def _target_value(ref: ArtifactRef, *, target: ArtifactReadTarget | None = None) -> dict:
        if not isinstance(ref, ArtifactRef):
            raise ValueError("artifact ref is invalid")
        ref = ArtifactRef.model_validate(
            ref.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        value = {"artifact": ref.model_dump(mode="json", by_alias=True, exclude_none=True)}
        if target is not None:
            if not isinstance(target, ArtifactReadTarget):
                raise ValueError("artifact read target is invalid")
            value.update(scope=target.scope, operation=target.operation)
        return value


def raise_for_status(response: ArtifactHTTPResponse) -> None:
    """Raise the private API error a non-2xx response carries."""

    if 200 <= response.status_code < 300:
        return
    error = _https.decode_private_error(response.body)
    if error is None:
        raise ArtifactAPIError(response.status_code, "invalid_error_response", False)
    raise ArtifactAPIError(response.status_code, error.code, error.retryable, error.request_id)


class MTLSArtifactTransport:
    """Named private RPCs over independently owned verified domain/bulk channels."""

    def __init__(
        self,
        base_url: str,
        rpc_connections: RPCConnections,
        bulk_connections: RPCConnections,
        timeout_seconds: float,
        runtime_instance_id: str,
    ) -> None:
        parse_private_rpc_origin("Artifact RPC URL", base_url)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Artifact RPC timeout must be finite and positive")
        if (
            is_blank(runtime_instance_id)
            or runtime_instance_id != trim_whitespace(runtime_instance_id)
            or len(runtime_instance_id) > 256
        ):
            raise ValueError("Runtime Agent instance ID is invalid")
        from contractor_runtime._allocation_rpc import AllocationRPC
        from contractor_runtime._bulk_rpc import BulkRPC

        origin = base_url.rstrip("/")
        self._rpc = AllocationRPC(rpc_connections, origin, runtime_instance_id, timeout_seconds)
        self._bulk = BulkRPC(bulk_connections, origin, runtime_instance_id, timeout_seconds)

    async def planner_operation(self, allocation_id, value, *, timeout_seconds):
        return await self._rpc.planner_operation(
            allocation_id, value, timeout_seconds=timeout_seconds
        )

    async def gateway_recovery(self, allocation_id, value):
        return await self._rpc.gateway_recovery(allocation_id, value)

    async def submit_audit_evidence(self, allocation_id, value):
        return await self._rpc.submit_audit_evidence(allocation_id, value)

    async def submit_audit_judgments(self, allocation_id, value):
        return await self._rpc.submit_audit_judgments(allocation_id, value)

    async def mutate_sitemap(self, allocation_id, value):
        return await self._rpc.mutate_sitemap(allocation_id, value)

    async def snapshot_sitemap(self, allocation_id, value):
        return await self._rpc.snapshot_sitemap(allocation_id, value)

    async def collect_sitemap(self, allocation_id, value):
        return await self._rpc.collect_sitemap(allocation_id, value)

    async def list_artifacts(self, allocation_id, value):
        return await self._rpc.list_artifacts(allocation_id, value)

    async def read_artifact_metadata(self, allocation_id, value):
        return await self._rpc.read_artifact_metadata(allocation_id, value)

    async def check_artifact_access(self, allocation_id, value):
        return await self._rpc.check_artifact_access(allocation_id, value)

    async def list_findings(self, allocation_id, value):
        return await self._rpc.list_findings(allocation_id, value)

    async def read_finding(self, allocation_id, value):
        return await self._rpc.read_finding(allocation_id, value)

    async def select_findings(self, allocation_id, value):
        return await self._rpc.select_findings(allocation_id, value)

    async def read_artifact(self, allocation_id, value, *, max_response_bytes):
        return await self._bulk.read_artifact(
            allocation_id, value, max_response_bytes=max_response_bytes
        )

    async def write_artifact(self, allocation_id, value, data):
        return await self._bulk.write_artifact(allocation_id, value, data)

    async def submit_finding(self, allocation_id, submission):
        return await self._bulk.submit_finding(allocation_id, submission)


def _validate_list_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= artifact_list_max_items():
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
