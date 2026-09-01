"""Allocation-owned, static-operation Caido GraphQL transport."""

from __future__ import annotations

import asyncio
import json
import math
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import httpx

from contractor_runtime.adapters.host import (
    AdapterFactoryError,
    AdapterHandles,
    AdapterSettings,
    RuntimeAdapterBuildContext,
    RuntimeAdapterMetricsState,
)
from contractor_runtime.contracts import CaidoSettingsV2, RuntimeAdapterRef

MAX_CAIDO_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CAIDO_VARIABLE_BYTES = 2 * 1024 * 1024
MAX_CAIDO_VARIABLE_DEPTH = 16
MAX_CAIDO_VARIABLE_ITEMS = 10_000
MAX_CAIDO_RESPONSE_DEPTH = 32
MAX_CAIDO_RESPONSE_ITEMS = 100_000
MAX_CAIDO_VARIABLE_KEY_BYTES = 128
MAX_CAIDO_VARIABLE_STRING_BYTES = 1024 * 1024
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
CaidoOperationID = Literal[
    "automate_entry_requests",
    "automate_session",
    "findings_by_offset",
    "request_detail",
    "requests_by_offset",
    "scopes",
    "sitemap_descendants",
    "sitemap_root",
    "workflows",
]


@dataclass(frozen=True, slots=True)
class _StaticOperation:
    operation_name: str
    document: str


# The handle accepts these identifiers only. Tool code never passes a GraphQL
# document, and adding another operation requires a reviewed source change.
_STATIC_OPERATIONS: Mapping[str, _StaticOperation] = MappingProxyType(
    {
        "automate_entry_requests": _StaticOperation(
            operation_name="AutomateEntryRequests",
            document=(
                "query AutomateEntryRequests($id: ID!, $limit: Int, $offset: Int, "
                "$order: AutomateEntryRequestOrderInput) { automateEntry(id: $id) { id name "
                "requestsByOffset(limit: $limit, offset: $offset, order: $order) { count { "
                "value } nodes { sequenceId error payloads { position raw } request { id method "
                "host path query response { statusCode length roundtripTime } } } } } }"
            ),
        ),
        "automate_session": _StaticOperation(
            operation_name="AutomateSession",
            document=(
                "query AutomateSession($id: ID!) { automateSession(id: $id) { id name entries { "
                "id name createdAt } settings { strategy placeholders { start end } } } }"
            ),
        ),
        "findings_by_offset": _StaticOperation(
            operation_name="FindingsByOffset",
            document=(
                "query FindingsByOffset($limit: Int, $offset: Int, $order: FindingOrderInput) { "
                "findingsByOffset(limit: $limit, offset: $offset, order: $order) { count { value } "
                "nodes { id title description host path reporter createdAt request { id method "
                "host path } } } }"
            ),
        ),
        "request_detail": _StaticOperation(
            operation_name="RequestDetail",
            document=(
                "query RequestDetail($id: ID!) { request(id: $id) { id method host path port query "
                "isTls raw createdAt source response { id statusCode length roundtripTime raw } } }"
            ),
        ),
        "requests_by_offset": _StaticOperation(
            operation_name="RequestsByOffset",
            document=(
                "query RequestsByOffset($limit: Int, $offset: Int, $filter: HTTPQL, "
                "$order: RequestResponseOrderInput) { requestsByOffset(limit: $limit, offset: "
                "$offset, filter: $filter, order: $order) { count { value } nodes { id method host "
                "path port query isTls source createdAt response { statusCode length roundtripTime "
                "} } } }"
            ),
        ),
        "scopes": _StaticOperation(
            operation_name="Scopes",
            document=("query Scopes { scopes { id name allowlist denylist } }"),
        ),
        "sitemap_descendants": _StaticOperation(
            operation_name="SitemapDescendants",
            document=(
                "query SitemapDescendants($parentId: ID!, $depth: SitemapDescendantsDepth!) { "
                "sitemapDescendantEntries(parentId: $parentId, depth: $depth) { nodes { id label "
                "kind hasDescendants parentId metadata { ... on SitemapEntryMetadataDomain { isTls "
                "port } } } } }"
            ),
        ),
        "sitemap_root": _StaticOperation(
            operation_name="SitemapRoot",
            document=(
                "query SitemapRoot($scopeId: ID) { sitemapRootEntries(scopeId: $scopeId) { nodes { "
                "id label kind hasDescendants metadata { ... on SitemapEntryMetadataDomain { isTls "
                "port } } } } }"
            ),
        ),
        "workflows": _StaticOperation(
            operation_name="Workflows",
            document="query Workflows { workflows { id name kind enabled global } }",
        ),
    }
)


class CaidoClientError(RuntimeError):
    """Bounded failure safe to retain in a tool result or metric."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        if code not in {
            "caido_request_invalid",
            "caido_request_failed",
            "caido_response_invalid",
            "caido_response_too_large",
        }:
            raise ValueError("unknown Caido client error code")
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class CaidoCloseError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Caido client close failed")


class CaidoGraphQLClient:
    """Narrow handle exposing only implementation-enumerated operations."""

    __slots__ = ("_client", "_graphql_url", "_metrics")

    def __init__(
        self,
        *,
        endpoint: str,
        bearer_token: str | None,
        ca_bundle_pem: str | None,
        timeout_seconds: float,
        metrics: RuntimeAdapterMetricsState,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        tls_context = ssl.create_default_context()
        if ca_bundle_pem is not None:
            tls_context.load_verify_locations(cadata=ca_bundle_pem)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if bearer_token is not None:
            headers["Authorization"] = f"Bearer {bearer_token}"
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
        selected_transport = transport or httpx.AsyncHTTPTransport(
            verify=tls_context,
            trust_env=False,
            retries=0,
            limits=limits,
        )
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(
            transport=selected_transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            limits=limits,
            headers=headers,
        )
        self._graphql_url = f"{endpoint.rstrip('/')}/graphql"
        self._metrics: RuntimeAdapterMetricsState | None = metrics

    @property
    def closed(self) -> bool:
        return self._client is None

    async def execute(
        self,
        operation: CaidoOperationID | str,
        variables: Mapping[str, JSONValue] | None = None,
    ) -> dict[str, Any]:
        selected = _STATIC_OPERATIONS.get(operation)
        client = self._client
        metrics = self._metrics
        if selected is None or client is None or metrics is None:
            raise CaidoClientError("caido_request_invalid", retryable=False)
        try:
            selected_variables = _validate_variables(variables)
            body = _encode_request(selected, selected_variables)
            request = client.build_request("POST", self._graphql_url, content=body)
            response = await client.send(request, stream=True)
            try:
                if response.status_code < 200 or response.status_code >= 300:
                    raise CaidoClientError(
                        "caido_request_failed",
                        retryable=response.status_code in _TRANSIENT_STATUS_CODES,
                    )
                raw = await _read_bounded_response(response)
            finally:
                await response.aclose()
            decoded = _decode_response(raw)
        except asyncio.CancelledError:
            raise
        except CaidoClientError as error:
            metrics.record_operation(succeeded=False, error_code="request_failed")
            raise error from None
        except Exception:
            metrics.record_operation(succeeded=False, error_code="request_failed")
            raise CaidoClientError("caido_request_failed", retryable=True) from None
        metrics.record_operation(succeeded=True)
        return decoded

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._graphql_url = ""
        self._metrics = None
        if client is not None:
            await client.aclose()

    def __repr__(self) -> str:
        return f"CaidoGraphQLClient(active={self._client is not None!r})"


class CaidoGraphQLAdapterFactory:
    ref = "caido-graphql@1"

    async def probe(self) -> bool:
        client: httpx.AsyncClient | None = None
        try:
            context = ssl.create_default_context()
            transport = httpx.AsyncHTTPTransport(
                verify=context,
                trust_env=False,
                retries=0,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            )
            client = httpx.AsyncClient(transport=transport, trust_env=False)
        except Exception:
            return False
        finally:
            if client is not None:
                await client.aclose()
        return True

    async def create(
        self,
        context: RuntimeAdapterBuildContext,
        settings: AdapterSettings,
    ) -> CaidoGraphQLAdapter:
        if not isinstance(settings, CaidoSettingsV2):
            raise AdapterFactoryError(retryable=False)
        try:
            return CaidoGraphQLAdapter(context, settings)
        except Exception:
            raise AdapterFactoryError(retryable=False) from None

    def __repr__(self) -> str:
        return "CaidoGraphQLAdapterFactory(ref='caido-graphql@1')"


class CaidoGraphQLAdapter:
    ref: RuntimeAdapterRef = "caido-graphql@1"

    def __init__(
        self,
        context: RuntimeAdapterBuildContext,
        settings: CaidoSettingsV2,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.metrics = RuntimeAdapterMetricsState()
        token = (
            settings.bearer_token.get_secret_value() if settings.bearer_token is not None else None
        )
        client = CaidoGraphQLClient(
            endpoint=settings.endpoint,
            bearer_token=token,
            ca_bundle_pem=settings.ca_bundle_pem,
            timeout_seconds=min(
                float(context.request_timeout_seconds),
                float(settings.request_timeout_seconds),
            ),
            metrics=self.metrics,
            transport=transport,
        )
        self._handle: CaidoGraphQLClient | None = client
        self.handles = AdapterHandles(caido_graphql=client)
        self._closed = False

    async def flush(self) -> None:
        return

    async def close(self) -> None:
        if self._closed:
            return
        handle = self._handle
        self._handle = None
        self.handles = AdapterHandles()
        self._closed = True
        if handle is None:
            return
        try:
            await handle.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise CaidoCloseError from None

    def __repr__(self) -> str:
        return f"CaidoGraphQLAdapter(ref={self.ref!r}, closed={self._closed!r})"


def _validate_variables(variables: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    if variables is None:
        return {}
    if not isinstance(variables, Mapping):
        raise CaidoClientError("caido_request_invalid", retryable=False)
    result = dict(variables)
    items = 0

    def visit(value: object, depth: int) -> None:
        nonlocal items
        if depth > MAX_CAIDO_VARIABLE_DEPTH:
            raise CaidoClientError("caido_request_invalid", retryable=False)
        items += 1
        if items > MAX_CAIDO_VARIABLE_ITEMS:
            raise CaidoClientError("caido_request_invalid", retryable=False)
        if value is None or isinstance(value, (str, bool, int, float)):
            if (
                isinstance(value, str)
                and len(value.encode("utf-8")) > MAX_CAIDO_VARIABLE_STRING_BYTES
            ):
                raise CaidoClientError("caido_request_invalid", retryable=False)
            if isinstance(value, float) and not math.isfinite(value):
                raise CaidoClientError("caido_request_invalid", retryable=False)
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key.encode("utf-8")) > MAX_CAIDO_VARIABLE_KEY_BYTES
                ):
                    raise CaidoClientError("caido_request_invalid", retryable=False)
                visit(nested, depth + 1)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                visit(nested, depth + 1)
            return
        raise CaidoClientError("caido_request_invalid", retryable=False)

    visit(result, 0)
    try:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise CaidoClientError("caido_request_invalid", retryable=False) from None
    if len(encoded) > MAX_CAIDO_VARIABLE_BYTES:
        raise CaidoClientError("caido_request_invalid", retryable=False)
    return result


def _encode_request(
    operation: _StaticOperation,
    variables: Mapping[str, JSONValue],
) -> bytes:
    return json.dumps(
        {
            "operationName": operation.operation_name,
            "query": operation.document,
            "variables": variables,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


async def _read_bounded_response(response: httpx.Response) -> bytes:
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError:
            raise CaidoClientError("caido_response_invalid", retryable=False) from None
        if content_length < 0:
            raise CaidoClientError("caido_response_invalid", retryable=False)
        if content_length > MAX_CAIDO_RESPONSE_BYTES:
            raise CaidoClientError("caido_response_too_large", retryable=False)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_CAIDO_RESPONSE_BYTES:
            raise CaidoClientError("caido_response_too_large", retryable=False)
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_response(raw: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise CaidoClientError("caido_response_invalid", retryable=False) from None
    _validate_response_shape(decoded)
    assert isinstance(decoded, dict)
    errors = decoded.get("errors")
    if errors:
        raise CaidoClientError("caido_request_failed", retryable=False)
    data = decoded.get("data")
    if not isinstance(data, dict):
        raise CaidoClientError("caido_response_invalid", retryable=False)
    return data


def _validate_response_shape(value: object) -> None:
    items = 0

    def visit(nested: object, depth: int) -> None:
        nonlocal items
        if depth > MAX_CAIDO_RESPONSE_DEPTH:
            raise CaidoClientError("caido_response_invalid", retryable=False)
        items += 1
        if items > MAX_CAIDO_RESPONSE_ITEMS:
            raise CaidoClientError("caido_response_invalid", retryable=False)
        if nested is None or isinstance(nested, (str, bool, int)):
            return
        if isinstance(nested, float):
            if not math.isfinite(nested):
                raise CaidoClientError("caido_response_invalid", retryable=False)
            return
        if isinstance(nested, dict):
            for key, child in nested.items():
                if not isinstance(key, str):
                    raise CaidoClientError("caido_response_invalid", retryable=False)
                visit(child, depth + 1)
            return
        if isinstance(nested, list):
            for child in nested:
                visit(child, depth + 1)
            return
        raise CaidoClientError("caido_response_invalid", retryable=False)

    visit(value, 0)
    if not isinstance(value, dict) or not set(value) <= {"data", "errors", "extensions"}:
        raise CaidoClientError("caido_response_invalid", retryable=False)
    if "errors" in value and not isinstance(value["errors"], list):
        raise CaidoClientError("caido_response_invalid", retryable=False)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite number")
