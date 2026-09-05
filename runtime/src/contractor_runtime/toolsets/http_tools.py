"""Bounded allocation-scoped HTTP exploration tools."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import re
import secrets
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient, ProxyRequestError
from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactClient,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings, RuntimeSettingsV2
from contractor_runtime.toolsets.artifact_visibility import HTTP_BODY_ARTIFACT_PREFIX
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

HTTP_BODY_MEDIA_TYPE = "application/vnd.contractor.http-body+json"
MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
MAX_PREVIEW_CHARACTERS = 8192
MAX_READ_UNITS = 8192
MAX_HEADERS = 64
MAX_QUERY_KEYS = 64
MAX_HEADER_VALUE_BYTES = 8192
MAX_HEADER_BYTES = 64 * 1024
MAX_QUERY_BYTES = 64 * 1024
MAX_URL_BYTES = 8192
MAX_HISTORY = 128
MAX_COOKIES = 128
MAX_REDIRECTS = 10
MAX_ATTEMPTS = 3
BODY_SCHEMA_VERSION = "1.0"

_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "DELETE", "HEAD", "OPTIONS"})
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_BODY_TYPES = frozenset({"none", "json", "form", "text"})
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization", "proxy-connection", "set-cookie"}
)
_FORBIDDEN_REQUEST_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-request-id",
    }
)
_HEADER_NAME = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TEXTUAL_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/problem+json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-www-form-urlencoded",
    "application/graphql",
)
_ERROR_RETRYABILITY = MappingProxyType(
    {
        "http_request_invalid": False,
        "http_target_denied": False,
        "http_request_failed": True,
        "http_response_too_large": False,
        "http_body_not_found": False,
    }
)


class HTTPToolError(RuntimeError):
    """Stable content-free model-facing HTTP failure."""

    def __init__(self, code: str) -> None:
        normalized = code if code in _ERROR_RETRYABILITY else "http_request_failed"
        self.code = normalized
        self.retryable = _ERROR_RETRYABILITY[normalized]
        super().__init__(f"HTTP operation failed ({normalized})")


class HTTPToolsetFactory:
    ref = "http-tools@1"
    exported_tools = frozenset(
        {
            "http_request",
            "http_read_body",
            "http_history",
            "http_session_set",
            "http_session_get",
            "http_session_clear",
        }
    )
    infrastructure_channels = MappingProxyType({"http_request": frozenset({"runtime-http-client"})})

    def __init__(
        self,
        artifact_client_factory: ArtifactClientFactory | None = None,
        direct_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._artifact_client_factory = artifact_client_factory or _unconfigured_client
        self._direct_client_factory = direct_client_factory or _direct_client

    async def probe(self) -> frozenset[str]:
        client: httpx.AsyncClient | None = None
        try:
            client = self._direct_client_factory()
        except Exception:
            return frozenset()
        finally:
            if client is not None:
                await client.aclose()
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, project_workspace
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("http-tools@1 requires State.metrics")

        proxy = adapter_handles.tool_http
        request_selected = "http_request" in selected
        proxy_required = request_selected and _tool_http_proxy_required(runtime_settings)
        if proxy is not None and not isinstance(proxy, ProxyHTTPClient):
            raise TypeError("http-tools@1 received an invalid runtime-http-client handle")
        if proxy_required and proxy is None:
            raise RuntimeError("http-tools@1 requires the resolved tool-http proxy route")

        direct_client = (
            self._direct_client_factory() if request_selected and proxy is None else None
        )
        session = _HTTPSession(
            artifact_client=self._artifact_client_factory(allocation_id, runtime_settings),
            namespace=namespace,
            timeout_cap_seconds=runtime_settings.request_timeout_seconds,
            forbidden_origins=_private_origins(runtime_settings),
            secrets_for_metrics=_runtime_secrets(runtime_settings),
            proxy=proxy,
            direct_client=direct_client,
            target_origin=_target_origin(runtime_settings),
            target_authorization=_target_authorization(runtime_settings),
        )
        builders: dict[str, Callable[[], _HTTPTool]] = {
            "http_request": lambda: HTTPRequestTool(session, metrics),
            "http_read_body": lambda: HTTPReadBodyTool(session, metrics),
            "http_history": lambda: HTTPHistoryTool(session, metrics),
            "http_session_set": lambda: HTTPSessionSetTool(session, metrics),
            "http_session_get": lambda: HTTPSessionGetTool(session, metrics),
            "http_session_clear": lambda: HTTPSessionClearTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


@dataclass(frozen=True, slots=True)
class _StoredBody:
    artifact: ArtifactRef
    kind: Literal["text", "binary"]


@dataclass(frozen=True, slots=True)
class _RequestRecord:
    request_id: int
    tag: str
    method: str
    final_url: str
    status: int
    content_type: str
    body_kind: Literal["empty", "text", "binary"]
    body_bytes: int
    artifact: ArtifactRef | None
    redirects: int
    retries: int
    elapsed_ms: int
    preview: str | None
    headers: dict[str, str] = field(repr=False)
    headers_truncated: bool = False

    def projection(self, *, include_preview: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "request_tag": self.tag,
            "method": self.method,
            "final_url": self.final_url,
            "status": self.status,
            "content_type": self.content_type,
            "content_length": self.body_bytes,
            "headers": dict(self.headers),
            "headers_truncated": self.headers_truncated,
            "body_kind": self.body_kind,
            "body_artifact": (
                None
                if self.artifact is None
                else self.artifact.model_dump(by_alias=True, exclude_none=True)
            ),
            "redirects": self.redirects,
            "retries": self.retries,
            "elapsed_ms": self.elapsed_ms,
        }
        if include_preview:
            result["body_preview"] = self.preview
            result["body_truncated"] = (
                self.body_kind == "text"
                and self.preview is not None
                and len(self.preview) >= MAX_PREVIEW_CHARACTERS
            )
        return result


class _HTTPSession:
    def __init__(
        self,
        *,
        artifact_client: ArtifactClient,
        namespace: str,
        timeout_cap_seconds: int,
        forbidden_origins: frozenset[tuple[str, str, int]],
        secrets_for_metrics: tuple[str, ...],
        proxy: ProxyHTTPClient | None,
        direct_client: httpx.AsyncClient | None,
        target_origin: tuple[str, str, int] | None,
        target_authorization: str | None,
    ) -> None:
        self._artifact_client = artifact_client
        self._namespace = namespace
        self._timeout_cap_seconds = timeout_cap_seconds
        self._forbidden_origins = forbidden_origins
        self._secrets_for_metrics = secrets_for_metrics
        self._proxy = proxy
        self._direct_client = direct_client
        self._target_origin = target_origin
        self._target_authorization = target_authorization
        self._lock = asyncio.Lock()
        self._history: deque[_RequestRecord] = deque(maxlen=MAX_HISTORY)
        self._bodies: dict[int, _StoredBody] = {}
        self._default_headers: dict[str, str] = {}
        self._cookies = httpx.Cookies()
        self._auth_kind: Literal["none", "basic", "bearer"] = "none"
        self._auth_username: str | None = None
        self._auth_secret: str | None = None
        self._next_request_id = 1
        self._nonce = secrets.token_hex(8)
        self._closed = False

    @property
    def metric_secrets(self) -> tuple[str, ...]:
        dynamic = tuple(
            value
            for value in (self._auth_username, self._auth_secret)
            if isinstance(value, str) and value
        )
        return self._secrets_for_metrics + dynamic

    async def request(
        self,
        *,
        url: str,
        method: str,
        headers: Mapping[str, Any] | None,
        query: Mapping[str, Any] | None,
        body_type: str,
        body: Any,
        timeout_seconds: int | float | None,
        follow_redirects: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            request_id = self._next_request_id
            self._next_request_id += 1
            started_ns = time.perf_counter_ns()
            selected_method = _method(method)
            selected_url = _url_with_query(url, query)
            selected_headers = _headers(headers)
            payload, content_type = _request_body(body_type, body)
            if content_type is not None and "content-type" not in {
                name.lower() for name in selected_headers
            }:
                selected_headers["Content-Type"] = content_type
            selected_timeout = _timeout(timeout_seconds, self._timeout_cap_seconds)
            if type(follow_redirects) is not bool:
                raise HTTPToolError("http_request_invalid")

            merged_headers = _merge_headers(self._default_headers, selected_headers)
            merged_headers["X-Request-Id"] = f"r{self._nonce}-h{request_id:06d}"
            tag = merged_headers["X-Request-Id"]
            # httpx keeps its own cookie jar in addition to the allocation
            # session jar.  Treat that transport jar as scratch space: stale
            # cookies left by a cancelled/failed prior call must never affect
            # the next request.
            self._clear_transport_cookies()
            response, final_method, final_url, redirects, retries = await self._send_following(
                method=selected_method,
                url=selected_url,
                headers=merged_headers,
                payload=payload,
                timeout_seconds=selected_timeout,
                follow_redirects=follow_redirects,
            )
            try:
                body_bytes = await _read_response_body(response)
                content_type = _content_type(response.headers)
                body_kind, preview, envelope = _encode_body(content_type, body_bytes)
                candidate_cookies = httpx.Cookies()
                candidate_cookies.update(self._cookies)
                candidate_cookies.update(response.cookies)
                if len(candidate_cookies) > MAX_COOKIES:
                    # httpx has already observed the response on its own
                    # client jar. Erase both copies before failing so an
                    # oversized Set-Cookie fan-out cannot dirty later calls.
                    self._cookies.clear()
                    self._clear_transport_cookies()
                    raise HTTPToolError("http_request_failed")
                artifact: ArtifactRef | None = None
                if envelope is not None:
                    encoded = json.dumps(
                        envelope,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if len(encoded) > MAX_ARTIFACT_BYTES:
                        raise HTTPToolError("http_response_too_large")
                    written = await self._artifact_client.write_artifact(
                        ArtifactRef(
                            namespace=self._namespace,
                            name=f"{HTTP_BODY_ARTIFACT_PREFIX}{self._nonce}.{request_id:06d}",
                        ),
                        data=encoded,
                        media_type=HTTP_BODY_MEDIA_TYPE,
                        expected_revision=None,
                    )
                    artifact = written.artifact.require_exact()
                    assert body_kind in {"text", "binary"}
                    self._bodies[request_id] = _StoredBody(artifact=artifact, kind=body_kind)
                safe_headers, headers_truncated = _response_headers(response.headers)
                self._cookies = candidate_cookies
                record = _RequestRecord(
                    request_id=request_id,
                    tag=tag,
                    method=final_method,
                    final_url=final_url,
                    status=response.status_code,
                    content_type=content_type,
                    body_kind=body_kind,
                    body_bytes=len(body_bytes),
                    artifact=artifact,
                    redirects=redirects,
                    retries=retries,
                    elapsed_ms=_elapsed_ms(started_ns),
                    preview=preview,
                    headers=safe_headers,
                    headers_truncated=headers_truncated,
                )
                self._history.append(record)
                return record.projection(include_preview=True)
            except HTTPToolError:
                raise
            except (ArtifactTransportError, ValueError, TypeError):
                raise HTTPToolError("http_request_failed") from None
            finally:
                await response.aclose()

    async def _send_following(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout_seconds: float,
        follow_redirects: bool,
    ) -> tuple[httpx.Response, str, str, int, int]:
        redirects = 0
        retries = 0
        current_method = method
        current_url = url
        current_headers = headers
        current_payload = payload
        allow_session_auth = True
        allow_session_cookies = True
        while True:
            _validate_target(current_url, self._forbidden_origins)
            try:
                response = await self._send_once(
                    current_method,
                    current_url,
                    headers=current_headers,
                    content=current_payload,
                    timeout=timeout_seconds,
                    allow_session_auth=allow_session_auth,
                    allow_session_cookies=allow_session_cookies,
                )
            except asyncio.CancelledError:
                raise
            except (ProxyRequestError, httpx.TransportError, httpx.TimeoutException):
                if current_method in _IDEMPOTENT_METHODS and retries + 1 < MAX_ATTEMPTS:
                    retries += 1
                    await asyncio.sleep(0)
                    continue
                raise HTTPToolError("http_request_failed") from None

            if (
                response.status_code in _RETRYABLE_STATUS
                and current_method in _IDEMPOTENT_METHODS
                and retries + 1 < MAX_ATTEMPTS
            ):
                retries += 1
                await response.aclose()
                await asyncio.sleep(0)
                continue

            location = response.headers.get("location")
            if not (
                follow_redirects and location and response.status_code in {301, 302, 303, 307, 308}
            ):
                return response, current_method, str(response.url), redirects, retries
            if redirects >= MAX_REDIRECTS:
                await response.aclose()
                raise HTTPToolError("http_request_failed")

            next_url = urljoin(current_url, location)
            _validate_target(next_url, self._forbidden_origins)
            next_method = current_method
            next_payload = current_payload
            next_headers = dict(current_headers)
            if response.status_code == 303 or (
                response.status_code in {301, 302} and current_method == "POST"
            ):
                next_method = "GET"
                next_payload = b""
                next_headers = {
                    name: value
                    for name, value in next_headers.items()
                    if name.lower() not in {"content-type", "content-encoding"}
                }
            if _origin(current_url) != _origin(next_url):
                next_headers = {
                    name: value
                    for name, value in next_headers.items()
                    if name.lower() not in {"authorization", "cookie"}
                }
                allow_session_auth = False
                allow_session_cookies = False
            await response.aclose()
            current_url = next_url
            current_method = next_method
            current_payload = next_payload
            current_headers = next_headers
            redirects += 1

    async def _send_once(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers"))
        allow_session_auth = bool(kwargs.pop("allow_session_auth", True))
        allow_session_cookies = bool(kwargs.pop("allow_session_cookies", True))
        if (
            self._target_origin is not None
            and self._target_authorization is not None
            and _origin(url) == self._target_origin
        ):
            headers = {
                name: value for name, value in headers.items() if name.lower() != "authorization"
            }
            headers["Authorization"] = self._target_authorization
        elif allow_session_auth and "authorization" not in {name.lower() for name in headers}:
            if self._auth_kind == "bearer" and self._auth_secret is not None:
                headers["Authorization"] = f"Bearer {self._auth_secret}"
            elif (
                self._auth_kind == "basic"
                and self._auth_username is not None
                and self._auth_secret is not None
            ):
                raw = f"{self._auth_username}:{self._auth_secret}".encode()
                headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
        kwargs["headers"] = headers
        if allow_session_cookies:
            kwargs["cookies"] = self._cookies
        if self._proxy is not None:
            return await self._proxy.stream_request(method, url, **kwargs)
        client = self._direct_client
        if client is None:
            raise HTTPToolError("http_request_failed")
        try:
            request = client.build_request(method, url, **kwargs)
            return await client.send(request, stream=True, follow_redirects=False)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError:
            raise
        except Exception:
            raise HTTPToolError("http_request_failed") from None

    async def read_body(self, request_id: int, offset: int, limit: int) -> dict[str, Any]:
        if type(request_id) is not int or request_id <= 0:
            raise HTTPToolError("http_request_invalid")
        if type(offset) is not int or offset < 0:
            raise HTTPToolError("http_request_invalid")
        if type(limit) is not int or not 1 <= limit <= MAX_READ_UNITS:
            raise HTTPToolError("http_request_invalid")
        async with self._lock:
            self._require_open()
            stored = self._bodies.get(request_id)
            if stored is None:
                raise HTTPToolError("http_body_not_found")
            try:
                value = await self._artifact_client.read_artifact(stored.artifact)
                if value.media_type != HTTP_BODY_MEDIA_TYPE:
                    raise HTTPToolError("http_body_not_found")
                kind, content = _decode_body_artifact(value.data)
                if kind != stored.kind:
                    raise HTTPToolError("http_body_not_found")
            except HTTPToolError:
                raise
            except Exception:
                raise HTTPToolError("http_body_not_found") from None
            if kind == "text":
                assert isinstance(content, str)
                selected = content[offset : offset + limit]
                return {
                    "request_id": request_id,
                    "kind": "text",
                    "unit": "characters",
                    "offset": offset,
                    "length": len(selected),
                    "total": len(content),
                    "eof": offset + len(selected) >= len(content),
                    "data": selected,
                }
            assert isinstance(content, bytes)
            selected_bytes = content[offset : offset + limit]
            return {
                "request_id": request_id,
                "kind": "binary",
                "unit": "bytes",
                "offset": offset,
                "length": len(selected_bytes),
                "total": len(content),
                "eof": offset + len(selected_bytes) >= len(content),
                "data_b64": base64.b64encode(selected_bytes).decode("ascii"),
            }

    async def history(self, limit: int) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY:
            raise HTTPToolError("http_request_invalid")
        async with self._lock:
            self._require_open()
            selected = tuple(self._history)[-limit:]
            return [record.projection(include_preview=False) for record in selected]

    async def set_session(
        self,
        *,
        headers: Mapping[str, Any] | None,
        cookies: Mapping[str, Any] | None,
        auth: Mapping[str, Any] | None,
        replace_cookies: bool,
        replace_headers: bool,
    ) -> dict[str, Any]:
        if type(replace_cookies) is not bool or type(replace_headers) is not bool:
            raise HTTPToolError("http_request_invalid")
        selected_headers = None if headers is None else _headers(headers)
        selected_cookies = None if cookies is None else _cookie_values(cookies)
        selected_auth = _auth(auth)
        async with self._lock:
            self._require_open()
            candidate_headers = self._default_headers
            if selected_headers is not None:
                candidate_headers = _merge_headers(
                    {} if replace_headers else self._default_headers,
                    selected_headers,
                )
            candidate_cookies = self._cookies
            if selected_cookies is not None:
                candidate_cookies = httpx.Cookies()
                if not replace_cookies:
                    candidate_cookies.update(self._cookies)
                for name, value in selected_cookies.items():
                    candidate_cookies.set(name, value)
                if len(candidate_cookies) > MAX_COOKIES:
                    raise HTTPToolError("http_request_invalid")
            # Commit only after every sparse component has passed validation.
            # One invalid cookie update must not partially replace headers.
            if selected_cookies is not None and replace_cookies:
                self._clear_transport_cookies()
            if selected_headers is not None:
                self._default_headers = candidate_headers
            if selected_cookies is not None:
                self._cookies = candidate_cookies
            if selected_auth is not None:
                self._erase_auth()
                self._auth_kind, self._auth_username, self._auth_secret = selected_auth
            return self._session_projection()

    async def get_session(self) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            return self._session_projection()

    async def clear_session(self) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            self._clear_mutable_session()
            return self._session_projection()

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            # Runtime adapters are deliberately terminated before the later
            # release cleanup closes model tools.  At that point a proxy
            # handle is detached and must not be dereferenced merely to erase
            # allocation-local state.  The adapter already closed its client;
            # clear the session-owned copy independently here.
            self._clear_local_session()
            self._bodies.clear()
            self._nonce = ""
            self._secrets_for_metrics = ()
            self._target_origin = None
            self._target_authorization = None
            client = self._direct_client
            self._direct_client = None
            self._proxy = None
            self._closed = True
        if client is not None:
            await client.aclose()

    def _session_projection(self) -> dict[str, Any]:
        headers = {
            name: ("[REDACTED]" if _is_sensitive_header(name) else value)
            for name, value in sorted(
                self._default_headers.items(), key=lambda item: item[0].lower()
            )
        }
        cookie_names = sorted({cookie.name for cookie in self._cookies.jar})
        return {
            "auth_kind": self._auth_kind,
            "default_headers": headers,
            "cookie_names": cookie_names,
            "cookie_count": len(cookie_names),
            "history_count": len(self._history),
        }

    def _clear_mutable_session(self) -> None:
        self._clear_local_session()
        self._clear_transport_cookies()

    def _clear_transport_cookies(self) -> None:
        if self._proxy is not None:
            self._proxy.clear_cookies()
        if self._direct_client is not None:
            self._direct_client.cookies.clear()

    def _clear_local_session(self) -> None:
        self._default_headers.clear()
        self._cookies.clear()
        self._history.clear()
        self._erase_auth()

    def _erase_auth(self) -> None:
        self._auth_kind = "none"
        self._auth_username = None
        self._auth_secret = None

    def _require_open(self) -> None:
        if self._closed:
            raise HTTPToolError("http_request_failed")


class _HTTPTool:
    name: str
    description: str

    def __init__(self, session: _HTTPSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        await self._session.close()

    def _success(
        self,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
        started_ns: int,
    ) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            result=result,
            secrets=self._session.metric_secrets,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _failure(
        self,
        arguments: Mapping[str, Any],
        error: Exception,
        started_ns: int,
    ) -> None:
        bounded = (
            error if isinstance(error, HTTPToolError) else HTTPToolError("http_request_failed")
        )
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            error=bounded,
            secrets=self._session.metric_secrets,
            duration_ms=_elapsed_ms(started_ns),
        )


class HTTPRequestTool(_HTTPTool):
    name = "http_request"
    description = "Send one bounded HTTP/HTTPS request and retain its complete body as an artifact."

    async def __call__(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        body: Any = None,
        body_type: str = "none",
        timeout: int | float | None = None,
        follow_redirects: bool = True,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        safe_arguments = {
            "method": (
                method.upper()
                if isinstance(method, str) and method.upper() in _METHODS
                else "invalid"
            ),
            "headerCount": len(headers) if isinstance(headers, dict) else 0,
            "queryCount": len(query) if isinstance(query, dict) else 0,
            "bodyType": (
                body_type if isinstance(body_type, str) and body_type in _BODY_TYPES else "invalid"
            ),
            "hasBody": body is not None,
            "followRedirects": follow_redirects if type(follow_redirects) is bool else False,
        }
        try:
            result = await self._session.request(
                url=url,
                method=method,
                headers=headers,
                query=query,
                body_type=body_type,
                body=body,
                timeout_seconds=timeout,
                follow_redirects=follow_redirects,
            )
            self._success(
                safe_arguments,
                {
                    "statusClass": result["status"] // 100,
                    "bodyKind": result["body_kind"],
                    "bodyBytes": result["content_length"],
                    "redirects": result["redirects"],
                    "retries": result["retries"],
                },
                started_ns,
            )
            return result
        except Exception as error:
            self._failure(safe_arguments, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_request_failed") from None


class HTTPReadBodyTool(_HTTPTool):
    name = "http_read_body"
    description = "Read one bounded text-character or binary-byte slice by HTTP request ID."

    async def __call__(
        self,
        request_id: int,
        offset: int = 0,
        length: int = MAX_READ_UNITS,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "requestId": request_id if type(request_id) is int else -1,
            "offset": offset if type(offset) is int else -1,
            "length": length if type(length) is int else -1,
        }
        try:
            result = await self._session.read_body(request_id, offset, length)
            self._success(
                arguments,
                {"kind": result["kind"], "length": result["length"], "eof": result["eof"]},
                started_ns,
            )
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_body_not_found") from None


class HTTPHistoryTool(_HTTPTool):
    name = "http_history"
    description = "List bounded HTTP request summaries from this allocation session, oldest first."

    async def __call__(self, limit: int = MAX_HISTORY) -> list[dict[str, Any]]:
        started_ns = time.perf_counter_ns()
        arguments = {"limit": limit if type(limit) is int else -1}
        try:
            result = await self._session.history(limit)
            self._success(arguments, {"count": len(result)}, started_ns)
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_request_failed") from None


class HTTPSessionSetTool(_HTTPTool):
    name = "http_session_set"
    description = "Replace selected allocation-memory HTTP headers, cookies or authentication."

    async def __call__(
        self,
        cookies: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: dict[str, Any] | None = None,
        replace_cookies: bool = False,
        replace_headers: bool = False,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "headerCount": len(headers) if isinstance(headers, dict) else 0,
            "cookieCount": len(cookies) if isinstance(cookies, dict) else 0,
            "authKind": (
                auth.get("kind")
                if isinstance(auth, dict)
                and isinstance(auth.get("kind"), str)
                and auth.get("kind") in {"none", "basic", "bearer"}
                else "invalid"
            ),
            "replaceCookies": replace_cookies,
            "replaceHeaders": replace_headers,
        }
        try:
            result = await self._session.set_session(
                headers=headers,
                cookies=cookies,
                auth=auth,
                replace_cookies=replace_cookies,
                replace_headers=replace_headers,
            )
            self._success(arguments, {"authKind": result["auth_kind"]}, started_ns)
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_request_invalid") from None


class HTTPSessionGetTool(_HTTPTool):
    name = "http_session_get"
    description = "Return a redacted view of allocation-memory HTTP session state."

    async def __call__(self) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.get_session()
            self._success({}, {"authKind": result["auth_kind"]}, started_ns)
            return result
        except Exception as error:
            self._failure({}, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_request_failed") from None


class HTTPSessionClearTool(_HTTPTool):
    name = "http_session_clear"
    description = "Erase allocation-memory HTTP headers, cookies, auth and request history."

    async def __call__(self) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await self._session.clear_session()
            self._success({}, {"cleared": True}, started_ns)
            return result
        except Exception as error:
            self._failure({}, error, started_ns)
            if isinstance(error, HTTPToolError):
                raise
            raise HTTPToolError("http_request_failed") from None


def _direct_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
    )


def _tool_http_proxy_required(settings: RuntimeSettings) -> bool:
    return (
        isinstance(settings, RuntimeSettingsV2)
        and settings.http_proxy is not None
        and "tool-http" in settings.http_proxy.targets
    )


def _private_origins(settings: RuntimeSettings) -> frozenset[tuple[str, str, int]]:
    values = [settings.llm_gateway_url, settings.artifact_api_url]
    if isinstance(settings, RuntimeSettingsV2):
        if settings.telemetry is not None:
            values.append(settings.telemetry.endpoint)
        if settings.http_proxy is not None:
            values.append(settings.http_proxy.proxy_url)
        if settings.caido is not None:
            values.append(settings.caido.endpoint)
    return frozenset(_origin(value) for value in values)


def _runtime_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    values: list[str] = []
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if isinstance(settings, RuntimeSettingsV2) and settings.http_proxy is not None:
        proxy = settings.http_proxy
        if proxy.basic_auth is not None:
            values.extend(
                (
                    proxy.basic_auth.username.get_secret_value(),
                    proxy.basic_auth.password.get_secret_value(),
                )
            )
        if proxy.bearer_token is not None:
            values.append(proxy.bearer_token.get_secret_value())
    if isinstance(settings, RuntimeSettingsV2) and settings.http_origin_target is not None:
        target = settings.http_origin_target
        if target.basic_auth is not None:
            values.extend(
                (
                    target.basic_auth.username.get_secret_value(),
                    target.basic_auth.password.get_secret_value(),
                )
            )
        if target.bearer_token is not None:
            values.append(target.bearer_token.get_secret_value())
        authorization = _target_authorization(settings)
        if authorization is not None:
            values.append(authorization)
    return tuple(value for value in values if value)


def _target_origin(settings: RuntimeSettings) -> tuple[str, str, int] | None:
    if not isinstance(settings, RuntimeSettingsV2) or settings.http_origin_target is None:
        return None
    return _origin(settings.http_origin_target.url)


def _target_authorization(settings: RuntimeSettings) -> str | None:
    if not isinstance(settings, RuntimeSettingsV2) or settings.http_origin_target is None:
        return None
    target = settings.http_origin_target
    if target.bearer_token is not None:
        return f"Bearer {target.bearer_token.get_secret_value()}"
    if target.basic_auth is not None:
        username = target.basic_auth.username.get_secret_value()
        password = target.basic_auth.password.get_secret_value()
        raw = f"{username}:{password}".encode()
        return f"Basic {base64.b64encode(raw).decode('ascii')}"
    return None


def _method(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in _METHODS:
        raise HTTPToolError("http_request_invalid")
    return value.upper()


def _url_with_query(url: object, query: Mapping[str, Any] | None) -> str:
    if not isinstance(url, str) or not 1 <= len(url.encode("utf-8")) <= MAX_URL_BYTES:
        raise HTTPToolError("http_request_invalid")
    try:
        parsed = urlsplit(url)
        existing = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
    except (UnicodeError, ValueError):
        raise HTTPToolError("http_request_invalid") from None
    additions = _query_pairs(query)
    if len(existing) + len(additions) > MAX_QUERY_KEYS:
        raise HTTPToolError("http_request_invalid")
    encoded = urlencode(existing + additions, doseq=True)
    if len(encoded.encode("utf-8")) > MAX_QUERY_BYTES:
        raise HTTPToolError("http_request_invalid")
    result = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, encoded, parsed.fragment))
    if len(result.encode("utf-8")) > MAX_URL_BYTES:
        raise HTTPToolError("http_request_invalid")
    return result


def _query_pairs(query: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    if query is None:
        return []
    if not isinstance(query, Mapping) or len(query) > MAX_QUERY_KEYS:
        raise HTTPToolError("http_request_invalid")
    result: list[tuple[str, str]] = []
    for key, value in query.items():
        if not isinstance(key, str) or not key or any(char in key for char in "\r\n\x00"):
            raise HTTPToolError("http_request_invalid")
        values = value if isinstance(value, list) else [value]
        if not values:
            result.append((key, ""))
        for item in values:
            if item is None:
                rendered = ""
            elif type(item) in {str, int, float, bool}:
                if isinstance(item, float) and not math.isfinite(item):
                    raise HTTPToolError("http_request_invalid")
                rendered = str(item).lower() if type(item) is bool else str(item)
            else:
                raise HTTPToolError("http_request_invalid")
            if len(rendered.encode("utf-8")) > MAX_HEADER_VALUE_BYTES:
                raise HTTPToolError("http_request_invalid")
            result.append((key, rendered))
    if len(result) > MAX_QUERY_KEYS:
        raise HTTPToolError("http_request_invalid")
    return result


def _headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping) or len(headers) > MAX_HEADERS:
        raise HTTPToolError("http_request_invalid")
    result: dict[str, str] = {}
    seen: set[str] = set()
    total = 0
    for name, raw_value in headers.items():
        try:
            encoded_name = name.encode("ascii") if isinstance(name, str) else b""
        except UnicodeEncodeError:
            raise HTTPToolError("http_request_invalid") from None
        if not isinstance(name, str) or _HEADER_NAME.fullmatch(encoded_name) is None:
            raise HTTPToolError("http_request_invalid")
        normalized = name.lower()
        if normalized in seen or normalized in _FORBIDDEN_REQUEST_HEADERS:
            raise HTTPToolError("http_request_invalid")
        if not isinstance(raw_value, str) or any(char in raw_value for char in "\r\n\x00"):
            raise HTTPToolError("http_request_invalid")
        size = len(raw_value.encode("utf-8"))
        if size > MAX_HEADER_VALUE_BYTES:
            raise HTTPToolError("http_request_invalid")
        total += len(name) + size
        if total > MAX_HEADER_BYTES:
            raise HTTPToolError("http_request_invalid")
        seen.add(normalized)
        result[name] = raw_value
    return result


def _merge_headers(defaults: Mapping[str, str], request: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, tuple[str, str]] = {
        name.lower(): (name, value) for name, value in defaults.items()
    }
    for name, value in request.items():
        result[name.lower()] = (name, value)
    if len(result) > MAX_HEADERS:
        raise HTTPToolError("http_request_invalid")
    if (
        sum(
            len(name.encode("ascii")) + len(value.encode("utf-8"))
            for name, value in result.values()
        )
        > MAX_HEADER_BYTES
    ):
        raise HTTPToolError("http_request_invalid")
    return {name: value for name, value in result.values()}


def _request_body(body_type: object, body: Any) -> tuple[bytes, str | None]:
    if not isinstance(body_type, str) or body_type not in _BODY_TYPES:
        raise HTTPToolError("http_request_invalid")
    try:
        if body_type == "none":
            if body is not None:
                raise HTTPToolError("http_request_invalid")
            return b"", None
        if body_type == "json":
            encoded = json.dumps(
                body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            media_type = "application/json"
        elif body_type == "form":
            encoded = urlencode(_query_pairs(body), doseq=True).encode("utf-8")
            media_type = "application/x-www-form-urlencoded"
        else:
            if not isinstance(body, str):
                raise HTTPToolError("http_request_invalid")
            encoded = body.encode("utf-8")
            media_type = "text/plain; charset=utf-8"
    except HTTPToolError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise HTTPToolError("http_request_invalid") from None
    if len(encoded) > MAX_REQUEST_BODY_BYTES:
        raise HTTPToolError("http_request_invalid")
    return encoded, media_type


def _timeout(value: object, cap: int) -> float:
    selected: float
    if value is None:
        selected = min(float(cap), 120.0)
    elif type(value) in {int, float}:
        selected = float(value)
    else:
        raise HTTPToolError("http_request_invalid")
    if not math.isfinite(selected) or selected < 1 or selected > 120:
        raise HTTPToolError("http_request_invalid")
    return min(selected, float(cap))


def _validate_target(url: str, forbidden: frozenset[tuple[str, str, int]]) -> None:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise HTTPToolError("http_request_invalid")
        selected_origin = _origin(url)
    except HTTPToolError:
        raise
    except (UnicodeError, ValueError):
        raise HTTPToolError("http_request_invalid") from None
    host = selected_origin[1]
    denied = selected_origin in forbidden or host == "localhost" or host.endswith(".localhost")
    try:
        address = ipaddress.ip_address(host)
        denied = denied or address.is_loopback or address.is_link_local or address.is_unspecified
    except ValueError:
        pass
    if denied:
        raise HTTPToolError("http_target_denied")


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    port = parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
    return scheme, host, port


async def _read_response_body(response: httpx.Response) -> bytes:
    result = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(result) + len(chunk) > MAX_RESPONSE_BODY_BYTES:
                raise HTTPToolError("http_response_too_large")
            result.extend(chunk)
    except HTTPToolError:
        raise
    except (httpx.HTTPError, UnicodeError):
        raise HTTPToolError("http_request_failed") from None
    return bytes(result)


def _content_type(headers: httpx.Headers) -> str:
    value = headers.get("content-type", "application/octet-stream").strip()
    if not value or len(value.encode("utf-8")) > 256 or any(char in value for char in "\r\n\x00"):
        return "application/octet-stream"
    return value


def _encode_body(
    content_type: str, body: bytes
) -> tuple[Literal["empty", "text", "binary"], str | None, dict[str, Any] | None]:
    if not body:
        return "empty", "", None
    normalized = content_type.lower().split(";", 1)[0].strip()
    if normalized.startswith(_TEXTUAL_CONTENT_TYPES) or normalized.endswith("+json"):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return (
                "text",
                text[:MAX_PREVIEW_CHARACTERS],
                {
                    "schemaVersion": BODY_SCHEMA_VERSION,
                    "kind": "text",
                    "contentType": content_type,
                    "text": text,
                },
            )
    return (
        "binary",
        None,
        {
            "schemaVersion": BODY_SCHEMA_VERSION,
            "kind": "binary",
            "contentType": content_type,
            "dataBase64": base64.b64encode(body).decode("ascii"),
        },
    )


def _decode_body_artifact(payload: bytes) -> tuple[Literal["text", "binary"], str | bytes]:
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise HTTPToolError("http_body_not_found")
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeError):
        raise HTTPToolError("http_body_not_found") from None
    if not isinstance(value, dict) or value.get("schemaVersion") != BODY_SCHEMA_VERSION:
        raise HTTPToolError("http_body_not_found")
    if value.get("kind") == "text" and set(value) == {
        "schemaVersion",
        "kind",
        "contentType",
        "text",
    }:
        text = value.get("text")
        if isinstance(text, str):
            return "text", text
    if value.get("kind") == "binary" and set(value) == {
        "schemaVersion",
        "kind",
        "contentType",
        "dataBase64",
    }:
        encoded = value.get("dataBase64")
        if isinstance(encoded, str):
            try:
                return "binary", base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                pass
    raise HTTPToolError("http_body_not_found")


def _response_headers(headers: httpx.Headers) -> tuple[dict[str, str], bool]:
    result: dict[str, str] = {}
    total = 0
    truncated = False
    for name, value in headers.multi_items():
        normalized = name.lower()
        if _is_sensitive_header(normalized) or normalized.startswith("proxy-"):
            continue
        size = len(name.encode("utf-8")) + len(value.encode("utf-8"))
        if (
            len(result) >= MAX_HEADERS
            or len(value.encode("utf-8")) > MAX_HEADER_VALUE_BYTES
            or total + size > MAX_HEADER_BYTES
        ):
            truncated = True
            continue
        total += size
        result[name] = value
    return result, truncated


def _cookie_values(cookies: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(cookies, Mapping) or len(cookies) > MAX_COOKIES:
        raise HTTPToolError("http_request_invalid")
    result: dict[str, str] = {}
    for name, value in cookies.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name.encode("utf-8")) > 256
            or any(char in name for char in "\r\n\x00;,")
            or not isinstance(value, str)
            or len(value.encode("utf-8")) > MAX_HEADER_VALUE_BYTES
            or any(char in value for char in "\r\n\x00")
        ):
            raise HTTPToolError("http_request_invalid")
        result[name] = value
    return result


def _auth(
    auth: Mapping[str, Any] | None,
) -> tuple[Literal["none", "basic", "bearer"], str | None, str | None] | None:
    if auth is None:
        return None
    if not isinstance(auth, Mapping) or len(auth) > 3:
        raise HTTPToolError("http_request_invalid")
    kind = auth.get("kind")
    if kind == "none":
        if set(auth) != {"kind"}:
            raise HTTPToolError("http_request_invalid")
        return "none", None, None
    if kind == "basic":
        username = auth.get("username")
        password = auth.get("password")
        if (
            set(auth) != {"kind", "username", "password"}
            or not isinstance(username, str)
            or not isinstance(password, str)
            or not 1 <= len(username.encode("utf-8")) <= 256
            or not 1 <= len(password.encode("utf-8")) <= MAX_HEADER_VALUE_BYTES
            or any(char in username + password for char in "\r\n\x00")
        ):
            raise HTTPToolError("http_request_invalid")
        return "basic", username, password
    if kind == "bearer":
        bearer_token = auth.get("token")
        if (
            set(auth) != {"kind", "token"}
            or not isinstance(bearer_token, str)
            or not 1 <= len(bearer_token.encode("utf-8")) <= MAX_HEADER_VALUE_BYTES
            or any(char in bearer_token for char in "\r\n\x00")
        ):
            raise HTTPToolError("http_request_invalid")
        return "bearer", None, bearer_token
    raise HTTPToolError("http_request_invalid")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)


def _is_sensitive_header(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized in _SENSITIVE_HEADERS or any(
        marker in normalized for marker in ("api-key", "auth-token", "access-token")
    )


class _UnavailableArtifactTransport:
    async def request(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ArtifactTransportError("Artifact transport is not configured")


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableArtifactTransport())
