"""Bounded allocation-scoped HTTP exploration tools."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import secrets
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlsplit

import httpx
import idna
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor_runtime import threads
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import (
    ProxyHTTPClient,
    ProxyRequestError,
    ProxyTargetDenied,
)
from contractor_runtime.adapters.otlp_retry import retry_after_seconds
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.contracts.fields import is_field_value
from contractor_runtime.core.http_limits import (
    MAX_ATTEMPTS,
    MAX_COOKIES,
    MAX_HEADER_VALUE_BYTES,
    MAX_HEADERS,
    MAX_HISTORY,
    MAX_PREVIEW_CHARACTERS,
    MAX_QUERY_BYTES,
    MAX_QUERY_KEYS,
    MAX_READ_UNITS,
    MAX_REDIRECTS,
    MAX_URL_BYTES,
    header_block_bytes,
)
from contractor_runtime.core.http_session import (
    discard_expired_cookies,
    extract_sendable_cookies,
    matching_cookies,
)
from contractor_runtime.core.target_policy import (
    TargetDenied,
    TargetPolicy,
    TargetPolicyConfig,
    validate_host_syntax,
)
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics, elapsed_ms
from contractor_runtime.http_body import (
    BOUNDED_ACCEPT_ENCODING,
    HTTP_TRANSPORT_ERRORS,
    BodyTooLarge,
    read_limited_body,
)
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _reject_unconfigured_client,
)
from contractor_runtime.toolsets.common.credentials import target_authorization
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.finding_locations import validate_web_url
from contractor_runtime.toolsets.common.http_capture import (
    HTTPObservation,
    RecentHTTPCaptures,
    allocation_http_captures,
)
from contractor_runtime.toolsets.common.http_evidence import HTTPAttempt
from contractor_runtime.toolsets.common.http_observations import (
    HTTPObservationSink,
    notify_http_observations,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.http.arguments import request_containers
from contractor_runtime.toolsets.http.capture import CapturedAttempt, CapturedExchange
from contractor_runtime.toolsets.http.cookies import (
    CookieCheckpoint,
)
from contractor_runtime.toolsets.http.errors import HTTPToolError
from contractor_runtime.toolsets.http.fields import is_cookie_pair
from contractor_runtime.toolsets.http.headers import (
    body_headers,
    merge_headers,
    request_utf8_size,
    validate_headers,
)
from contractor_runtime.toolsets.http.transport import PolicyHTTPTransport
from contractor_runtime.workspace import AllocationWorkspace


# A 16 MiB UTF-8 response can occupy almost 64 MiB as a Python str when a
# single non-BMP character forces four-byte storage for all characters.
def decoded_body_cache_max_bytes() -> int:
    return 4 * get_shared_config().http_response_body_max_bytes + 128


_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "DELETE", "HEAD", "OPTIONS"})
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# The first retry waits this long and each later one twice as long. A target's
# Retry-After replaces a shorter wait, up to the cap; a longer Retry-After or a
# wait that would leave less than half the remaining deadline returns the last
# response (or failure) instead of retrying.
RETRY_BACKOFF_SECONDS = 0.25
MAX_RETRY_AFTER_SECONDS = 5.0
_retry_sleep = asyncio.sleep
_BODY_TYPES = frozenset({"none", "json", "form", "text"})
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization", "proxy-connection", "set-cookie"}
)
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
            "http_credentials",
            "http_credential_read",
        }
    )
    infrastructure_channels = MappingProxyType({"http_request": frozenset({"runtime-http-client"})})
    # Every tool's result carries captured HTTP traffic or HTTP session
    # material, including tools added later.

    def __init__(
        self,
        direct_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        *,
        target_policy: TargetPolicyConfig | None = None,
        artifact_client_factory: ArtifactClientFactory = _reject_unconfigured_client,
        now: Callable[[], float] = time.time,
    ) -> None:
        # Tests may inject a fixed client; production clients enforce the
        # allocation's target policy on every TCP connection.
        self._direct_client_factory = direct_client_factory
        self._target_policy = target_policy or TargetPolicyConfig()
        self._client_factory = artifact_client_factory
        self._now = now

    async def probe(self) -> frozenset[str]:
        client: httpx.AsyncClient | None = None
        try:
            client = self._new_direct_client(TargetPolicy())
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
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
        preparation_deadline: float | None = None,
        deployment_marker: str | None = None,
        controlled_http: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, project_workspace
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(resources, "http-tools@1")

        if controlled_http is not None:
            from contractor_runtime.toolsets.http.controlled import ControlledHTTPSession

            session = ControlledHTTPSession(
                controlled_http,
                runtime_settings.request_timeout_seconds,
                captures=allocation_http_captures(resources),
                deployment_marker=deployment_marker,
            )
            return _selected_http_tools(selected, session, metrics)

        proxy = adapter_handles.tool_http
        request_selected = "http_request" in selected
        proxy_required = request_selected and _tool_http_proxy_required(runtime_settings)
        if proxy is not None and not isinstance(proxy, ProxyHTTPClient):
            raise TypeError("http-tools@1 received an invalid runtime-http-client handle")
        if proxy_required and proxy is None:
            raise RuntimeError("http-tools@1 requires the resolved tool-http proxy route")

        checkpoint = CookieCheckpoint(
            self._client_factory(allocation_id, runtime_settings),
            namespace,
            _target_origin(runtime_settings),
            self._now,
            runtime_settings.request_timeout_seconds,
        )
        policy = await self._target_policy.build(runtime_settings)
        direct_client = (
            self._new_direct_client(policy) if request_selected and proxy is None else None
        )
        session = _HTTPSession(
            timeout_cap_seconds=runtime_settings.request_timeout_seconds,
            target_policy=policy,
            secrets_for_metrics=_http_metric_secrets(runtime_settings),
            proxy=proxy,
            direct_client=direct_client,
            target_origin=_target_origin(runtime_settings),
            target_authorization=target_authorization(runtime_settings),
            captures=allocation_http_captures(resources),
            checkpoint=checkpoint,
            now=self._now,
            observation_sink=resources.http_observation_sink,
            deployment_marker=deployment_marker,
        )
        try:
            await session.restore(deadline=preparation_deadline)
        except BaseException:
            await session.close()
            raise
        return _selected_http_tools(selected, session, metrics)

    def _new_direct_client(self, policy: TargetPolicy) -> httpx.AsyncClient:
        if self._direct_client_factory is not None:
            return self._direct_client_factory()
        return _direct_client(policy)


@dataclass(frozen=True, slots=True)
class _CachedBody:
    request_id: int
    kind: Literal["text", "binary"]
    content: str | bytes
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _RequestRecord:
    request_id: int
    tag: str
    method: str
    final_url: str
    status: int | None
    content_type: str
    body_kind: Literal["empty", "text", "binary"]
    body_bytes: int
    redirects: int
    retries: int
    elapsed_ms: int
    preview: str | None
    headers: dict[str, str] = field(repr=False)
    headers_truncated: bool = False
    retained_body_truncated: bool = False
    error: str | None = None

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
            "body_truncated": self.retained_body_truncated,
            "body_kind": self.body_kind,
            "redirects": self.redirects,
            "retries": self.retries,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.status is None:
            result.pop("status")
        if self.error is not None:
            result["error"] = self.error
        if include_preview:
            result["body_preview"] = self.preview
            result["body_truncated"] = self.retained_body_truncated or (
                self.body_kind == "text"
                and self.preview is not None
                and len(self.preview) >= MAX_PREVIEW_CHARACTERS
            )
        return result


class _HTTPSession:
    def credential_catalog(self):
        return []

    def credential_values(self, credential_id):
        raise HTTPToolError("http_credential_unavailable")

    def __init__(
        self,
        *,
        timeout_cap_seconds: int,
        target_policy: TargetPolicy,
        secrets_for_metrics: tuple[str, ...],
        proxy: ProxyHTTPClient | None,
        direct_client: httpx.AsyncClient | None,
        target_origin: tuple[str, str, int] | None,
        target_authorization: str | None,
        captures: RecentHTTPCaptures,
        checkpoint: CookieCheckpoint,
        now: Callable[[], float],
        observation_sink: HTTPObservationSink | None = None,
        deployment_marker: str | None = None,
    ) -> None:
        self._timeout_cap_seconds = timeout_cap_seconds
        self._target_policy = target_policy
        self._secrets_for_metrics = secrets_for_metrics
        self._proxy = proxy
        self._direct_client = direct_client
        self._target_origin = target_origin
        self._target_authorization = target_authorization
        self._lock = asyncio.Lock()
        self._history: deque[_RequestRecord] = deque(maxlen=MAX_HISTORY)
        self._cached_body: _CachedBody | None = None
        self._captures = captures
        self._default_headers: dict[str, str] = {}
        self._cookies = httpx.Cookies()
        self._auth_kind: Literal["none", "basic", "bearer"] = "none"
        self._auth_username: str | None = None
        self._auth_secret: str | None = None
        self._nonce = secrets.token_hex(8)
        self._closed = False
        self._checkpoint = checkpoint
        self._now = now
        self._observation_sink = observation_sink
        self._deployment_marker = deployment_marker

    async def restore(self, *, deadline: float | None = None) -> None:
        self._cookies = await self._checkpoint.restore(deadline=deadline)

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
        invocation_id: str | None = None,
        credential_id: str | None = None,
    ) -> dict[str, Any]:
        if credential_id is not None:
            raise HTTPToolError("http_credential_unavailable")
        async with self._lock:
            self._require_open()
            self._checkpoint.require_confirmed()
            try:
                request_id = self._captures.new_http_request()
            except ValueError:
                raise HTTPToolError("http_request_invalid") from None
            started_ns = time.perf_counter_ns()
            selected_method = _method(method)
            selected_url = _url_with_query(url, query)
            selected_headers = validate_headers(headers)
            payload, content_type = _request_body(body_type, body)
            selected_timeout = _timeout(timeout_seconds, self._timeout_cap_seconds)
            if type(follow_redirects) is not bool:
                raise HTTPToolError("http_request_invalid")

            merged_headers = body_headers(self._default_headers, selected_headers, content_type)
            merged_headers["X-Request-Id"] = f"r{self._nonce}-h{request_id:06d}"
            tag = merged_headers["X-Request-Id"]
            captured = CapturedExchange(request_id, tag)
            # httpx keeps its own cookie jar in addition to the allocation
            # session jar.  Treat that transport jar as scratch space: stale
            # cookies left by a cancelled/failed prior call must never affect
            # the next request.
            self._clear_transport_cookies()
            # The tool timeout is one deadline for every hop, retry and the
            # complete body read, so a trickling target cannot hold the session
            # lock indefinitely. Finding publication has its own deadline.
            deadline = asyncio.get_running_loop().time() + selected_timeout
            published = False
            try:
                async with asyncio.timeout_at(deadline):
                    (
                        response,
                        final_method,
                        final_url,
                        redirects,
                        retries,
                        candidate_cookies,
                    ) = await self._send_following(
                        method=selected_method,
                        url=selected_url,
                        headers=merged_headers,
                        payload=payload,
                        timeout_seconds=selected_timeout,
                        deadline=deadline,
                        follow_redirects=follow_redirects,
                        attempts=captured.attempts,
                    )
                    try:
                        body_bytes = await _read_response_body(response)
                    finally:
                        await response.aclose()
                content_type = _content_type(response.headers)
                body_kind, preview, _ = await threads.to_thread_until_done(
                    _encode_body, content_type, body_bytes, name="http-body-preview"
                )
                safe_headers, headers_truncated = _response_headers(response.headers)
                captured.complete = True
                captured.response_body = body_bytes
                evidence = await threads.to_thread_until_done(
                    captured.evidence, name="http-exchange-capture"
                )
                try:
                    await self._captures.put(
                        evidence,
                        invocation_id,
                        observation=HTTPObservation(
                            captured.attempts[0].at, self._deployment_marker
                        ),
                    )
                except ValueError:
                    # A deterministic capture rejection is permanent, exactly
                    # as for a failed exchange below.
                    raise HTTPToolError("http_response_invalid") from None
                published = True
                self._cookies = candidate_cookies
                checkpoint_error = None
                try:
                    await self._save_cookies(candidate_cookies, deadline=deadline)
                except HTTPToolError as error:
                    # The target already completed. Keep its actual outcome;
                    # retrying a POST to repair Artifact storage is unsafe.
                    self._checkpoint.reject(error)
                    checkpoint_error = error.code
                record = _RequestRecord(
                    request_id=request_id,
                    tag=tag,
                    method=final_method,
                    final_url=final_url,
                    status=response.status_code,
                    content_type=content_type,
                    body_kind=body_kind,
                    body_bytes=len(body_bytes),
                    redirects=redirects,
                    retries=retries,
                    elapsed_ms=elapsed_ms(started_ns),
                    preview=preview,
                    headers=safe_headers,
                    headers_truncated=headers_truncated,
                )
                self._history.append(record)
                result = record.projection(include_preview=True)
                if checkpoint_error is not None:
                    result["cookie_checkpoint"] = {
                        "state": "unconfirmed",
                        "code": checkpoint_error,
                        "recovery": (
                            "Do not repeat this HTTP request. Call http_session_get if selected; "
                            "otherwise report unconfirmed cookie storage and stop HTTP requests."
                        ),
                    }
                return result
            except TimeoutError:
                captured.error = "timeout"
                raise HTTPToolError("http_request_failed") from None
            except asyncio.CancelledError:
                captured.complete = False
                captured.error = "cancelled"
                raise
            except HTTPToolError as error:
                captured.complete = False
                captured.error = {
                    "http_response_invalid": "response_invalid",
                    "http_response_too_large": "response_too_large",
                    "http_target_denied": "target_denied",
                    "http_request_invalid": "request_invalid",
                }.get(error.code, "transport_error")
                raise
            except HTTP_TRANSPORT_ERRORS:
                # A transport failure not already classified per hop, such as
                # closing a response stream.
                captured.complete = False
                captured.error = "transport_error"
                raise HTTPToolError("http_request_failed") from None
            except Exception:
                # An implementation failure keeps its type for the Worker's
                # tool-error handling; the retained exchange is still explicit
                # that no complete response was delivered.
                captured.complete = False
                captured.error = "transport_error"
                raise
            finally:
                if self._observation_sink is not None and captured.attempts:
                    observations = []
                    for index, attempt in enumerate(captured.attempts):
                        error = (
                            captured.error if index == len(captured.attempts) - 1 else attempt.error
                        )
                        status = (
                            attempt.status
                            if attempt.status is not None and 100 <= attempt.status <= 599
                            else 0
                        )
                        outcome = "response" if status else "transport-error"
                        if attempt.timed_out or (error == "timeout" and not status):
                            outcome, status = "timeout", 0
                        elif error == "cancelled" and not status:
                            outcome = "attempt"
                        observations.append(
                            {
                                "url": attempt.url,
                                "method": attempt.method,
                                "outcome": outcome,
                                "status": status,
                                "at": attempt.at,
                            }
                        )
                    await notify_http_observations(self._observation_sink, observations)
                if captured.attempts and not published:
                    if not captured.complete:
                        last = captured.attempts[-1]
                        self._history.append(
                            _RequestRecord(
                                request_id=request_id,
                                tag=tag,
                                method=last.method,
                                final_url=last.url,
                                status=last.status
                                if last.status is not None and 100 <= last.status <= 599
                                else None,
                                content_type="",
                                body_kind="empty",
                                body_bytes=0,
                                redirects=0,
                                retries=max(0, len(captured.attempts) - 1),
                                elapsed_ms=elapsed_ms(started_ns),
                                preview=None,
                                headers={},
                                error=captured.error or "transport_error",
                            )
                        )
                    evidence = await threads.to_thread_until_done(
                        captured.evidence, name="http-exchange-capture"
                    )
                    try:
                        await self._captures.put(
                            evidence,
                            invocation_id,
                            observation=HTTPObservation(
                                captured.attempts[0].at, self._deployment_marker
                            ),
                        )
                    except ValueError:
                        raise HTTPToolError("http_response_invalid") from None

    async def _send_following(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: bytes,
        timeout_seconds: float,
        deadline: float,
        follow_redirects: bool,
        attempts: list[CapturedAttempt],
    ) -> tuple[httpx.Response, str, str, int, int, httpx.Cookies]:
        # Commit this prospective jar only after the complete response is buffered.
        discard_expired_cookies(self._cookies, self._now())
        candidate_cookies = httpx.Cookies(self._cookies)
        redirects = 0
        retries = 0
        current_method = method
        current_url = url
        current_headers = headers
        current_payload = payload
        original_origin: tuple[str, str, int] | None = None
        while True:
            _validate_target(current_url, self._target_policy)
            if original_origin is None:
                # Derived only after validation: an unusable port is invalid
                # input (http_request_invalid), never a transport failure.
                original_origin = _origin(current_url)
            # Each hop is compared with the requested origin, not the previous
            # hop: a chain that leaves and returns regains the caller's
            # credentials only for the original origin.
            same_origin = _origin(current_url) == original_origin
            try:
                response = await self._send_once(
                    current_method,
                    current_url,
                    headers=(
                        current_headers if same_origin else _without_sensitive(current_headers)
                    ),
                    content=current_payload,
                    timeout=timeout_seconds,
                    allow_session_auth=same_origin,
                    allow_session_cookies=same_origin,
                    request_cookies=candidate_cookies,
                    capture_attempts=attempts,
                )
            except asyncio.CancelledError:
                raise
            except (ProxyRequestError, *HTTP_TRANSPORT_ERRORS):
                if current_method in _IDEMPOTENT_METHODS and retries + 1 < MAX_ATTEMPTS:
                    delay = _retry_delay(retries + 1, None, deadline)
                    if delay is not None:
                        retries += 1
                        await _retry_sleep(delay)
                        continue
                raise HTTPToolError("http_request_failed") from None

            handoff = False
            try:
                extract_sendable_cookies(candidate_cookies, response)
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and current_method in _IDEMPOTENT_METHODS
                    and retries + 1 < MAX_ATTEMPTS
                ):
                    delay = _retry_delay(
                        retries + 1,
                        retry_after_seconds(response.headers.get("retry-after")),
                        deadline,
                    )
                    if delay is not None:
                        retries += 1
                        # Release the connection before waiting.
                        await response.aclose()
                        await _retry_sleep(delay)
                        continue

                location = response.headers.get("location")
                if not (
                    follow_redirects
                    and location
                    and response.status_code in {301, 302, 303, 307, 308}
                ):
                    handoff = True
                    return (
                        response,
                        current_method,
                        str(response.url),
                        redirects,
                        retries,
                        candidate_cookies,
                    )
                if redirects >= MAX_REDIRECTS:
                    raise HTTPToolError("http_request_failed")

                try:
                    next_url = urldefrag(urljoin(current_url, location)).url
                except ValueError:
                    raise HTTPToolError("http_request_invalid") from None
                _validate_target(next_url, self._target_policy)
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
                current_url = next_url
                current_method = next_method
                current_payload = next_payload
                current_headers = next_headers
                redirects += 1
            finally:
                if not handoff:
                    await response.aclose()

    async def _send_once(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempts = kwargs.pop("capture_attempts")
        attempt: CapturedAttempt | None = None

        headers = dict(kwargs.pop("headers"))
        allow_session_auth = bool(kwargs.pop("allow_session_auth", True))
        allow_session_cookies = bool(kwargs.pop("allow_session_cookies", True))
        request_cookies = kwargs.pop("request_cookies")
        # httpx merges its own jar into build_request even when cookies is omitted.
        # Never let a redirect/retry response bypass the allocation cookie policy.
        self._clear_transport_cookies()
        target_credential = (
            self._target_origin is not None
            and self._target_authorization is not None
            and _origin(url) == self._target_origin
        )
        if target_credential:
            if not (self._target_authorization is not None):
                raise RuntimeError("Expected self._target_authorization is not None")
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
            kwargs["cookies"] = matching_cookies(request_cookies, url)

        def observe(request: httpx.Request) -> None:
            nonlocal attempt
            try:
                validate_web_url(str(request.url))
            except ValueError:
                raise HTTPToolError("http_request_invalid") from None
            captured = CapturedAttempt.from_request(request, target_credential=target_credential)
            if (
                header_block_bytes(captured.header_pairs())
                > get_shared_config().http_header_block_max_bytes
            ):
                # Session cookies or auth grew the block past what finding
                # evidence retains; refuse before anything is sent.
                raise HTTPToolError("http_request_invalid")
            attempt = captured
            attempts.append(attempt)

        try:
            if self._proxy is not None:
                response = await self._proxy.stream_request(
                    method,
                    url,
                    target_policy=self._target_policy,
                    request_observer=observe,
                    **kwargs,
                )
            else:
                client = self._direct_client
                if client is None:
                    raise HTTPToolError("http_request_failed")
                request = client.build_request(method, url, **kwargs)
                observe(request)
                response = await client.send(request, stream=True, follow_redirects=False)
            if not (attempt is not None):
                raise RuntimeError("Expected attempt is not None")
            attempt.receive(response)
            try:
                HTTPAttempt.model_validate(attempt.snapshot())
            except ValidationError:
                await response.aclose()
                raise HTTPToolError("http_response_invalid") from None
            return response
        except asyncio.CancelledError:
            if attempt is not None:
                attempt.error = "cancelled"
            raise
        except HTTPToolError:
            raise
        except (httpx.InvalidURL, idna.IDNAError):
            raise HTTPToolError("http_request_invalid") from None
        except UnicodeEncodeError:
            if attempt is not None:
                attempt.error = "transport_error"
                raise
            # httpx cannot encode a non-ASCII header value, including a session
            # cookie, into the request: invalid input on both routes.
            raise HTTPToolError("http_request_invalid") from None
        except (TargetDenied, ProxyTargetDenied):
            if attempt is not None and attempt.status is None:
                attempts.remove(attempt)
            # Connect-time and proxy-route denials share the pre-send code.
            raise HTTPToolError("http_target_denied") from None
        except (httpx.TimeoutException, ProxyRequestError) as error:
            if attempt is not None:
                attempt.error = "transport_error"
                attempt.timed_out = isinstance(error, httpx.TimeoutException) or error.timed_out
            raise
        except Exception:
            # Transport failures (HTTP_TRANSPORT_ERRORS, the proxy route's
            # failure) are classified and retried by the caller. Any other
            # exception is an implementation failure and keeps its type; either
            # way the retained attempt has no complete response.
            if attempt is not None:
                attempt.error = "transport_error"
            raise

    async def read_body(self, request_id: int, offset: int, limit: int) -> dict[str, Any]:
        if type(request_id) is not int or request_id <= 0:
            raise HTTPToolError("http_request_invalid")
        if type(offset) is not int or offset < 0:
            raise HTTPToolError("http_request_invalid")
        if type(limit) is not int or not 1 <= limit <= MAX_READ_UNITS:
            raise HTTPToolError("http_request_invalid")
        async with self._lock:
            self._require_open()
            if not self._captures.contains("http", str(request_id)):
                raise HTTPToolError("http_body_not_found")
            # A controlled transport may retain a bounded prefix. Its private
            # record permits inspection without claiming a complete response.
            truncated = any(
                record.request_id == request_id and record.retained_body_truncated
                for record in self._history
            )
            cached = self._cached_body
            if cached is not None and cached.request_id == request_id:
                kind, content, truncated = cached.kind, cached.content, cached.truncated
            else:
                self._cached_body = None
                try:
                    capture = await self._captures.get("http", str(request_id))
                    response = capture["attempts"][-1].get("response")
                    if response is None or (not response["complete"] and not truncated):
                        raise HTTPToolError("http_body_not_found")
                    message = response["message"]
                    body_bytes = await threads.to_thread_until_done(
                        lambda: base64.b64decode(message["body_base64"], validate=True),
                        name="http-body-base64",
                    )
                    if not body_bytes:
                        raise HTTPToolError("http_body_not_found")
                    headers = {row["name"].lower(): row["value"] for row in message["headers"]}
                    body_kind, _, decoded = await threads.to_thread_until_done(
                        _encode_body,
                        headers.get("content-type", ""),
                        body_bytes,
                        name="http-body-decode",
                    )
                    kind = body_kind
                    content = decoded if kind == "text" else body_bytes
                except ValueError:
                    # An evicted, foreign or undecodable buffered capture. Any
                    # other exception is an implementation failure.
                    raise HTTPToolError("http_body_not_found") from None
                if sys.getsizeof(content) <= decoded_body_cache_max_bytes():
                    self._cached_body = _CachedBody(request_id, kind, content, truncated)
            if kind == "text":
                if not (isinstance(content, str)):
                    raise RuntimeError("Expected isinstance(content, str)")
                selected = content[offset : offset + limit]
                return {
                    "request_id": request_id,
                    "kind": "text",
                    "unit": "characters",
                    "offset": offset,
                    "length": len(selected),
                    "total": len(content),
                    "eof": offset + len(selected) >= len(content),
                    "body_truncated": truncated,
                    "data": selected,
                }
            if not (isinstance(content, bytes)):
                raise RuntimeError("Expected isinstance(content, bytes)")
            selected_bytes = content[offset : offset + limit]
            return {
                "request_id": request_id,
                "kind": "binary",
                "unit": "bytes",
                "offset": offset,
                "length": len(selected_bytes),
                "total": len(content),
                "eof": offset + len(selected_bytes) >= len(content),
                "body_truncated": truncated,
                "data_b64": base64.b64encode(selected_bytes).decode("ascii"),
            }

    async def history(self, limit: int) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY:
            raise HTTPToolError("http_request_invalid")
        async with self._lock:
            self._require_open()
            selected = [
                record
                for record in self._history
                if self._captures.contains("http", str(record.request_id))
            ][-limit:]
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
        selected_headers = None if headers is None else validate_headers(headers)
        selected_cookies = None if cookies is None else _cookie_values(cookies)
        selected_auth = _auth(auth)
        async with self._lock:
            self._require_open()
            self._checkpoint.require_confirmed()
            candidate_headers = self._default_headers
            if selected_headers is not None:
                candidate_headers = merge_headers(
                    {} if replace_headers else self._default_headers,
                    selected_headers,
                )
            candidate_cookies = self._cookies
            if selected_cookies is not None:
                discard_expired_cookies(self._cookies, self._now())
                candidate_cookies = httpx.Cookies()
                if not replace_cookies:
                    candidate_cookies.update(self._cookies)
                for name, value in selected_cookies.items():
                    if self._target_origin is None:
                        candidate_cookies.set(name, value)
                    else:
                        host = self._target_origin[1]
                        candidate_cookies.set(name, value, domain=host)
                        # Explicit Project cookies have an exact host scope.
                        for cookie in candidate_cookies.jar:
                            if (cookie.name, cookie.domain, cookie.path) == (name, host, "/"):
                                cookie.domain_specified = False
                if len(candidate_cookies) > MAX_COOKIES:
                    raise HTTPToolError("http_request_invalid")
            if selected_cookies is not None:
                await self._save_cookies(candidate_cookies)
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
            if self._checkpoint.unconfirmed:
                try:
                    self._cookies = await self._checkpoint.recover()
                finally:
                    if not self._checkpoint.unconfirmed:
                        self._cookies = self._checkpoint.confirmed_cookies()
            return self._session_projection()

    async def clear_session(self) -> dict[str, Any]:
        async with self._lock:
            self._require_open()
            self._checkpoint.require_confirmed()
            try:
                await self._save_cookies(httpx.Cookies(), force=True)
            except asyncio.CancelledError:
                if not self._checkpoint.unconfirmed and not len(self._cookies):
                    self._clear_mutable_session()
                raise
            self._clear_mutable_session()
            return self._session_projection()

    async def _save_cookies(
        self, cookies: httpx.Cookies, *, deadline: float | None = None, force: bool = False
    ) -> None:
        try:
            await self._checkpoint.save(cookies, deadline=deadline, force=force)
        finally:
            # A cancelled waiter may have joined a successful PUT. Mirror that
            # accepted revision before releasing the session lock; otherwise a
            # later request could resurrect a durably cleared cookie jar.
            if not self._checkpoint.unconfirmed:
                self._cookies = self._checkpoint.confirmed_cookies()

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
            self._checkpoint.close()
            self._observation_sink = None
            self._cached_body = None
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
        discard_expired_cookies(self._cookies, self._now())
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
            "history_count": sum(
                self._captures.contains("http", str(record.request_id)) for record in self._history
            ),
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
        self._captures.clear("http")
        self._cached_body = None
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
        if self.name == "http_request":
            self.__doc__ += getattr(session, "scope_description", "")

    async def close(self) -> None:
        await self._session.close()

    def _recorded(
        self, arguments: Mapping[str, Any], fallback: str = "http_request_failed"
    ) -> RecordedToolCall:
        # HTTPToolError carries a stable code, and a transport-family failure
        # outside a request (a detached proxy route) keeps the tool's fallback
        # code. Any other exception is an implementation failure: it keeps its
        # type for the Worker's tool-error handling, which answers with a
        # bounded non-retryable failure (OBS-R32).
        return RecordedToolCall(
            self._metrics,
            self.name,
            arguments,
            secrets=self._session.metric_secrets,
            bound_error=lambda error: (
                HTTPToolError(fallback)
                if isinstance(error, (ProxyRequestError, *HTTP_TRANSPORT_ERRORS))
                else error
            ),
        )


class HTTPRequestTool(_HTTPTool):
    name = "http_request"
    description = """Send an HTTP/HTTPS request and retain its exchange in the recent buffer.

    Only a bounded text preview is returned inline. Use http_read_body to inspect
    more text or binary content. Ordinary HTTP exchanges can be saved with finding.
    Controlled Audit HTTP keeps a private response buffer; it does not create trusted
    engagement proof. Configured allocation limits may tighten the common HTTP buffer bounds.
    Audit calls use only explicit credential_id, never Project target credentials.
    Controlled HTTP failures report DNS, TLS, connection or timeout reasons. If a request
    may have been sent, report the unknown result without repeating the request.
    Audit cookie state is allocation-only, without artifact checkpoints.
    Request bodies are limited to 1 MiB and response
    bodies to 16 MiB. Target cookies persist within the same Run and Namespace.
    If cookie_checkpoint is unconfirmed, keep this HTTP outcome and call
    http_session_get to repair storage when selected; otherwise report the
    unconfirmed checkpoint. Do not repeat the HTTP request.

    Metadata and Runtime service destinations fail with http_target_denied, as
    do loopback and link-local ones unless they are the project target or an
    operator-allowed network. Redirects to another origin do not carry session
    credentials or secret headers.

    Args:
        url: Absolute HTTP or HTTPS URL; a #fragment is removed before sending.
        method: GET, POST, PUT, PATCH, DELETE, HEAD or OPTIONS; defaults to GET.
        headers: Object of string headers, native or serialized once as JSON,
            merged over session defaults; both
            together at most 48 KiB. The complete header block sent, including
            session cookies and auth, must stay within 64 KiB.
        query: Query parameter object, native or serialized once as JSON.
        body: Payload shaped according to body_type.
        body_type: "none" for no body, "json" for JSON data, "form" for a field
            mapping (native or serialized once as JSON), or "text" for a string;
            defaults to "none". Nested values are not converted.
        timeout: Overall deadline in seconds for the whole call (every redirect,
            retry and the complete response body), within the configured runtime
            cap; omit to use that cap.
        follow_redirects: Follow redirects up to the request limit; defaults to true.
        credential_id: Optional exact credential from http_credentials for this
            Audit request. Omit to avoid applying an Audit credential. It does
            not select a credential by URL or change the default session.

    Returns:
        Request ID, response status and headers, body kind and preview, timing,
        redirect/retry counts. Select exchanges with submit_check_result or finding;
        failed attempts can be inspected through http_history when selected.
    """

    async def __call__(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, Any] | str | None = None,
        query: dict[str, Any] | str | None = None,
        body: Any = None,
        body_type: str = "none",
        timeout: int | float | None = None,
        follow_redirects: bool = True,
        credential_id: str | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
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
        with self._recorded(safe_arguments) as call:
            headers, query, body = await threads.to_thread_until_done(
                request_containers, headers, query, body_type, body, name="http-tool-arguments"
            )
            result = await self._session.request(
                url=url,
                method=method,
                headers=headers,
                query=query,
                body_type=body_type,
                body=body,
                timeout_seconds=timeout,
                follow_redirects=follow_redirects,
                credential_id=credential_id,
                invocation_id=tool_context.invocation_id if tool_context is not None else None,
            )
            call.succeed(
                {
                    "statusClass": result["status"] // 100,
                    "bodyKind": result["body_kind"],
                    "bodyBytes": result["content_length"],
                    "redirects": result["redirects"],
                    "retries": result["retries"],
                }
            )
            return result


class HTTPReadBodyTool(_HTTPTool):
    name = "http_read_body"
    description = """Read a bounded slice of a response stored by http_request.

    If body_truncated is true, only a response prefix was retained. Total and
    eof describe the retained data, so eof does not prove the server body ended.

    Args:
        request_id: Integer request_id returned by http_request in this session.
        offset: Zero-based character offset for text or byte offset for binary.
        length: Maximum characters or bytes, from 1 to 8192; defaults to 8192.

    Returns:
        Text in data or base64 binary in data_b64, plus kind, unit, offset,
        returned length, total length and eof.
    """

    async def __call__(
        self,
        request_id: int,
        offset: int = 0,
        length: int = MAX_READ_UNITS,
    ) -> dict[str, Any]:
        arguments = {
            "requestId": request_id if type(request_id) is int else -1,
            "offset": offset if type(offset) is int else -1,
            "length": length if type(length) is int else -1,
        }
        with self._recorded(arguments, "http_body_not_found") as call:
            result = await self._session.read_body(request_id, offset, length)
            call.succeed({"kind": result["kind"], "length": result["length"], "eof": result["eof"]})
            return result


class HTTPHistoryTool(_HTTPTool):
    name = "http_history"
    description = """List recent HTTP request summaries from this allocation session.

    Args:
        limit: Number of most recent requests to include, from 1 to 128;
            defaults to 128.

    Returns:
        Summaries ordered oldest first within the selected recent requests.
        Use http_read_body with a request_id to read its stored body.
    """

    async def __call__(self, limit: int = MAX_HISTORY) -> list[dict[str, Any]]:
        arguments = {"limit": limit if type(limit) is int else -1}
        with self._recorded(arguments) as call:
            result = await self._session.history(limit)
            call.succeed({"count": len(result)})
            return result


class HTTPSessionSetTool(_HTTPTool):
    name = "http_session_set"
    description = """Update HTTP session cookies, default headers or authentication.

    Omitted fields keep their current values. Supplied cookies and headers merge
    by default; replacement flags clear their collections before applying values.
    Cookies are saved for continuation in this Run and Namespace. Headers and
    basic/bearer auth remain local to this allocation.
    Controlled Audits support only non-auth default headers here; cookies, auth
    and replace_cookies are invalid. Select credential_id per request instead.

    Args:
        cookies: Cookie name-to-string-value mapping.
        headers: Default header name-to-string-value mapping for future requests.
        auth: {"kind": "bearer", "token": "..."}, {"kind": "basic",
            "username": "...", "password": "..."}, or {"kind": "none"} to clear.
        replace_cookies: Clear existing cookies before applying cookies; default false.
        replace_headers: Clear existing headers before applying headers; default false.

    Returns:
        A redacted view of the updated session and its auth_kind.
    """

    async def __call__(
        self,
        cookies: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        auth: dict[str, Any] | None = None,
        replace_cookies: bool = False,
        replace_headers: bool = False,
    ) -> dict[str, Any]:
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
        with self._recorded(arguments, "http_request_invalid") as call:
            result = await self._session.set_session(
                headers=headers,
                cookies=cookies,
                auth=auth,
                replace_cookies=replace_cookies,
                replace_headers=replace_headers,
            )
            call.succeed({"authKind": result["auth_kind"]})
            return result


class HTTPSessionGetTool(_HTTPTool):
    name = "http_session_get"
    description = """Read the current HTTP session settings.

    Also reconcile an unconfirmed cookie checkpoint without sending another
    request to the target. Call this after cookie_checkpoint reports a failure.
    Controlled Audits return separate credential cookie summaries, without
    secret values or checkpoint persistence.

    Returns:
        A redacted view of cookies, default headers and auth_kind.
    """

    async def __call__(self) -> dict[str, Any]:
        with self._recorded({}) as call:
            result = await self._session.get_session()
            call.succeed({"authKind": result["auth_kind"]})
            return result


class HTTPSessionClearTool(_HTTPTool):
    name = "http_session_clear"
    description = """Clear HTTP session cookies, default headers, authentication and history.

    Ordinary HTTP commits the empty jar before returning. Controlled Audits clear
    allocation-local jars and block cleared cookie credentials; Basic/Bearer
    credentials remain available for explicit selection by credential_id.

    Returns:
        The cleared session view. Previously saved finding evidence remains stored.
    """

    async def __call__(self) -> dict[str, Any]:
        with self._recorded({}) as call:
            result = await self._session.clear_session()
            call.succeed({"cleared": True})
            return result


class HTTPCredentialsTool(_HTTPTool):
    name = "http_credentials"
    description = """List credentials authorized for this Audit's HTTP requests.

    Returns:
        Exact credential IDs, authentication kinds, permitted origins and declared roles.
        This tool returns no secret values and performs no target request.
    """

    async def __call__(self) -> list[dict[str, Any]]:
        with self._recorded({}) as call:
            result = self._session.credential_catalog()
            call.succeed({})
            return result


class HTTPCredentialReadTool(_HTTPTool):
    name = "http_credential_read"
    description = """Read values of one exact credential authorized for this Audit.

    Prefer credential_id in http_request when raw values are unnecessary. Do not
    copy secret values into reports, logs or published artifacts.

    Args:
        credential_id: Exact ID returned by http_credentials.

    Returns:
        The credential's authentication kind and secret values for explicit use.
        Missing or unassigned credentials are unavailable; this never sends HTTP.
    """

    async def __call__(self, credential_id: str) -> dict[str, Any]:
        with self._recorded({}) as call:
            result = self._session.credential_values(credential_id)
            call.succeed({})
            return result


def _selected_http_tools(selected, session, metrics):
    builders = {
        "http_request": HTTPRequestTool,
        "http_read_body": HTTPReadBodyTool,
        "http_history": HTTPHistoryTool,
        "http_session_set": HTTPSessionSetTool,
        "http_session_get": HTTPSessionGetTool,
        "http_session_clear": HTTPSessionClearTool,
        "http_credentials": HTTPCredentialsTool,
        "http_credential_read": HTTPCredentialReadTool,
    }
    return {name: builders[name](session, metrics) for name in selected}


def _direct_client(policy: TargetPolicy) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
    return httpx.AsyncClient(
        headers={"Accept-Encoding": BOUNDED_ACCEPT_ENCODING},
        transport=PolicyHTTPTransport(policy, limits=limits),
        trust_env=False,
        follow_redirects=False,
    )


def _tool_http_proxy_required(settings: RuntimeSettings) -> bool:
    return settings.http_proxy is not None and "tool-http" in settings.http_proxy.targets


def _http_metric_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    authorization = target_authorization(settings)
    secrets = runtime_secrets(settings)
    return secrets if authorization is None else (*secrets, authorization)


def _target_origin(settings: RuntimeSettings) -> tuple[str, str, int] | None:
    if settings.http_origin_target is None:
        return None
    return _origin(settings.http_origin_target.url)


def _method(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in _METHODS:
        raise HTTPToolError("http_request_invalid")
    return value.upper()


def _url_with_query(url: object, query: Mapping[str, Any] | None) -> str:
    if not isinstance(url, str) or not 1 <= request_utf8_size(url) <= MAX_URL_BYTES:
        raise HTTPToolError("http_request_invalid")
    # A fragment is client-side state that is never sent on the wire.
    url = url.partition("#")[0]
    try:
        parsed = urlsplit(url)
        existing = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
    except (UnicodeError, ValueError):
        raise HTTPToolError("http_request_invalid") from None
    additions = _query_pairs(query)
    if len(existing) + len(additions) > MAX_QUERY_KEYS:
        raise HTTPToolError("http_request_invalid")
    try:
        added_query = urlencode(additions, doseq=True)
    except UnicodeError:
        raise HTTPToolError("http_request_invalid") from None
    raw_query = parsed.query
    result = url
    if added_query:
        # Existing percent escapes, duplicate/bare keys and separators are part
        # of the requested target. Decode only for counting, never for rewriting.
        separator = "&" if raw_query else "" if "?" in url else "?"
        result = url + separator + added_query
        raw_query += ("&" if raw_query else "") + added_query
    if len(raw_query.encode("utf-8")) > MAX_QUERY_BYTES:
        raise HTTPToolError("http_request_invalid")
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
            if request_utf8_size(rendered) > MAX_HEADER_VALUE_BYTES:
                raise HTTPToolError("http_request_invalid")
            result.append((key, rendered))
    if len(result) > MAX_QUERY_KEYS:
        raise HTTPToolError("http_request_invalid")
    return result


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
    if len(encoded) > get_shared_config().http_request_body_max_bytes:
        raise HTTPToolError("http_request_invalid")
    return encoded, media_type


def _timeout(value: object, cap: int) -> float:
    selected: float
    if value is None:
        selected = min(float(cap), 120.0)
    elif type(value) in {int, float}:
        try:
            selected = float(value)
        except OverflowError:
            raise HTTPToolError("http_request_invalid") from None
    else:
        raise HTTPToolError("http_request_invalid")
    if not math.isfinite(selected) or selected < 1 or selected > 120:
        raise HTTPToolError("http_request_invalid")
    return min(selected, float(cap))


def _retry_delay(retry: int, retry_after: float | None, deadline: float) -> float | None:
    """Return the wait before retry number ``retry`` (1-based), or None to stop."""

    delay = RETRY_BACKOFF_SECONDS * 2 ** (retry - 1)
    if retry_after is not None:
        if not math.isfinite(retry_after) or retry_after > MAX_RETRY_AFTER_SECONDS:
            return None
        delay = max(delay, retry_after)
    remaining = deadline - asyncio.get_running_loop().time()
    # Leave the retry itself at least as much time as the wait before it.
    if delay * 2 > remaining:
        return None
    return delay


def _validate_target(url: str, policy: TargetPolicy) -> None:
    try:
        validate_web_url(url)
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise HTTPToolError("http_request_invalid")
        # The shared rule rejects percent-escapes and sub-delimiters that a
        # resolver or forward proxy could interpret differently.
        validate_host_syntax(host)
        _origin(url)
    except HTTPToolError:
        raise
    except (httpx.InvalidURL, UnicodeError, ValueError):
        raise HTTPToolError("http_request_invalid") from None
    # Literal and name checks run before every hop in both routes. A direct
    # client repeats the address check on the resolved peer at connect time.
    try:
        policy.check_url(url)
    except TargetDenied:
        raise HTTPToolError("http_target_denied") from None


def _without_sensitive(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if not _is_sensitive_header(name)}


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    port = parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
    return scheme, host, port


async def _read_response_body(response: httpx.Response) -> bytes:
    # Bounds both the received bytes and incremental gzip/deflate decoding.
    try:
        return await read_limited_body(response, get_shared_config().http_response_body_max_bytes)
    except BodyTooLarge:
        raise HTTPToolError("http_response_too_large") from None
    except (*HTTP_TRANSPORT_ERRORS, UnicodeError):
        raise HTTPToolError("http_request_failed") from None


def _content_type(headers: httpx.Headers) -> str:
    value = headers.get("content-type", "application/octet-stream").strip()
    if not value or len(value.encode("utf-8")) > 256 or any(char in value for char in "\r\n\x00"):
        return "application/octet-stream"
    return value


def _encode_body(
    content_type: str, body: bytes
) -> tuple[Literal["empty", "text", "binary"], str | None, str | bytes | None]:
    if not body:
        return "empty", "", None
    normalized = content_type.lower().split(";", 1)[0].strip()
    if normalized.startswith(_TEXTUAL_CONTENT_TYPES) or normalized.endswith("+json"):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return "text", text[:MAX_PREVIEW_CHARACTERS], text
    return "binary", None, body


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
            or total + size > get_shared_config().http_header_block_max_bytes
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
            or request_utf8_size(name) > 256
            or any(char in name for char in "\r\n\x00;,")
            or not isinstance(value, str)
            or request_utf8_size(value) > MAX_HEADER_VALUE_BYTES
            or any(char in value for char in "\r\n\x00")
            or not is_cookie_pair(name, value)
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
            or not 1 <= request_utf8_size(username) <= 256
            or not 1 <= request_utf8_size(password) <= MAX_HEADER_VALUE_BYTES
            or any(char in username + password for char in "\r\n\x00")
        ):
            raise HTTPToolError("http_request_invalid")
        return "basic", username, password
    if kind == "bearer":
        bearer_token = auth.get("token")
        if (
            set(auth) != {"kind", "token"}
            or not isinstance(bearer_token, str)
            or not 1 <= request_utf8_size(bearer_token) <= MAX_HEADER_VALUE_BYTES
            or not is_field_value(bearer_token)
        ):
            raise HTTPToolError("http_request_invalid")
        return "bearer", None, bearer_token
    raise HTTPToolError("http_request_invalid")


def _is_sensitive_header(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized in _SENSITIVE_HEADERS or any(
        marker in normalized for marker in ("api-key", "auth-token", "access-token")
    )
