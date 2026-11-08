"""Model-facing HTTP session over the allocation-owned scoped transport."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections import deque

import httpx

from contractor_runtime.adapters.http_transport import HTTPTransport, HTTPTransportError
from contractor_runtime.core.http_limits import MAX_HISTORY
from contractor_runtime.core.tool_errors import ToolResultDiagnostic
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.credentials import RUNTIME_CREDENTIAL
from contractor_runtime.toolsets.common.http_capture import HTTPObservation, RecentHTTPCaptures
from contractor_runtime.toolsets.http.capture import CapturedAttempt, CapturedExchange
from contractor_runtime.toolsets.http.errors import HTTPToolError, controlled_http_message
from contractor_runtime.toolsets.http.headers import body_headers, merge_headers, validate_headers
from contractor_runtime.toolsets.http.tools import (
    _encode_body,
    _HTTPSession,
    _method,
    _request_body,
    _RequestRecord,
    _response_headers,
    _timeout,
    _url_with_query,
)


class ControlledHTTPError(HTTPToolError, ToolResultDiagnostic):
    def __init__(self, code: str, *, message=None, scope=None, may_have_been_sent=None):
        super().__init__(code)
        self.retryable = False
        self.diagnostic_message = controlled_http_message(self.code)
        self.model_details = {}
        if message is not None:
            self.diagnostic_message = message
        if scope is not None:
            self.model_details["scope"] = scope
        if may_have_been_sent is not None:
            self.model_details["request_may_have_been_sent"] = may_have_been_sent
            if may_have_been_sent:
                self.diagnostic_message += (
                    " The request may have executed; its result is unknown."
                    " Do not repeat it automatically."
                )


class ControlledHTTPSession(_HTTPSession):
    """Reuse HTTP result/history readers with private allocation-only response storage."""

    def __init__(
        self, transport: HTTPTransport, timeout_cap: int, *, captures=None, deployment_marker=None
    ):
        self._transport = transport
        self._timeout_cap_seconds = timeout_cap
        self._lock = asyncio.Lock()
        self._history = deque(maxlen=MAX_HISTORY)
        # Explicit selection retains observations; capture alone establishes no vulnerability.
        self._captures = captures if captures is not None else RecentHTTPCaptures()
        self._deployment_marker = deployment_marker
        self._cached_body = None
        self._default_headers = {}
        self._next_request_id = 1
        self._nonce = secrets.token_hex(8)
        self._closed = False
        transport.attach_response_buffer(self._erase_response_buffer)

    @property
    def metric_secrets(self) -> tuple[str, ...]:
        return self._transport.private_values()

    @property
    def scope_description(self) -> str:
        return "\nAuthorized HTTP scope (DNS/IP restrictions are also enforced): " + json.dumps(
            self._transport.model_scope(), separators=(",", ":"), ensure_ascii=True
        )

    def credential_catalog(self):
        return self._transport.credentials()

    def _require_open(self):
        if self._closed:
            raise HTTPTransportError("http_allocation_closed")

    def credential_values(self, credential_id):
        try:
            return self._transport.read_credential(credential_id)
        except ValueError:
            raise ControlledHTTPError("http_credential_unavailable") from None

    async def request(self, **options):
        self._require_open()
        added = self._transport.own_invocation()
        try:
            timeout = _timeout(options.pop("timeout_seconds", None), self._timeout_cap_seconds)
            return await self._request(deadline=time.monotonic() + timeout, **options)
        except ControlledHTTPError:
            raise
        except HTTPToolError as error:
            raise ControlledHTTPError(error.code) from None
        finally:
            self._transport.finish_invocation(added)

    async def _request(
        self,
        *,
        url,
        method,
        headers,
        query,
        body_type,
        body,
        deadline,
        follow_redirects,
        invocation_id=None,
        credential_id=None,
    ):
        try:
            selected_method = _method(method)
        except HTTPToolError:
            raise ControlledHTTPError(
                "http_request_invalid",
                message="Use a supported HTTP method from the authorized scope.",
            ) from None
        if credential_id is not None and credential_id not in {
            entry["credential_id"] for entry in self.credential_catalog()
        }:
            raise ControlledHTTPError("http_credential_unavailable")
        try:
            selected_url = _url_with_query(url, query)
        except HTTPToolError:
            raise ControlledHTTPError(
                "http_request_invalid", message="Use an absolute HTTP(S) URL and a query mapping."
            ) from None
        try:
            payload, content_type = _request_body(body_type, body)
        except HTTPToolError:
            raise ControlledHTTPError(
                "http_request_invalid",
                message="Match body to body_type: none, json, form (mapping), or text (string).",
            ) from None
        if type(follow_redirects) is not bool:
            raise ControlledHTTPError(
                "http_request_invalid", message="follow_redirects must be a boolean."
            )
        try:
            async with asyncio.timeout_at(deadline):
                await self._lock.acquire()
        except TimeoutError:
            # No transport attempt can exist before this session owns its lock.
            # Authority loss still belongs to Worker lifecycle, not the model.
            self._transport.own_invocation()
            raise ControlledHTTPError("http_timeout", may_have_been_sent=False) from None
        try:
            self._require_open()
            selected_headers = body_headers(
                self._default_headers, validate_headers(headers), content_type
            )
            request_id = self._next_request_id
            self._next_request_id += 1
            started = time.monotonic()
            tag = f"r{self._nonce}-h{request_id:06d}"
            selected_headers = merge_headers(selected_headers, {"X-Request-Id": tag})
            capture = _ExchangeCapture(request_id, tag)
            completed = False
            try:
                outcome, response = await self._transport.request(
                    selected_url,
                    credential_id=credential_id,
                    method=selected_method,
                    headers=[
                        (name.encode("ascii"), value.encode("ascii"))
                        for name, value in selected_headers.items()
                    ],
                    body=payload,
                    timeout=max(0, deadline - time.monotonic()),
                    follow_redirects=follow_redirects,
                    capture=capture,
                )
                completed = response is not None and outcome.kind == "completed"
            except ValueError:
                raise ControlledHTTPError("http_request_invalid") from None
            finally:
                if capture.exchange.attempts:
                    if not completed:
                        last = capture.exchange.attempts[-1]
                        self._history.append(
                            _RequestRecord(
                                request_id=request_id,
                                tag=tag,
                                method=last.method,
                                final_url=last.url,
                                status=last.status,
                                content_type="",
                                body_kind="empty",
                                body_bytes=0,
                                redirects=0,
                                retries=max(0, len(capture.exchange.attempts) - 1),
                                elapsed_ms=int((time.monotonic() - started) * 1000),
                                preview=None,
                                headers={},
                                error=capture.exchange.error or "transport_error",
                            )
                        )
                    evidence = await to_thread_until_done(
                        capture.exchange.evidence, name="controlled-http-capture"
                    )
                    await self._captures.put(
                        evidence,
                        invocation_id,
                        observation=HTTPObservation(
                            capture.exchange.attempts[0].at, self._deployment_marker
                        ),
                    )
            if response is None or outcome.kind != "completed":
                code = {
                    "policy_denied": "http_target_denied",
                    "dns_failed": "http_dns_failed",
                    "tls_failed": "http_tls_failed",
                    "connect_failed": "http_connection_failed",
                    "transport_failed": "http_connection_failed",
                    "timeout": "http_timeout",
                    "response_invalid": "http_response_invalid",
                }.get(outcome.reason, "http_request_failed")
                raise ControlledHTTPError(
                    code,
                    scope=self._transport.model_scope() if code == "http_target_denied" else None,
                    may_have_been_sent=outcome.request_may_have_been_sent,
                )
            message = httpx.Response(
                response.status, headers=response.headers, content=response.body
            )
            content_type = message.headers.get("content-type", "")
            kind, preview, _ = await to_thread_until_done(
                _encode_body, content_type, response.body, name="audit-http-body-preview"
            )
            safe_headers, headers_truncated = _response_headers(message.headers)
            record = _RequestRecord(
                request_id=request_id,
                tag=tag,
                method=response.method,
                final_url=response.url,
                status=response.status,
                content_type=content_type,
                body_kind=kind,
                body_bytes=len(response.body),
                redirects=response.redirects,
                retries=response.retries,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                preview=preview,
                headers=safe_headers,
                headers_truncated=headers_truncated,
                retained_body_truncated=response.truncated,
            )
            self._history.append(record)
            result = record.projection(include_preview=True)
            if not response.session_valid:
                result["session_error"] = "http_cookie_state_invalid"
            return result
        finally:
            self._lock.release()

    async def get_session(self):
        async with self._lock:
            self._require_open()
            sessions = await self._transport.session_summaries()
            anonymous = next(item for item in sessions if item["credential_id"] is None)
            return {
                "auth_kind": "none",
                "default_headers": dict(self._default_headers),
                "cookie_names": anonymous["cookie_names"],
                "cookie_count": anonymous["cookie_count"],
                "history_count": len(self._history),
                "credentials": self.credential_catalog(),
                "credential_sessions": [
                    item for item in sessions if item["credential_id"] is not None
                ],
            }

    async def set_session(self, *, headers, cookies, auth, replace_cookies, replace_headers):
        # There is no single implicit credential session for a whole Audit.
        # Raw values remain usable as explicit per-request Cookie/Authorization.
        if (
            cookies is not None
            or auth is not None
            or replace_cookies
            or type(replace_headers) is not bool
        ):
            raise ControlledHTTPError("http_request_invalid")
        selected = validate_headers(headers)
        if any(name.lower() in {"authorization", "cookie"} for name in selected):
            raise ControlledHTTPError("http_request_invalid")
        async with self._lock:
            self._require_open()
            self._default_headers = merge_headers(
                {} if replace_headers else self._default_headers, selected
            )
        return await self.get_session()

    async def clear_session(self):
        async with self._lock:
            self._require_open()
            await self._transport.clear_cookies()
            self._default_headers.clear()
            self._history.clear()
            self._captures.clear("http")
            self._cached_body = None
        return await self.get_session()

    async def close(self):
        await self._transport.close()
        async with self._lock:
            self._erase_response_buffer()

    def _erase_response_buffer(self):
        self._closed = True
        self._history.clear()
        self._captures.clear("http")
        self._cached_body = None
        self._default_headers.clear()


class _ExchangeCapture:
    """Receive actual transport attempts; no storage or authority of its own."""

    def __init__(self, request_id, tag):
        self.exchange = CapturedExchange(request_id, tag)

    def request(self, method, url, headers, body, injected_headers):
        safe = [
            {
                "name": name.decode("ascii"),
                "value": (
                    RUNTIME_CREDENTIAL
                    if name.lower() in injected_headers
                    and name.lower() in {b"authorization", b"cookie"}
                    else value.decode("latin-1")
                ),
            }
            for name, value in headers
        ]
        self.exchange.attempts.append(CapturedAttempt(method, url, safe, body))
        self.exchange.complete = False
        self.exchange.response_body = b""
        self.exchange.error = None

    def finish(self, response, error):
        attempt = self.exchange.attempts[-1]
        if response is not None:
            attempt.status = response.status
            attempt.response_headers = [
                {"name": k.decode("ascii"), "value": v.decode("latin-1")}
                for k, v in response.headers
            ]
            self.exchange.response_body = response.body
            self.exchange.complete = error is None and not response.truncated
            if response.truncated and error is None:
                error = "response_too_large"
        self.exchange.error = error
        if error is not None and response is None:
            attempt.error = error
