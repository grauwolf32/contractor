"""Model-facing HTTP session over the allocation-owned scoped transport."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque

import httpx

from contractor_runtime.adapters.http_transport import HTTPTransport, HTTPTransportError
from contractor_runtime.core.http_limits import MAX_HISTORY
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.http_capture import RecentHTTPCaptures
from contractor_runtime.toolsets.http.capture import CapturedAttempt, CapturedExchange
from contractor_runtime.toolsets.http.errors import HTTPToolError
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


class ControlledHTTPError(HTTPToolError):
    def __init__(self, code: str):
        super().__init__(code)
        self.retryable = False


class ControlledHTTPSession(_HTTPSession):
    """Reuse HTTP result/history readers with private allocation-only response storage."""

    def __init__(self, transport: HTTPTransport, timeout_cap: int):
        self._transport = transport
        self._timeout_cap_seconds = timeout_cap
        self._lock = asyncio.Lock()
        self._history = deque(maxlen=MAX_HISTORY)
        # Interactive responses do not acquire trusted engagement capture/proof status.
        self._captures = RecentHTTPCaptures()
        self._cached_body = None
        self._default_headers = {}
        self._next_request_id = 1
        self._nonce = secrets.token_hex(8)
        self._closed = False
        transport.attach_response_buffer(self._erase_response_buffer)

    @property
    def metric_secrets(self) -> tuple[str, ...]:
        return self._transport.private_values()

    def credential_catalog(self):
        try:
            return self._transport.credentials()
        except HTTPTransportError:
            raise ControlledHTTPError("http_credential_unavailable") from None

    def credential_values(self, credential_id):
        try:
            return self._transport.read_credential(credential_id)
        except (ValueError, HTTPTransportError):
            raise ControlledHTTPError("http_credential_unavailable") from None

    async def request(self, **options):
        self._require_open()
        timeout = _timeout(options.get("timeout_seconds"), self._timeout_cap_seconds)
        try:
            added = self._transport.own_invocation()
        except HTTPTransportError:
            raise ControlledHTTPError("http_target_denied") from None
        try:
            async with asyncio.timeout(timeout):
                return await self._request(**options)
        except TimeoutError:
            raise ControlledHTTPError("http_request_failed") from None
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
        timeout_seconds,
        follow_redirects,
        invocation_id=None,
        credential_id=None,
    ):
        selected_method = _method(method)
        if credential_id is not None and credential_id not in {
            entry["credential_id"] for entry in self.credential_catalog()
        }:
            raise ControlledHTTPError("http_credential_unavailable")
        selected_url = _url_with_query(url, query)
        payload, content_type = _request_body(body_type, body)
        if type(follow_redirects) is not bool:
            raise ControlledHTTPError("http_request_invalid")
        if timeout_seconds is None:
            timeout_seconds = self._timeout_cap_seconds
        if (
            type(timeout_seconds) not in {int, float}
            or not 0 < timeout_seconds <= self._timeout_cap_seconds
        ):
            raise ControlledHTTPError("http_request_invalid")
        async with self._lock:
            self._require_open()
            selected_headers = body_headers(
                self._default_headers, validate_headers(headers), content_type
            )
            request_id = self._next_request_id
            self._next_request_id += 1
            started = time.monotonic()
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
                    timeout=timeout_seconds,
                    follow_redirects=follow_redirects,
                )
            except ValueError:
                raise ControlledHTTPError("http_request_invalid") from None
            except TimeoutError:
                raise ControlledHTTPError("http_request_failed") from None
            except HTTPTransportError as error:
                raise ControlledHTTPError(
                    "http_target_denied"
                    if error.code == "http_scope_denied"
                    else "http_request_failed"
                ) from None
            if response is None or outcome.kind != "completed":
                code = (
                    "http_target_denied"
                    if outcome.reason == "policy_denied"
                    else "http_request_failed"
                )
                raise ControlledHTTPError(code)
            message = httpx.Response(
                response.status, headers=response.headers, content=response.body
            )
            content_type = message.headers.get("content-type", "")
            kind, preview, _ = await to_thread_until_done(
                _encode_body, content_type, response.body, name="audit-http-body-preview"
            )
            safe_headers, headers_truncated = _response_headers(message.headers)
            tag = f"r{self._nonce}-h{request_id:06d}"
            attempt = CapturedAttempt.from_request(
                httpx.Request(
                    selected_method, selected_url, headers=selected_headers, content=payload
                )
            )
            attempt.receive(message)
            exchange = CapturedExchange(
                request_id,
                tag,
                attempts=[attempt],
                complete=not response.truncated,
                response_body=response.body,
                error="response_too_large" if response.truncated else None,
            )
            evidence = await to_thread_until_done(
                exchange.evidence, name="audit-http-response-buffer"
            )
            await self._captures.put(evidence, invocation_id)
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
