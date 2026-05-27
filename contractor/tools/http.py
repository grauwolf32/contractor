from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, TypeAlias, TypedDict

import httpx
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

JSONLike: TypeAlias = dict[str, Any] | list[Any]
HTTPRequestMethod: TypeAlias = Literal[
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"
]
BodyType: TypeAlias = Literal["json", "form", "text", "none"]
AuthKind: TypeAlias = Literal["bearer", "basic", "none"]
ContextLike: TypeAlias = ToolContext | CallbackContext

_TEXTUAL_MIME_PREFIXES: tuple[str, ...] = ("text/",)
_TEXTUAL_MIME_EXACT: frozenset[str] = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/ecmascript",
        "application/xhtml+xml",
        "application/x-www-form-urlencoded",
        "application/x-yaml",
        "application/yaml",
    }
)
_TEXTUAL_MIME_SUFFIXES: tuple[str, ...] = ("+json", "+xml", "+yaml")
_AUTH_HEADER: str = "Authorization"
_REDACTED: str = "***redacted***"


class ArtifactContext(Protocol):
    async def save_artifact(self, *, filename: str, artifact: types.Part) -> None: ...
    async def load_artifact(self, *, filename: str) -> types.Part | None: ...


class HistorySummary(TypedDict):
    request_id: int
    method: str
    url: str
    status: int
    content_type: str
    content_length: int
    elapsed_ms: int


class ResponseRecord(TypedDict):
    request_id: int
    method: str
    final_url: str
    status: int
    content_type: str
    content_length: int
    headers: dict[str, str]
    body_kind: Literal["text", "binary", "empty"]
    body_preview: str | None
    body_truncated: bool
    body_artifact: str | None
    elapsed_ms: int


class AuthState(TypedDict, total=False):
    kind: AuthKind
    token: str
    username: str
    password: str


class SessionState(TypedDict):
    cookies: dict[str, str]
    default_headers: dict[str, str]
    auth: AuthState | None
    history: list[HistorySummary]
    next_request_id: int


ToolFn: TypeAlias = Callable[..., Any]


@dataclass(slots=True)
class RetryConfig:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


class HTTPClientError(Exception):
    pass


def _is_textual_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    primary = content_type.split(";", 1)[0].strip().lower()
    if not primary:
        return False
    if primary in _TEXTUAL_MIME_EXACT:
        return True
    if primary.startswith(_TEXTUAL_MIME_PREFIXES):
        return True
    return any(primary.endswith(s) for s in _TEXTUAL_MIME_SUFFIXES)


class HTTPClient:
    """
    Async HTTP client backing the agent-facing tools.

    - Cookies, default headers, auth, and request history persist via the
      session artifact `http/<name>/session.json`.
    - Each response body is offloaded to its own artifact
      `http/<name>/responses/<request_id>.json` so the agent's context is
      never flooded by large payloads.
    """

    def __init__(
        self,
        name: str,
        *,
        proxy: str | None = None,
        history_size: int = 20,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        user_agent: str = "LLM-Agent-HTTP-Tools/1.0",
        body_preview_chars: int = 2048,
        retry_config: RetryConfig | None = None,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be > 0")
        if body_preview_chars <= 0:
            raise ValueError("body_preview_chars must be > 0")

        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self._history_size = history_size
        self._body_preview_chars = body_preview_chars
        self._history: deque[HistorySummary] = deque(maxlen=history_size)
        self._default_headers: dict[str, str] = {}
        self._auth: AuthState | None = None
        self._next_request_id: int = 1
        self._state_lock = asyncio.Lock()

        self._client = httpx.AsyncClient(
            proxy=proxy,
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HTTPClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ── session state ────────────────────────────────────────────────

    def session_artifact_name(self) -> str:
        return f"http/{self.name}/session.json"

    def body_artifact_name(self, request_id: int) -> str:
        return f"http/{self.name}/responses/{request_id:08d}.json"

    def get_cookies(self) -> dict[str, str]:
        return dict(self._client.cookies.items())

    def set_cookies(
        self, cookies: Mapping[str, str], *, replace: bool = False
    ) -> None:
        if replace:
            self._client.cookies.clear()
        for k, v in cookies.items():
            self._client.cookies.set(str(k), str(v))

    def set_default_headers(
        self, headers: Mapping[str, str], *, replace: bool = False
    ) -> None:
        if replace:
            self._default_headers.clear()
        for k, v in headers.items():
            self._default_headers[str(k)] = str(v)

    def set_auth(self, auth: AuthState | Mapping[str, str] | None) -> None:
        if auth is None or auth.get("kind") in (None, "none"):
            self._auth = None
            return
        kind = auth["kind"]
        if kind == "bearer":
            token = auth.get("token")
            if not token:
                raise HTTPClientError("bearer auth requires 'token'")
            self._auth = {"kind": "bearer", "token": str(token)}
        elif kind == "basic":
            username = auth.get("username")
            if not username:
                raise HTTPClientError("basic auth requires 'username'")
            self._auth = {
                "kind": "basic",
                "username": str(username),
                "password": str(auth.get("password", "")),
            }
        else:
            raise HTTPClientError(f"unsupported auth kind: {kind!r}")

    def get_auth_kind(self) -> AuthKind:
        if self._auth is None:
            return "none"
        return self._auth["kind"]

    def get_history(self, limit: int | None = None) -> list[HistorySummary]:
        items = list(self._history)
        if limit is not None and limit > 0:
            items = items[-limit:]
        return items

    def clear_session_state(self) -> None:
        self._client.cookies.clear()
        self._default_headers.clear()
        self._auth = None
        self._history.clear()
        self._next_request_id = 1

    def _redacted_default_headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in self._default_headers.items():
            if k.lower() == _AUTH_HEADER.lower():
                out[k] = _REDACTED
            else:
                out[k] = v
        return out

    def _dump_session_state(self) -> SessionState:
        return {
            "cookies": self.get_cookies(),
            "default_headers": dict(self._default_headers),
            "auth": dict(self._auth) if self._auth is not None else None,
            "history": list(self._history),
            "next_request_id": self._next_request_id,
        }

    def _restore_session_state(self, state: SessionState) -> None:
        self.clear_session_state()
        self.set_cookies(state.get("cookies", {}))
        self.set_default_headers(state.get("default_headers", {}))
        if state.get("auth"):
            self.set_auth(state["auth"])
        for item in state.get("history", [])[-self._history_size:]:
            self._history.append(item)
        self._next_request_id = max(int(state.get("next_request_id", 1)), 1)

    async def save_session(self, ctx: ArtifactContext) -> None:
        async with self._state_lock:
            payload = self._dump_session_state()
            await ctx.save_artifact(
                filename=self.session_artifact_name(),
                artifact=types.Part.from_text(
                    text=json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            )

    async def load_session(self, ctx: ArtifactContext) -> bool:
        async with self._state_lock:
            artifact = await ctx.load_artifact(filename=self.session_artifact_name())
            if artifact is None or not artifact.text:
                self.clear_session_state()
                return False

            try:
                raw = json.loads(artifact.text)
            except json.JSONDecodeError as exc:
                raise HTTPClientError(
                    "Stored HTTP session artifact is not valid JSON"
                ) from exc

            if not isinstance(raw, dict):
                raise HTTPClientError("Stored HTTP session artifact is malformed")

            state: SessionState = {
                "cookies": {
                    str(k): str(v) for k, v in (raw.get("cookies") or {}).items()
                },
                "default_headers": {
                    str(k): str(v)
                    for k, v in (raw.get("default_headers") or {}).items()
                },
                "auth": raw.get("auth")
                if isinstance(raw.get("auth"), dict)
                else None,
                "history": [
                    item
                    for item in (raw.get("history") or [])
                    if isinstance(item, dict)
                ],
                "next_request_id": int(raw.get("next_request_id") or 1),
            }
            self._restore_session_state(state)
            return True

    # ── body persistence ─────────────────────────────────────────────

    async def _save_body(
        self,
        ctx: ArtifactContext,
        request_id: int,
        *,
        kind: Literal["text", "binary"],
        text: str | None,
        data: bytes | None,
        content_type: str,
    ) -> str:
        payload: dict[str, Any] = {"kind": kind, "content_type": content_type}
        if kind == "text":
            payload["text"] = text or ""
        else:
            payload["data_b64"] = base64.b64encode(data or b"").decode("ascii")

        artifact_name = self.body_artifact_name(request_id)
        await ctx.save_artifact(
            filename=artifact_name,
            artifact=types.Part.from_text(
                text=json.dumps(payload, ensure_ascii=False)
            ),
        )
        return artifact_name

    async def _load_body(
        self, ctx: ArtifactContext, request_id: int
    ) -> dict[str, Any] | None:
        artifact = await ctx.load_artifact(
            filename=self.body_artifact_name(request_id)
        )
        if artifact is None or not artifact.text:
            return None
        try:
            raw = json.loads(artifact.text)
        except json.JSONDecodeError as exc:
            raise HTTPClientError(
                f"Stored body for request {request_id} is not valid JSON"
            ) from exc
        if not isinstance(raw, dict):
            raise HTTPClientError(
                f"Stored body for request {request_id} is malformed"
            )
        return raw

    # ── request execution ───────────────────────────────────────────

    def _apply_auth_headers(self, headers: dict[str, str]) -> dict[str, str]:
        if self._auth is None:
            return headers
        if any(k.lower() == _AUTH_HEADER.lower() for k in headers):
            return headers
        out = dict(headers)
        if self._auth["kind"] == "bearer":
            out[_AUTH_HEADER] = f"Bearer {self._auth['token']}"
        elif self._auth["kind"] == "basic":
            creds = (
                f"{self._auth['username']}:{self._auth.get('password', '')}"
            ).encode()
            out[_AUTH_HEADER] = "Basic " + base64.b64encode(creds).decode("ascii")
        return out

    def _build_request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str] | None,
        query: Mapping[str, object] | None,
        body: Any,
        body_type: BodyType,
    ) -> httpx.Request:
        merged_headers: dict[str, str] = dict(self._default_headers)
        if headers:
            for k, v in headers.items():
                merged_headers[str(k)] = str(v)
        merged_headers = self._apply_auth_headers(merged_headers)

        kwargs: dict[str, Any] = {
            "method": method.upper(),
            "url": url,
            "headers": merged_headers,
            "params": dict(query) if query is not None else None,
        }

        if body_type == "json":
            kwargs["json"] = body
        elif body_type == "form":
            if body is not None and not isinstance(body, Mapping):
                raise HTTPClientError("body_type='form' requires a mapping body")
            kwargs["data"] = dict(body) if body is not None else None
        elif body_type == "text":
            if body is not None and not isinstance(body, (str, bytes)):
                raise HTTPClientError("body_type='text' requires str or bytes body")
            kwargs["content"] = body
        elif body_type == "none":
            pass
        else:
            raise HTTPClientError(f"unsupported body_type: {body_type!r}")

        return self._client.build_request(**kwargs)

    async def _send_with_retries(
        self,
        request: httpx.Request,
        *,
        follow_redirects: bool,
        timeout: float | None,
    ) -> httpx.Response:
        saved_timeout = self._client.timeout
        if timeout is not None:
            self._client.timeout = httpx.Timeout(timeout)

        last_error: BaseException | None = None
        try:
            for attempt in range(1, self.retry_config.attempts + 1):
                try:
                    response = await self._client.send(
                        request, follow_redirects=follow_redirects
                    )
                    if response.status_code not in self.retry_config.retry_on_statuses:
                        return response
                    last_error = HTTPClientError(
                        f"Retryable HTTP status {response.status_code} "
                        f"for {request.method} {request.url}"
                    )
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:
                    last_error = exc

                if attempt < self.retry_config.attempts:
                    delay = min(
                        self.retry_config.base_delay * (2 ** (attempt - 1)),
                        self.retry_config.max_delay,
                    )
                    await asyncio.sleep(delay)
        finally:
            self._client.timeout = saved_timeout

        if last_error is None:
            raise HTTPClientError("Request failed for an unknown reason")
        raise HTTPClientError(str(last_error)) from last_error

    async def request(
        self,
        *,
        url: str,
        method: HTTPRequestMethod | str = "GET",
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
        body: Any = None,
        body_type: BodyType = "none",
        timeout: float | None = None,
        follow_redirects: bool = True,
        ctx: ArtifactContext | None = None,
    ) -> ResponseRecord:
        if ctx is not None:
            await self.load_session(ctx)

        request_id = self._next_request_id
        self._next_request_id += 1

        request = self._build_request(
            method=method,
            url=url,
            headers=headers,
            query=query,
            body=body,
            body_type=body_type,
        )

        start = time.monotonic()
        response = await self._send_with_retries(
            request, follow_redirects=follow_redirects, timeout=timeout
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        content_type = response.headers.get("content-type", "")
        content_length = len(response.content)

        body_kind: Literal["text", "binary", "empty"]
        body_preview: str | None
        body_truncated = False
        body_artifact: str | None = None
        text_for_artifact: str | None = None
        bytes_for_artifact: bytes | None = None

        if content_length == 0:
            body_kind = "empty"
            body_preview = ""
        elif _is_textual_content_type(content_type):
            body_kind = "text"
            text = response.text
            text_for_artifact = text
            if len(text) > self._body_preview_chars:
                body_preview = (
                    text[: self._body_preview_chars]
                    + f"\n... [truncated, total {len(text)} chars]"
                )
                body_truncated = True
            else:
                body_preview = text
        else:
            body_kind = "binary"
            body_preview = None
            bytes_for_artifact = response.content

        if ctx is not None and body_kind != "empty":
            body_artifact = await self._save_body(
                ctx,
                request_id,
                kind=body_kind,
                text=text_for_artifact,
                data=bytes_for_artifact,
                content_type=content_type,
            )

        record: ResponseRecord = {
            "request_id": request_id,
            "method": request.method,
            "final_url": str(response.url),
            "status": response.status_code,
            "content_type": content_type,
            "content_length": content_length,
            "headers": dict(response.headers),
            "body_kind": body_kind,
            "body_preview": body_preview,
            "body_truncated": body_truncated,
            "body_artifact": body_artifact,
            "elapsed_ms": elapsed_ms,
        }

        summary: HistorySummary = {
            "request_id": request_id,
            "method": request.method,
            "url": str(response.url),
            "status": response.status_code,
            "content_type": content_type,
            "content_length": content_length,
            "elapsed_ms": elapsed_ms,
        }
        self._history.append(summary)

        if ctx is not None:
            await self.save_session(ctx)

        return record

    async def read_body(
        self,
        request_id: int,
        *,
        offset: int = 0,
        length: int = 4096,
        ctx: ArtifactContext | None = None,
    ) -> dict[str, Any]:
        if offset < 0:
            raise HTTPClientError("offset must be >= 0")
        if length <= 0:
            raise HTTPClientError("length must be > 0")
        if ctx is None:
            raise HTTPClientError("read_body requires a tool context")

        raw = await self._load_body(ctx, request_id)
        if raw is None:
            raise HTTPClientError(f"no stored body for request_id={request_id}")

        kind = raw.get("kind")
        content_type = raw.get("content_type", "")

        if kind == "text":
            text = raw.get("text", "")
            slice_ = text[offset : offset + length]
            return {
                "request_id": request_id,
                "kind": "text",
                "content_type": content_type,
                "offset": offset,
                "length": len(slice_),
                "total_length": len(text),
                "data": slice_,
            }
        if kind == "binary":
            data = base64.b64decode(raw.get("data_b64", ""))
            slice_ = data[offset : offset + length]
            return {
                "request_id": request_id,
                "kind": "binary",
                "content_type": content_type,
                "offset": offset,
                "length": len(slice_),
                "total_length": len(data),
                "data_b64": base64.b64encode(slice_).decode("ascii"),
            }
        raise HTTPClientError(f"unknown stored body kind: {kind!r}")


def http_tools(
    name: str,
    *,
    proxy: str | None = None,
    history_size: int = 20,
    timeout: float = 30.0,
    verify_ssl: bool = True,
    user_agent: str = "LLM-Agent-HTTP-Tools/1.0",
    body_preview_chars: int = 2048,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
) -> list[ToolFn]:
    cli = HTTPClient(
        name=name,
        proxy=proxy,
        history_size=history_size,
        timeout=timeout,
        verify_ssl=verify_ssl,
        user_agent=user_agent,
        body_preview_chars=body_preview_chars,
        retry_config=RetryConfig(
            attempts=attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on_statuses=retry_on_statuses,
        ),
    )

    async def http_request(
        url: str,
        method: HTTPRequestMethod = "GET",
        headers: dict[str, str] | None = None,
        query: dict[str, Any] | None = None,
        body: Any = None,
        body_type: BodyType = "none",
        timeout: float | None = None,
        follow_redirects: bool = True,
        tool_context: ToolContext | None = None,
    ) -> ResponseRecord | dict[str, str]:
        """
        Issue an HTTP request and return a small response record.

        The full response body is NEVER returned inline. Inspect `body_preview`
        (a leading slice of textual bodies) and call `http_read_body(request_id,
        offset, length)` to fetch more of the body when you need it. Binary
        responses have `body_preview=None`; use `http_read_body` to read them.

        Args:
            url: target URL.
            method: HTTP method.
            headers: per-request headers, merged on top of session defaults.
            query: querystring parameters.
            body: request body; shape depends on `body_type`.
            body_type:
                "json" - body is dict/list, sent as JSON.
                "form" - body is dict, sent as application/x-www-form-urlencoded.
                "text" - body is str/bytes, sent verbatim (set Content-Type via headers).
                "none" - no body sent.
            timeout: per-request timeout in seconds (overrides client default).
            follow_redirects: whether to follow 3xx responses.

        Returns the response record, or {"error": "..."} on failure.
        """
        try:
            return await cli.request(
                url=url,
                method=method,
                headers=headers,
                query=query,
                body=body,
                body_type=body_type,
                timeout=timeout,
                follow_redirects=follow_redirects,
                ctx=tool_context,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def http_read_body(
        request_id: int,
        offset: int = 0,
        length: int = 4096,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """
        Read a slice of a previously stored response body.

        For text bodies returns {"kind": "text", "data": "<slice>", ...}.
        For binary bodies returns {"kind": "binary", "data_b64": "<slice>", ...}.

        Args:
            request_id: id from a prior http_request response record.
            offset: char/byte offset to start reading from.
            length: max chars/bytes to return in this slice.
        """
        try:
            return await cli.read_body(
                request_id=request_id,
                offset=offset,
                length=length,
                ctx=tool_context,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def http_history(
        limit: int = 10,
        tool_context: ToolContext | None = None,
    ) -> list[HistorySummary] | dict[str, str]:
        """
        Return summaries of recent requests, oldest first.

        Each summary contains: request_id, method, url, status, content_type,
        content_length, elapsed_ms. Bodies are not included; use
        http_read_body(request_id) to fetch a stored body.
        """
        try:
            if tool_context is not None:
                await cli.load_session(tool_context)
            return cli.get_history(limit=limit if limit > 0 else None)
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def http_session_set(
        cookies: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        auth: dict[str, str] | None = None,
        replace_cookies: bool = False,
        replace_headers: bool = False,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """
        Update persistent session state. All args are optional and applied sparsely.

        Args:
            cookies: cookie name -> value, merged into the cookie jar.
            headers: default headers added to every subsequent request.
            auth: auth config. Examples:
                {"kind": "bearer", "token": "..."}
                {"kind": "basic",  "username": "...", "password": "..."}
                {"kind": "none"}    # clear auth
            replace_cookies: if true, clear the cookie jar before applying `cookies`.
            replace_headers: if true, clear default headers before applying `headers`.

        Returns the updated session view (auth values are redacted).
        """
        try:
            if tool_context is not None:
                await cli.load_session(tool_context)
            if cookies is not None:
                cli.set_cookies(cookies, replace=replace_cookies)
            if headers is not None:
                cli.set_default_headers(headers, replace=replace_headers)
            if auth is not None:
                cli.set_auth(auth)
            if tool_context is not None:
                await cli.save_session(tool_context)
            return {
                "status": "ok",
                "cookies": cli.get_cookies(),
                "default_headers": cli._redacted_default_headers(),
                "auth_kind": cli.get_auth_kind(),
            }
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def http_session_get(
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Return the current session: cookies, default headers, and auth kind."""
        try:
            if tool_context is not None:
                await cli.load_session(tool_context)
            return {
                "cookies": cli.get_cookies(),
                "default_headers": cli._redacted_default_headers(),
                "auth_kind": cli.get_auth_kind(),
            }
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def http_session_clear(
        tool_context: ToolContext | None = None,
    ) -> dict[str, str]:
        """Clear cookies, default headers, auth, and history. Stored bodies remain."""
        try:
            cli.clear_session_state()
            if tool_context is not None:
                await cli.save_session(tool_context)
            return {"status": "ok", "message": "session cleared"}
        except HTTPClientError as exc:
            return {"error": str(exc)}

    return [
        http_request,
        http_read_body,
        http_history,
        http_session_set,
        http_session_get,
        http_session_clear,
    ]
