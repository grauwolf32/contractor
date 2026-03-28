from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, TypeAlias, TypedDict

import httpx
from bs4 import BeautifulSoup
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types


JSONLike: TypeAlias = dict[str, Any] | list[Any]
ParsedBody: TypeAlias = JSONLike | str
HTTPRequestMethod: TypeAlias = Literal[
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"
]
ContextLike: TypeAlias = ToolContext | CallbackContext


class ArtifactContext(Protocol):
    async def save_artifact(self, *, filename: str, artifact: types.Part) -> None: ...
    async def load_artifact(self, *, filename: str) -> types.Part | None: ...


class ResponseRecord(TypedDict):
    url: str
    method: str
    status_code: int
    headers: dict[str, str]
    cookies: dict[str, str]
    body: ParsedBody
    attempt_count: int


class SessionState(TypedDict):
    cookies: dict[str, str]
    history: list[ResponseRecord]
    last_response: ResponseRecord | None


ToolFn: TypeAlias = Callable[..., Any]


@dataclass(slots=True)
class RetryConfig:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


class HTTPClientError(Exception):
    pass


class HTTPClient:
    """
    Async HTTP client with persistent session storage for LLM agents.

    Features:
    - Shared cookie/session state
    - Optional proxy configured at initialization
    - Automatic retries on all requests
    - Bounded request history
    - Auto-parse response body as JSON if possible, else text
    - Internal artifact-backed session persistence
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
        retry_config: RetryConfig | None = None,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be > 0")

        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self._history_size = history_size
        self._history: deque[ResponseRecord] = deque(maxlen=history_size)
        self._last_response_record: ResponseRecord | None = None
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

    def set_cookies(self, cookies: Mapping[str, str]) -> None:
        for key, value in cookies.items():
            self._client.cookies.set(str(key), str(value))

    def get_cookies(self) -> dict[str, str]:
        return dict(self._client.cookies.items())

    def clear_session_state(self) -> None:
        self._client.cookies.clear()
        self._history.clear()
        self._last_response_record = None

    def session_artifact_name(self) -> str:
        return f"http/{self.name}/session.json"

    def _dump_session_state(self) -> SessionState:
        return {
            "cookies": self.get_cookies(),
            "history": list(self._history),
            "last_response": self._last_response_record,
        }

    def _restore_session_state(self, state: SessionState) -> None:
        self.clear_session_state()

        self.set_cookies(state["cookies"])

        for item in state["history"][-self._history_size :]:
            self._history.append(item)

        self._last_response_record = state["last_response"]

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

            cookies = raw.get("cookies")
            history = raw.get("history")
            last_response = raw.get("last_response")

            if not isinstance(cookies, dict):
                raise HTTPClientError("Stored HTTP session cookies are malformed")
            if not isinstance(history, list):
                raise HTTPClientError("Stored HTTP session history is malformed")
            if last_response is not None and not isinstance(last_response, dict):
                raise HTTPClientError("Stored HTTP session last_response is malformed")

            state: SessionState = {
                "cookies": {str(k): str(v) for k, v in cookies.items()},
                "history": [item for item in history if isinstance(item, dict)],
                "last_response": last_response
                if isinstance(last_response, dict)
                else None,
            }

            self._restore_session_state(state)
            return True

    def _parse_response_body(self, response: httpx.Response) -> ParsedBody:
        """
        Return dict/list if JSON-decoding succeeds, otherwise text.
        """
        text = response.text
        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            try:
                parsed = response.json()
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                return text

        try:
            parsed = response.json()
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

        return text

    def _make_response_record(
        self,
        response: httpx.Response,
        parsed_body: ParsedBody,
        *,
        attempt_count: int,
    ) -> ResponseRecord:
        return {
            "url": str(response.url),
            "method": response.request.method,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "cookies": self.get_cookies(),
            "body": parsed_body,
            "attempt_count": attempt_count,
        }

    async def _send_with_retries(
        self,
        request: httpx.Request,
    ) -> tuple[httpx.Response, int]:
        last_error: BaseException | None = None

        for attempt in range(1, self.retry_config.attempts + 1):
            try:
                response = await self._client.send(request)

                if response.status_code not in self.retry_config.retry_on_statuses:
                    return response, attempt

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

        if last_error is None:
            raise HTTPClientError("Request failed for an unknown reason")

        raise HTTPClientError(str(last_error)) from last_error

    async def request(
        self,
        *,
        url: str,
        method: HTTPRequestMethod | str = "GET",
        cookies: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: JSONLike | None = None,
        params: Mapping[str, object] | None = None,
        ctx: ArtifactContext | None = None,
    ) -> ResponseRecord:
        if ctx is not None:
            await self.load_session(ctx)

        if cookies:
            self.set_cookies(cookies)

        request = self._client.build_request(
            method=method.upper(),
            url=url,
            headers=dict(headers) if headers is not None else None,
            json=json_body,
            params=dict(params) if params is not None else None,
        )

        response, attempt_count = await self._send_with_retries(request)
        parsed_body = self._parse_response_body(response)
        record = self._make_response_record(
            response,
            parsed_body,
            attempt_count=attempt_count,
        )

        self._last_response_record = record
        self._history.append(record)

        if ctx is not None:
            await self.save_session(ctx)

        return record

    async def get_last_response(
        self,
        ctx: ArtifactContext | None = None,
    ) -> ResponseRecord | None:
        if ctx is not None:
            await self.load_session(ctx)
        return self._last_response_record

    async def get_request_history(
        self,
        ctx: ArtifactContext | None = None,
    ) -> list[ResponseRecord]:
        if ctx is not None:
            await self.load_session(ctx)
        return list(self._history)

    async def get_session_cookies(
        self,
        ctx: ArtifactContext | None = None,
    ) -> dict[str, str]:
        if ctx is not None:
            await self.load_session(ctx)
        return self.get_cookies()

    async def update_cookies(
        self,
        cookies: Mapping[str, str],
        ctx: ArtifactContext | None = None,
    ) -> dict[str, str]:
        if ctx is not None:
            await self.load_session(ctx)
        self.set_cookies(cookies)
        if ctx is not None:
            await self.save_session(ctx)
        return self.get_cookies()

    async def clear_session(
        self,
        ctx: ArtifactContext | None = None,
    ) -> None:
        if ctx is not None:
            await self.load_session(ctx)
        self.clear_session_state()
        if ctx is not None:
            await self.save_session(ctx)


def http_tools(
    name: str,
    *,
    proxy: str | None = None,
    history_size: int = 20,
    timeout: float = 30.0,
    verify_ssl: bool = True,
    user_agent: str = "LLM-Agent-HTTP-Tools/1.0",
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
) -> list[ToolFn]:
    cli = HTTPClient(
        name="http_tools",
        proxy=proxy,
        history_size=history_size,
        timeout=timeout,
        verify_ssl=verify_ssl,
        user_agent=user_agent,
        retry_config=RetryConfig(
            attempts=attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on_statuses=retry_on_statuses,
        ),
    )

    async def request(
        url: str,
        method: HTTPRequestMethod = "GET",
        cookies: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: JSONLike | None = None,
        params: dict[str, Any] | None = None,
        tool_context: ToolContext | None = None,
    ) -> ResponseRecord | dict[str, str]:
        """
        Generic request tool.

        Notes:
        - cookies update the current session cookies instead of replacing them
        - persisted session is loaded before the request when tool_context is available
        - response body is parsed as JSON if possible, else string
        """
        try:
            return await cli.request(
                url=url,
                method=method,
                cookies=cookies,
                headers=headers,
                json_body=json_body,
                params=params,
                ctx=tool_context,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def fetch(
        url: str,
        tool_context: ToolContext | None = None,
    ) -> ResponseRecord | dict[str, str]:
        """
        Convenience GET request.
        """
        try:
            return await cli.request(url=url, method="GET", ctx=tool_context)
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def get_json(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        tool_context: ToolContext | None = None,
    ) -> JSONLike | dict[str, object]:
        """
        GET request that expects JSON.
        Returns only the parsed JSON body, or an error payload.
        """
        try:
            result = await cli.request(
                url=url,
                method="GET",
                params=params,
                headers=headers,
                cookies=cookies,
                ctx=tool_context,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

        body = result["body"]
        if isinstance(body, (dict, list)):
            return body

        return {
            "error": "Response is not valid JSON",
            "response": result,
        }

    async def post_json(
        url: str,
        json_body: JSONLike | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, object] | None = None,
        cookies: Mapping[str, str] | None = None,
        tool_context: ToolContext | None = None,
    ) -> JSONLike | dict[str, object]:
        """
        POST JSON and expect JSON response.
        Returns the parsed JSON body, or an error payload.
        """
        try:
            result = await cli.request(
                url=url,
                method="POST",
                json_body=json_body,
                headers=headers,
                params=params,
                cookies=cookies,
                ctx=tool_context,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

        body = result["body"]
        if isinstance(body, (dict, list)):
            return body

        return {
            "error": "Response is not valid JSON",
            "response": result,
        }

    async def clear_session(
        tool_context: ToolContext | None = None,
    ) -> dict[str, str]:
        """
        Clear current session cookies and in-memory state.
        Persists the cleared state when tool_context is available.
        """
        try:
            await cli.clear_session(ctx=tool_context)
            return {"status": "ok", "message": "Session state cleared"}
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def set_cookies(
        cookies: Mapping[str, str],
        tool_context: ToolContext | None = None,
    ) -> dict[str, object]:
        """
        Update the existing cookie jar; does not replace it.
        """
        try:
            updated = await cli.update_cookies(cookies, ctx=tool_context)
        except HTTPClientError as exc:
            return {"error": str(exc)}

        return {
            "status": "ok",
            "cookies": updated,
        }

    async def get_cookies(
        tool_context: ToolContext | None = None,
    ) -> dict[str, str] | dict[str, str]:
        """
        Return current session cookies as a plain dict.
        """
        try:
            return await cli.get_session_cookies(ctx=tool_context)
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def last_response(
        tool_context: ToolContext | None = None,
    ) -> ResponseRecord | dict[str, str] | None:
        """
        Return the last response record.
        """
        try:
            return await cli.get_last_response(ctx=tool_context)
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def request_history(
        tool_context: ToolContext | None = None,
    ) -> list[ResponseRecord] | dict[str, str]:
        """
        Return bounded request history, oldest first.
        """
        try:
            return await cli.get_request_history(ctx=tool_context)
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def extract_from_html(
        selector: str,
        url: str | None = None,
        attr: str | None = None,
        first_only: bool = False,
        strip: bool = True,
        tool_context: ToolContext | None = None,
    ) -> str | list[str | None] | None | dict[str, object]:
        """
        Extract data from HTML using CSS selectors.

        Behavior:
        - If url is provided, fetch that page first.
        - Otherwise use the body of the last response.
        - If attr is provided, extract that attribute.
        - Otherwise extract text content.

        Returns:
        - first_only=True  -> single value or None
        - first_only=False -> list of values
        """
        html: str | None = None

        if url is not None:
            try:
                result = await cli.request(url=url, method="GET", ctx=tool_context)
            except HTTPClientError as exc:
                return {"error": str(exc)}

            body = result["body"]
            if not isinstance(body, str):
                return {
                    "error": "extract_from_html expected HTML/text response, got JSON",
                    "response": result,
                }
            html = body
        else:
            try:
                last = await cli.get_last_response(ctx=tool_context)
            except HTTPClientError as exc:
                return {"error": str(exc)}

            if last is None:
                return {"error": "No last response available to extract HTML from"}

            body = last["body"]
            if not isinstance(body, str):
                return {
                    "error": "Last response body is not HTML/text",
                    "response": last,
                }
            html = body

        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)

        extracted: list[str | None] = []
        for el in elements:
            value = el.get(attr) if attr is not None else el.get_text(strip=strip)
            extracted.append(value)

        if first_only:
            return extracted[0] if extracted else None

        return extracted

    return [
        fetch,
        request,
        get_json,
        post_json,
        clear_session,
        set_cookies,
        get_cookies,
        last_response,
        request_history,
        extract_from_html,
    ]
