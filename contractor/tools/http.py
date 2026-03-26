from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional, TypeAlias

import httpx
from bs4 import BeautifulSoup


JSONLike: TypeAlias = dict[str, Any] | list[Any]
ParsedBody: TypeAlias = JSONLike | str
ResponseRecord: TypeAlias = dict[str, Any]


@dataclass
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
    """

    def __init__(
        self,
        name: str,
        *,
        proxy: Optional[str] = None,
        history_size: int = 20,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        user_agent: str = "LLM-Agent-HTTP-Tools/1.0",
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be > 0")

        self.name = name
        self.retry_config = retry_config or RetryConfig()
        self._history: Deque[ResponseRecord] = deque(maxlen=history_size)
        self._last_response_record: Optional[ResponseRecord] = None

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

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    def set_cookies(self, cookies: dict[str, str]) -> None:
        for key, value in cookies.items():
            self._client.cookies.set(key, value)

    def get_cookies(self) -> dict[str, str]:
        return dict(self._client.cookies.items())

    def _parse_response_body(self, response: httpx.Response) -> ParsedBody:
        text = response.text
        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            try:
                return response.json()
            except Exception:
                return text

        try:
            return response.json()
        except Exception:
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

    async def _send_with_retries(self, request: httpx.Request) -> httpx.Response:
        last_error: Optional[BaseException] = None

        for attempt in range(1, self.retry_config.attempts + 1):
            try:
                response = await self._client.send(request)

                if response.status_code not in self.retry_config.retry_on_statuses:
                    return response

                last_error = HTTPClientError(
                    f"Retryable HTTP status {response.status_code} for "
                    f"{request.method} {request.url}"
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
        method: str = "GET",
        cookies: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[Any] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> ResponseRecord:
        if cookies:
            self.set_cookies(cookies)

        request = self._client.build_request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_body,
            params=params,
        )

        response = await self._send_with_retries(request)
        parsed_body = self._parse_response_body(response)
        record = self._make_response_record(
            response,
            parsed_body,
            attempt_count=self.retry_config.attempts,
        )

        self._last_response_record = record
        self._history.append(record)
        return record


def http_tools(
    proxy: Optional[str] = None,
    history_size: int = 20,
    timeout: float = 30.0,
    verify_ssl: bool = True,
    user_agent: str = "LLM-Agent-HTTP-Tools/1.0",
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
) -> list[Callable[..., Any]]:
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
        method: str = "GET",
        cookies: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[Any] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> ResponseRecord:
        try:
            return await cli.request(
                url=url,
                method=method,
                cookies=cookies,
                headers=headers,
                json_body=json_body,
                params=params,
            )
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def fetch(url: str) -> ResponseRecord:
        try:
            return await cli.request(url=url, method="GET")
        except HTTPClientError as exc:
            return {"error": str(exc)}

    async def get_json(
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        cookies: Optional[dict[str, str]] = None,
    ) -> JSONLike:
        try:
            result = await cli.request(
                url=url,
                method="GET",
                params=params,
                headers=headers,
                cookies=cookies,
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
        json_body: Any,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        cookies: Optional[dict[str, str]] = None,
    ) -> JSONLike:
        try:
            result = await cli.request(
                url=url,
                method="POST",
                json_body=json_body,
                headers=headers,
                params=params,
                cookies=cookies,
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

    def clear_session() -> dict[str, str]:
        cli._client.cookies.clear()
        return {"status": "ok", "message": "Session cookies cleared"}

    def set_cookies(cookies: dict[str, str]) -> dict[str, Any]:
        if not isinstance(cookies, dict):
            return {"error": "cookies must be a dictionary"}

        cli.set_cookies(cookies)
        return {
            "status": "ok",
            "cookies": cli.get_cookies(),
        }

    def get_cookies() -> dict[str, str]:
        return cli.get_cookies()

    def last_response() -> Optional[ResponseRecord]:
        return cli._last_response_record

    def request_history() -> list[ResponseRecord]:
        return list(cli._history)

    async def extract_from_html(
        selector: str,
        url: Optional[str] = None,
        attr: Optional[str] = None,
        first_only: bool = False,
        strip: bool = True,
    ) -> Any | list[Any] | dict[str, Any] | None:
        html: Optional[str] = None

        if url is not None:
            try:
                result = await cli.request(url=url, method="GET")
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
            if cli._last_response_record is None:
                return {"error": "No last response available to extract HTML from"}

            body = cli._last_response_record["body"]
            if not isinstance(body, str):
                return {
                    "error": "Last response body is not HTML/text",
                    "response": cli._last_response_record,
                }
            html = body

        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)

        extracted: list[Any] = []
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
