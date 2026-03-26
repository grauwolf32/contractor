from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Literal, Optional, Union

import httpx
from bs4 import BeautifulSoup


JSONLike = Union[Dict[str, Any], List[Any]]
ParsedBody = Union[JSONLike, str]


@dataclass
class RetryConfig:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on_statuses: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


class HTTPToolError(Exception):
    pass


class HTTPTools:
    """
    Async HTTP tool session for LLM agents.

    Features:
    - Shared cookie/session state
    - Optional proxy configured at initialization
    - Automatic retries on all requests
    - Bounded request history
    - Auto parse response body as JSON if possible, else text
    """

    def __init__(
        self,
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

        self.retry_config = retry_config or RetryConfig()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._last_response_record: Optional[Dict[str, Any]] = None

        self._client = httpx.AsyncClient(
            proxy=proxy,
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HTTPToolSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def _parse_response_body(self, response: httpx.Response) -> ParsedBody:
        """
        Return dict/list if JSON-decoding succeeds, otherwise text.
        """
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
    ) -> Dict[str, Any]:
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
    ) -> httpx.Response:
        last_error: Optional[BaseException] = None

        for attempt in range(1, self.retry_config.attempts + 1):
            try:
                response = await self._client.send(request)

                if response.status_code not in self.retry_config.retry_on_statuses:
                    return response

                last_error = HTTPToolError(
                    f"Retryable HTTP status {response.status_code} for {request.method} {request.url}"
                )

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_error = exc

            if attempt < self.retry_config.attempts:
                delay = min(
                    self.retry_config.base_delay * (2 ** (attempt - 1)),
                    self.retry_config.max_delay,
                )
                await asyncio.sleep(delay)

        if last_error is None:
            raise HTTPToolError("Request failed for an unknown reason")

        raise HTTPToolError(str(last_error)) from last_error

    async def request(
        self,
        *,
        url: str,
        method: str = "GET",
        cookies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generic request tool.

        Notes:
        - cookies update the current session cookies instead of replacing them
        - response body is parsed as JSON if possible, else string
        """
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

    async def fetch(self, url: str) -> Dict[str, Any]:
        """
        Convenience GET request.
        """
        return await self.request(url=url, method="GET")

    async def get_json(
        self,
        *,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
    ) -> JSONLike:
        """
        GET request that expects JSON.
        Raises if the response is not valid JSON object/list.
        """
        result = await self.request(
            url=url,
            method="GET",
            params=params,
            headers=headers,
            cookies=cookies,
        )
        body = result["body"]

        if isinstance(body, (dict, list)):
            return body

        raise HTTPToolError("Response is not valid JSON")

    async def post_json(
        self,
        *,
        url: str,
        json_body: Any,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        cookies: Optional[Dict[str, str]] = None,
    ) -> JSONLike:
        """
        POST JSON and expect JSON response.
        Raises if the response is not valid JSON object/list.
        """
        result = await self.request(
            url=url,
            method="POST",
            json_body=json_body,
            headers=headers,
            params=params,
            cookies=cookies,
        )
        body = result["body"]

        if isinstance(body, (dict, list)):
            return body

        raise HTTPToolError("Response is not valid JSON")

    def clear_session(self) -> Dict[str, str]:
        """
        Clear current session cookies.
        """
        self._client.cookies.clear()
        return {"status": "ok", "message": "Session cookies cleared"}

    def set_cookies(self, cookies: Dict[str, str]) -> Dict[str, Any]:
        """
        Update existing cookie jar, do not replace it.
        """
        if not isinstance(cookies, dict):
            raise ValueError("cookies must be a dictionary")

        for key, value in cookies.items():
            self._client.cookies.set(key, value)

        return {
            "status": "ok",
            "cookies": self.get_cookies(),
        }

    def get_cookies(self) -> Dict[str, str]:
        """
        Return current session cookies as a plain dict.
        """
        return dict(self._client.cookies.items())

    def last_response(self) -> Optional[Dict[str, Any]]:
        """
        Return the last response record.
        """
        return self._last_response_record

    def request_history(self) -> List[Dict[str, Any]]:
        """
        Return bounded request history, oldest first.
        """
        return list(self._history)

    async def extract_from_html(
        self,
        *,
        selector: str,
        url: Optional[str] = None,
        attr: Optional[str] = None,
        first_only: bool = False,
        strip: bool = True,
    ) -> Union[Any, List[Any]]:
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
        html: Optional[str] = None

        if url is not None:
            result = await self.fetch(url)
            body = result["body"]
            if not isinstance(body, str):
                raise HTTPToolError("extract_from_html expected HTML/text response, got JSON")
            html = body
        else:
            if self._last_response_record is None:
                raise HTTPToolError("No last response available to extract HTML from")

            body = self._last_response_record["body"]
            if not isinstance(body, str):
                raise HTTPToolError("Last response body is not HTML/text")
            html = body

        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)

        extracted: List[Any] = []
        for el in elements:
            if attr is not None:
                value = el.get(attr)
            else:
                value = el.get_text(strip=strip)
            extracted.append(value)

        if first_only:
            return extracted[0] if extracted else None

        return extracted