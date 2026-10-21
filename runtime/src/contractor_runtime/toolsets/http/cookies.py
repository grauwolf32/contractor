"""Ordinary Run/Namespace cookie artifact; the Artifact API owns CAS and fencing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from http.cookiejar import Cookie
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.document_write import (
    DocumentWriteUnconfirmed,
    write_document_exact,
)
from contractor_runtime.toolsets.common.http_limits import MAX_COOKIE_SESSION_BYTES, MAX_COOKIES
from contractor_runtime.toolsets.http.errors import HTTPToolError

COOKIE_BINDING = "http-cookie-session"
COOKIE_MEDIA_TYPE = "application/vnd.contractor.http-cookie-session+json"


class _CookieRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[0, 1]
    name: str
    value: str | None
    port: str | None
    port_specified: bool
    domain: str
    domain_specified: bool
    domain_initial_dot: bool
    path: str
    path_specified: bool
    secure: bool
    expires: int | None
    discard: bool
    comment: str | None
    comment_url: str | None
    rest: dict[str, str | None]
    rfc2109: bool

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and type(value.get("version")) is not int:
            raise ValueError("invalid cookie version")
        return value

    @model_validator(mode="after")
    def validate_cookie(self) -> _CookieRow:
        for value in (self.name, self.value, self.domain, self.path, self.port):
            if value is not None and any(c in value for c in "\r\n\x00"):
                raise ValueError("invalid cookie")
        if not self.name or not self.path.startswith("/"):
            raise ValueError("invalid cookie")
        if self.domain_initial_dot and not self.domain.startswith("."):
            raise ValueError("invalid cookie domain")
        if not self.domain and self.domain_specified:
            raise ValueError("invalid cookie domain")
        if self.port_specified and self.port is None:
            raise ValueError("invalid cookie port")
        return self


class _CookieDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    target_origin: tuple[str, str, int] | None
    cookies: list[_CookieRow] = Field(max_length=MAX_COOKIES)

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("invalid cookie schema")
        return value


def encode_cookies(
    cookies: httpx.Cookies, origin: tuple[str, str, int] | None, now: float
) -> bytes:
    """Canonical bytes, including complete CookieJar matching/lifetime attributes."""
    try:
        # Cookie's internal attribute is named _rest; never serialize arbitrary
        # instance members, headers, auth defaults or Runtime settings.
        rows = []
        fields = tuple(_CookieRow.model_fields)
        for cookie in cookies.jar:
            if cookie.is_expired(now):
                continue
            rows.append(
                _CookieRow(
                    **{
                        name: cookie._rest if name == "rest" else getattr(cookie, name)
                        for name in fields
                    }
                )
            )
        rows.sort(key=lambda row: (row.domain, row.path, row.name))
        document = _CookieDocument(schema_version=1, target_origin=origin, cookies=rows)
        data = json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > MAX_COOKIE_SESSION_BYTES:
            raise ValueError("cookie document too large")
        return data
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise HTTPToolError("http_cookie_state_invalid") from None


def decode_cookies(data: bytes, origin: tuple[str, str, int] | None, now: float) -> httpx.Cookies:
    """Validate the complete bounded document before exposing any cookie."""
    try:
        if len(data) > MAX_COOKIE_SESSION_BYTES:
            raise ValueError("cookie document too large")
        raw = strict_json_loads(data)
        # JSON arrays represent the fixed origin tuple; all other validation
        # remains strict (including bool versus integer).
        if isinstance(raw, dict) and isinstance(raw.get("target_origin"), list):
            raw["target_origin"] = tuple(raw["target_origin"])
        document = _CookieDocument.model_validate(raw)
        if document.target_origin != origin:
            raise ValueError("cookie origin changed")
        cookies = httpx.Cookies()
        identities = set()
        for row in document.cookies:
            identity = (row.domain, row.path, row.name)
            if identity in identities:
                raise ValueError("duplicate cookie identity")
            identities.add(identity)
            cookie = Cookie(**row.model_dump())
            if not cookie.is_expired(now):
                cookies.jar.set_cookie(cookie)
        return cookies
    except (ValueError, TypeError, OverflowError, UnicodeError, ValidationError, RecursionError):
        raise HTTPToolError("http_cookie_state_invalid") from None


def matching_cookies(cookies: httpx.Cookies, url: str) -> httpx.Cookies:
    """Enforce host-only cookies as well as CookieJar's domain/path/Secure policy.

    Python's default Netscape policy accepts host-only cookies on subdomains;
    filter those before httpx applies the remaining RFC matching rules.
    Explicit unscoped http_session_set cookies retain their documented scope.
    """
    host = urlsplit(url).hostname or ""
    result = httpx.Cookies()
    for cookie in cookies.jar:
        if (
            cookie.domain
            and not cookie.domain_specified
            and cookie.domain not in {host, host + ".local"}
        ):
            continue
        result.jar.set_cookie(cookie)
    return result


async def _joined[T](operation: Awaitable[T]) -> T:
    """Cancellation cannot release the session owner while an Artifact PUT lives."""
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        # Retrieve a possible exception without leaking its content. The inner
        # operation records unconfirmed state before its bounded task ends.
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


class CookieCheckpoint:
    def __init__(
        self,
        client: ArtifactClient,
        namespace: str,
        origin: tuple[str, str, int] | None,
        now: Callable[[], float],
        timeout: float,
    ) -> None:
        self._client = client
        self._target = ArtifactRef(namespace=namespace, name=COOKIE_BINDING)
        self._origin = origin
        self._now = now
        self._timeout = timeout
        self._revision: str | None = None
        self._data: bytes | None = None
        self._pending: bytes | None = None
        self._error: HTTPToolError | None = None

    @property
    def unconfirmed(self) -> bool:
        return self._error is not None

    def confirmed_cookies(self) -> httpx.Cookies:
        """Local mirror after a joined cancellation, including a committed clear."""
        if self._data is None:
            return httpx.Cookies()
        return decode_cookies(self._data, self._origin, self._now())

    def require_confirmed(self) -> None:
        if self._error is not None:
            raise self._error

    async def restore(self, *, deadline: float | None = None) -> httpx.Cookies:
        return await _joined(self._restore(deadline))

    async def _restore(self, deadline: float | None) -> httpx.Cookies:
        try:
            limit = asyncio.get_running_loop().time() + self._timeout
            if deadline is not None:
                limit = min(limit, deadline)
            async with asyncio.timeout_at(limit):
                for attempt in range(2):
                    try:
                        value = await self._client.read_artifact(
                            self._target,
                            max_bytes=MAX_COOKIE_SESSION_BYTES,
                        )
                        break
                    except ArtifactAPIError as error:
                        if error.code == "artifact_not_found":
                            self._data = None
                            self._revision = None
                            return httpx.Cookies()
                        if not error.retryable or attempt:
                            raise
                    except ArtifactResponseLimitError:
                        raise
                    except ArtifactTransportError:
                        if attempt:
                            raise
                if value.media_type != COOKIE_MEDIA_TYPE:
                    raise HTTPToolError("http_cookie_state_invalid")
                cookies = await threads.to_thread_until_done(
                    decode_cookies,
                    value.data,
                    self._origin,
                    self._now(),
                    name="http-cookie-restore",
                )
                self._data, self._revision = value.data, value.artifact.revision
                return cookies
        except ArtifactResponseLimitError:
            raise HTTPToolError("http_cookie_state_invalid") from None
        except (ArtifactAPIError, ArtifactTransportError, TimeoutError) as error:
            raise _storage_error(error) from None

    async def save(
        self, cookies: httpx.Cookies, *, deadline: float | None = None, force: bool = False
    ) -> None:
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self._timeout
        data = await threads.to_thread_until_done(
            encode_cookies,
            cookies,
            self._origin,
            self._now(),
            name="http-cookie-encode",
        )
        if data == self._data or (not force and self._data is None and not len(cookies)):
            return
        self._pending = data
        await _joined(self._save(data, deadline))

    async def _save(self, data: bytes, deadline: float | None) -> None:
        try:
            limit = asyncio.get_running_loop().time() + self._timeout
            if deadline is not None:
                limit = min(limit, deadline)
            async with asyncio.timeout_at(limit):
                result = await write_document_exact(
                    self._client,
                    self._target,
                    data=data,
                    media_type=COOKIE_MEDIA_TYPE,
                    expected_revision=self._revision,
                    max_bytes=MAX_COOKIE_SESSION_BYTES,
                    reload_tool="http_session_get",
                    read_tool="http_session_get",
                )
                self._data, self._revision = data, result.artifact.revision
                self._pending, self._error = None, None
        except (ArtifactAPIError, ArtifactTransportError, ToolInputError, TimeoutError) as error:
            self._error = _storage_error(error)
            raise self._error from None

    async def recover(self) -> httpx.Cookies:
        # Read the authoritative binding. Adopt identical committed bytes;
        # retry only the exact pending PUT when the expected revision is still
        # current. A newer different revision is never overwritten.
        pending, previous = self._pending, self._revision
        deadline = asyncio.get_running_loop().time() + self._timeout
        cookies = await self.restore(deadline=deadline)
        if pending is not None and self._data != pending:
            if self._revision != previous:
                self._pending = None
                raise HTTPToolError("http_cookie_state_conflict")
            await _joined(self._save(pending, deadline))
            cookies = decode_cookies(pending, self._origin, self._now())
        self._pending, self._error = None, None
        return cookies

    def reject(self, error: HTTPToolError) -> None:
        self._error = error

    def close(self) -> None:
        self._data = self._pending = None
        self._origin = None


def _storage_error(error: Exception) -> HTTPToolError:
    if isinstance(error, DocumentWriteUnconfirmed) and error.conflict:
        return HTTPToolError("http_cookie_state_conflict")
    if isinstance(error, ArtifactAPIError) and not error.retryable:
        return HTTPToolError("http_cookie_write_denied")
    return HTTPToolError("http_cookie_storage_unavailable")
