"""Ordinary Run/Namespace cookie artifact; the Artifact API owns CAS and fencing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.http_session import (
    MAX_COOKIE_SESSION_BYTES,
    CookieStateError,
)
from contractor_runtime.core.http_session import (
    decode_cookies as _decode_cookies,
)
from contractor_runtime.core.http_session import (
    encode_cookies as _encode_cookies,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.document_write import (
    DocumentWriteUnconfirmed,
    write_document_exact,
)
from contractor_runtime.toolsets.http.errors import HTTPToolError

COOKIE_BINDING = "http-cookie-session"
COOKIE_MEDIA_TYPE = "application/vnd.contractor.http-cookie-session+json"


def encode_cookies(cookies, origin, now):
    try:
        return _encode_cookies(cookies, origin, now)
    except CookieStateError:
        raise HTTPToolError("http_cookie_state_invalid") from None


def decode_cookies(data, origin, now):
    try:
        return _decode_cookies(data, origin, now)
    except CookieStateError:
        raise HTTPToolError("http_cookie_state_invalid") from None


async def _joined[T](operation: Awaitable[T]) -> T:
    """Cancellation cannot release the session owner while an Artifact PUT lives."""
    task = asyncio.create_task(operation)
    cancelled = await join_owned_task(task)
    if cancelled:
        # Retrieve a possible exception without leaking its content. The inner
        # operation records unconfirmed state before its bounded task ends.
        threads.consume_task_exception(task)
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
