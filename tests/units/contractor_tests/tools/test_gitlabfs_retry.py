"""Retry-policy regressions for the GitLab filesystem loader."""

from __future__ import annotations

from types import SimpleNamespace

import aiohttp
import pytest
from yarl import URL

from contractor.tools.fs.gitlabfs import GitlabAsyncLoader, GitlabFileSystemSettings


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status
        self.request_info = SimpleNamespace(real_url=URL("https://gitlab.test/api"))
        self.history = ()
        self.released = False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=self.request_info,
                history=self.history,
                status=self.status,
                message="failure",
            )

    async def text(self) -> str:
        return "failure"

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0
        self.responses: list[_Response] = []

    async def request(self, *args, **kwargs) -> _Response:
        self.calls += 1
        response = _Response(self.status)
        self.responses.append(response)
        return response


class _TruncatedResponse(_Response):
    async def text(self) -> str:
        raise aiohttp.ClientPayloadError("truncated response")


class _TruncatedSession(_Session):
    async def request(self, *args, **kwargs) -> _Response:
        self.calls += 1
        response = _TruncatedResponse(self.status)
        self.responses.append(response)
        return response


def _loader(*, max_retries: int = 2) -> GitlabAsyncLoader:
    settings = GitlabFileSystemSettings(
        max_retries=max_retries,
        retry_backoff_factor=0,
    )
    return GitlabAsyncLoader(settings=settings, project_id="group/project")


@pytest.mark.asyncio
async def test_permanent_404_is_not_retried():
    session = _Session(404)

    with pytest.raises(aiohttp.ClientResponseError, match="failure"):
        await _loader()._request_with_retry(
            session, "GET", "https://gitlab.test/missing"
        )

    assert session.calls == 1


@pytest.mark.asyncio
async def test_configured_retry_status_still_retries():
    session = _Session(503)

    with pytest.raises(aiohttp.ClientResponseError, match="failure"):
        await _loader()._request_with_retry(
            session, "GET", "https://gitlab.test/unavailable"
        )

    assert session.calls == 3
    assert all(response.released for response in session.responses)


@pytest.mark.asyncio
async def test_retry_status_payload_failure_is_released_and_retried():
    session = _TruncatedSession(503)

    with pytest.raises(aiohttp.ClientPayloadError, match="truncated response"):
        await _loader()._request_with_retry(
            session, "GET", "https://gitlab.test/truncated"
        )

    assert session.calls == 3
    assert all(response.released for response in session.responses)
