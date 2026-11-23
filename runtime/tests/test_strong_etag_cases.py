"""One strong entity-tag grammar for the Runtime's If-Match and ETag (API-R70)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "api/testdata/v1alpha1/strong-etag-cases.json").read_text())
TIMESTAMPS = {
    "x-contractor-binding-created-at": "2026-09-01T10:11:12Z",
    "x-contractor-revision-created-at": "2026-09-01T10:11:12Z",
}
TARGET = ArtifactRef(namespace="analysis", name="report")


class RecordingTransport:
    """Answers every request with one response and records request headers."""

    def __init__(self, response: ArtifactHTTPResponse) -> None:
        self.response = response
        self.headers: list[dict[str, str]] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        del method, path, body, max_response_bytes
        self.headers.append(dict(headers))
        return self.response


def read_response(etag: str) -> ArtifactHTTPResponse:
    headers = {"content-type": "text/plain", "content-length": "1", "etag": etag, **TIMESTAMPS}
    return ArtifactHTTPResponse(200, headers, b"x")


def write_response(revision: str) -> ArtifactHTTPResponse:
    body = json.dumps(
        {
            "apiVersion": "contractor/v1alpha1",
            "artifact": {"namespace": "analysis", "name": "report", "revision": revision},
            "mediaType": "text/plain",
            "size": 1,
        }
    ).encode()
    headers = {
        "content-type": "application/json",
        "content-length": str(len(body)),
        "etag": f'"{revision}"',
        **TIMESTAMPS,
    }
    return ArtifactHTTPResponse(200, headers, body)


@pytest.mark.parametrize("case", CASES["valid"], ids=lambda case: case["header"][:40])
def test_shared_strong_etag_valid_cases_round_trip(case: dict[str, str]) -> None:
    async def scenario() -> None:
        read = await ArtifactClient(
            "allocation-1", RecordingTransport(read_response(case["header"]))
        ).read_artifact(TARGET)
        assert read.artifact.revision == case["revision"]
        transport = RecordingTransport(write_response("revision-new"))
        await ArtifactClient("allocation-1", transport).write_artifact(
            TARGET, data=b"x", media_type="text/plain", expected_revision=case["revision"]
        )
        assert transport.headers[0]["If-Match"] == case["header"]

    asyncio.run(scenario())


@pytest.mark.parametrize("header", CASES["invalidHeaders"])
def test_shared_strong_etag_invalid_headers_are_refused(header: str) -> None:
    async def scenario() -> None:
        client = ArtifactClient("allocation-1", RecordingTransport(read_response(header)))
        with pytest.raises(ArtifactTransportError, match="revision ETag"):
            await client.read_artifact(TARGET)

    asyncio.run(scenario())


@pytest.mark.parametrize("revision", CASES["invalidRevisions"])
def test_shared_strong_etag_invalid_revisions_are_never_sent(revision: str) -> None:
    async def scenario() -> None:
        transport = RecordingTransport(write_response("revision-new"))
        with pytest.raises(ValueError, match="strong ETag"):
            await ArtifactClient("allocation-1", transport).write_artifact(
                TARGET, data=b"x", media_type="text/plain", expected_revision=revision
            )
        assert transport.headers == []

    asyncio.run(scenario())
