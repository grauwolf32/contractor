from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactAPIError,
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef


def test_client_preserves_exact_revisions_without_any_scope_selector() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            [
                json_response(
                    200,
                    {
                        "apiVersion": "contractor/v1alpha1",
                        "artifacts": [
                            {"namespace": "analysis", "name": "report"},
                            {"namespace": "inputs", "name": "source"},
                        ],
                    },
                ),
                ArtifactHTTPResponse(
                    200,
                    {
                        "content-type": "text/plain",
                        "content-length": "7",
                        "etag": '"revision-read"',
                    },
                    b"payload",
                ),
                json_response(
                    201,
                    {
                        "apiVersion": "contractor/v1alpha1",
                        "artifact": {
                            "namespace": "inputs",
                            "name": "new",
                            "revision": "revision-write",
                        },
                        "mediaType": "text/plain",
                        "size": 3,
                    },
                    etag='"revision-write"',
                ),
            ]
        )
        client = ArtifactClient("allocation-1", transport)

        listed = await client.list_artifacts()
        assert [ref.revision for ref in listed] == [None, None]
        value = await client.read_artifact(ArtifactRef(namespace="inputs", name="source"))
        assert value.artifact.revision == "revision-read"
        assert value.data == b"payload"
        assert b"payload" not in repr(value).encode()
        written = await client.write_artifact(
            ArtifactRef(namespace="inputs", name="new"),
            data=b"new",
            media_type="text/plain",
            expected_revision=None,
        )
        assert written.artifact.revision == "revision-write"

        assert [request.path for request in transport.requests] == [
            "/allocations/allocation-1/artifacts",
            "/allocations/allocation-1/artifacts/inputs/source",
            "/allocations/allocation-1/artifacts/inputs/new",
        ]
        assert all("run" not in request.path.lower() for request in transport.requests)
        assert transport.requests[-1].headers["If-None-Match"] == "*"

    asyncio.run(scenario())


def test_exact_read_and_cas_update_use_unambiguous_revision_channels() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            [
                ArtifactHTTPResponse(
                    200,
                    {"content-type": "application/json", "etag": '"revision-old"'},
                    b"{}",
                ),
                json_response(
                    200,
                    {
                        "apiVersion": "contractor/v1alpha1",
                        "artifact": {
                            "namespace": "analysis",
                            "name": "report",
                            "revision": "revision-new",
                        },
                        "mediaType": "application/json",
                        "size": 2,
                    },
                    etag='"revision-new"',
                ),
            ]
        )
        client = ArtifactClient("allocation-1", transport)
        await client.read_artifact(
            ArtifactRef(namespace="analysis", name="report", revision="revision-old")
        )
        result = await client.write_artifact(
            ArtifactRef(namespace="analysis", name="report"),
            data=b"{}",
            media_type="application/json",
            expected_revision="revision-old",
        )
        assert result.artifact.revision == "revision-new"
        assert transport.requests[0].path.endswith("?revision=revision-old")
        assert transport.requests[1].headers["If-Match"] == '"revision-old"'
        assert "If-None-Match" not in transport.requests[1].headers

    asyncio.run(scenario())


def test_api_error_is_typed_and_does_not_echo_server_message() -> None:
    async def scenario() -> None:
        recognizable = "server-message-must-not-escape"
        transport = FakeTransport(
            [
                json_response(
                    409,
                    {
                        "code": "allocation_write_fenced",
                        "message": recognizable,
                        "retryable": False,
                    },
                )
            ]
        )
        client = ArtifactClient("allocation-1", transport)
        with pytest.raises(ArtifactAPIError) as raised:
            await client.write_artifact(
                ArtifactRef(namespace="inputs", name="source"),
                data=b"new",
                media_type="text/plain",
                expected_revision="revision-old",
            )
        assert raised.value.code == "allocation_write_fenced"
        assert not raised.value.retryable
        assert recognizable not in str(raised.value)

    asyncio.run(scenario())


def test_client_rejects_versioned_write_and_mismatched_or_oversized_responses() -> None:
    async def scenario() -> None:
        client = ArtifactClient("allocation-1", FakeTransport([]))
        with pytest.raises(ValueError, match="versionless"):
            await client.write_artifact(
                ArtifactRef(namespace="analysis", name="report", revision="revision-1"),
                data=b"x",
                media_type="text/plain",
                expected_revision=None,
            )
        with pytest.raises(ValueError, match="16 MiB"):
            await client.write_artifact(
                ArtifactRef(namespace="analysis", name="report"),
                data=b"x" * (MAX_ARTIFACT_BYTES + 1),
                media_type="text/plain",
                expected_revision=None,
            )
        with pytest.raises(ValueError, match="strong ETag"):
            await client.write_artifact(
                ArtifactRef(namespace="analysis", name="report"),
                data=b"x",
                media_type="text/plain",
                expected_revision="revision-1,revision-2",
            )

        mismatch = ArtifactClient(
            "allocation-1",
            FakeTransport(
                [
                    json_response(
                        201,
                        {
                            "apiVersion": "contractor/v1alpha1",
                            "artifact": {
                                "namespace": "other",
                                "name": "report",
                                "revision": "revision-1",
                            },
                            "mediaType": "text/plain",
                            "size": 1,
                        },
                        etag='"revision-1"',
                    )
                ]
            ),
        )
        with pytest.raises(ArtifactTransportError, match="does not match"):
            await mismatch.write_artifact(
                ArtifactRef(namespace="analysis", name="report"),
                data=b"x",
                media_type="text/plain",
                expected_revision=None,
            )

    asyncio.run(scenario())


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    maximum: int


class FakeTransport:
    def __init__(self, responses: list[ArtifactHTTPResponse]) -> None:
        self.responses = responses
        self.requests: list[RecordedRequest] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        self.requests.append(RecordedRequest(method, path, dict(headers), body, max_response_bytes))
        return self.responses.pop(0)


def json_response(
    status: int,
    value: object,
    *,
    etag: str | None = None,
) -> ArtifactHTTPResponse:
    body = json.dumps(value, separators=(",", ":")).encode()
    headers = {"content-type": "application/json", "content-length": str(len(body))}
    if etag is not None:
        headers["etag"] = etag
    return ArtifactHTTPResponse(status, headers, body)
