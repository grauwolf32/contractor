from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactAPIError,
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef

CREATED_AT = "2026-09-01T10:11:12.123456Z"
UPDATED_AT = "2026-09-01T10:11:13.987654Z"
TIMESTAMP_HEADERS = {
    "x-contractor-binding-created-at": CREATED_AT,
    "x-contractor-revision-created-at": UPDATED_AT,
}


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
                        **TIMESTAMP_HEADERS,
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
        assert value.binding_created_at == datetime(2026, 9, 1, 10, 11, 12, 123456, UTC)
        assert value.revision_created_at == datetime(2026, 9, 1, 10, 11, 13, 987654, UTC)
        assert b"payload" not in repr(value).encode()
        written = await client.write_artifact(
            ArtifactRef(namespace="inputs", name="new"),
            data=b"new",
            media_type="text/plain",
            expected_revision=None,
        )
        assert written.artifact.revision == "revision-write"
        assert written.api_version == "contractor/v1alpha1"
        assert written.binding_created_at == value.binding_created_at
        assert written.revision_created_at == value.revision_created_at
        assert set(written.model_dump(by_alias=True)) == {
            "apiVersion",
            "artifact",
            "mediaType",
            "size",
        }

        assert [request.path for request in transport.requests] == [
            "/allocations/allocation-1/artifacts",
            "/allocations/allocation-1/artifacts/inputs/source",
            "/allocations/allocation-1/artifacts/inputs/new",
        ]
        assert all("run" not in request.path.lower() for request in transport.requests)
        assert transport.requests[-1].headers["If-None-Match"] == "*"

    asyncio.run(scenario())


def test_filtered_list_validates_query_and_response_without_widening_scope() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            [
                json_response(
                    200,
                    {
                        "apiVersion": "contractor/v1alpha1",
                        "artifacts": [{"namespace": "builder", "name": "memory.one"}],
                    },
                )
            ]
        )
        client = ArtifactClient("allocation-1", transport)
        refs = await client.list_artifacts("builder", name_prefix="memory.", limit=129)
        assert len(refs) == 1 and refs[0].revision is None
        assert transport.requests[0].path.endswith(
            "?namespace=builder&namePrefix=memory.&limit=129"
        )
        for kwargs in (
            {"name_prefix": "memory.", "limit": 129},
            {"namespace": "builder", "name_prefix": "memory."},
            {"namespace": "builder", "limit": 129},
            {"namespace": "builder", "name_prefix": "memory%", "limit": 129},
            {"namespace": "builder", "name_prefix": "memory.", "limit": 0},
            {"namespace": "builder", "name_prefix": "memory.", "limit": 257},
            {"namespace": "builder", "name_prefix": "memory.", "limit": True},
        ):
            with pytest.raises(ValueError):
                await client.list_artifacts(**kwargs)
        assert len(transport.requests) == 1
        for invalid in (
            [{"namespace": "foreign", "name": "memory.one"}],
            [{"namespace": "builder", "name": "ordinary"}],
            [{"namespace": "builder", "name": "memory.one", "revision": "hidden"}],
            [{"namespace": "builder", "name": f"memory.n{i}"} for i in range(130)],
        ):
            malformed = ArtifactClient(
                "allocation-1",
                FakeTransport(
                    [
                        json_response(
                            200, {"apiVersion": "contractor/v1alpha1", "artifacts": invalid}
                        )
                    ]
                ),
            )
            with pytest.raises(ArtifactTransportError):
                await malformed.list_artifacts("builder", name_prefix="memory.", limit=129)

    asyncio.run(scenario())


def test_exact_read_and_cas_update_use_unambiguous_revision_channels() -> None:
    async def scenario() -> None:
        transport = FakeTransport(
            [
                ArtifactHTTPResponse(
                    200,
                    {
                        "content-type": "application/json",
                        "etag": '"revision-old"',
                        **TIMESTAMP_HEADERS,
                    },
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
        assert client.observation_cursor == 2
        assert [ref.revision for ref in client.observed_exact_refs_since(0)] == [
            "revision-old",
            "revision-new",
        ]
        assert client.known_exact_refs == (result.artifact,)
        client.clear_observations()
        assert client.observation_cursor == 0
        assert client.observed_exact_refs_since(0) == ()
        assert client.known_exact_refs == (result.artifact,)
        assert transport.requests[0].path.endswith("?revision=revision-old")
        assert transport.requests[1].headers["If-Match"] == '"revision-old"'
        assert "If-None-Match" not in transport.requests[1].headers

    asyncio.run(scenario())


def test_exact_read_rejects_a_response_for_a_different_revision() -> None:
    async def scenario() -> None:
        client = ArtifactClient(
            "allocation-1",
            FakeTransport(
                [
                    ArtifactHTTPResponse(
                        200,
                        {
                            "content-type": "application/json",
                            "etag": '"revision-other"',
                            **TIMESTAMP_HEADERS,
                        },
                        b"{}",
                    )
                ]
            ),
        )
        with pytest.raises(ArtifactTransportError, match="requested revision"):
            await client.read_artifact(
                ArtifactRef(
                    namespace="analysis",
                    name="report",
                    revision="revision-requested",
                )
            )

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
                        "requestId": "artifact-request-1",
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
        assert raised.value.request_id == "artifact-request-1"
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
        with pytest.raises(ValueError, match="64 MiB"):
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


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("2026-09-01T10:11:12Z", True),
        ("2026-09-01T10:11:12.123456789Z", True),
        ("2026-09-01T10:11:12+00:00", False),
        ("2026-09-01T10:11:12.1234567890Z", False),
        ("2026-02-30T10:11:12Z", False),
        ("", False),
    ],
)
def test_artifact_timestamp_headers_are_strict_utc_rfc3339(value: str, valid: bool) -> None:
    async def scenario() -> None:
        headers = {
            "content-type": "text/plain",
            "etag": '"revision-read"',
            "x-contractor-binding-created-at": value,
            "x-contractor-revision-created-at": value,
        }
        client = ArtifactClient(
            "allocation-1", FakeTransport([ArtifactHTTPResponse(200, headers, b"x")])
        )
        if not valid:
            with pytest.raises(ArtifactTransportError, match="timestamp metadata"):
                await client.read_artifact(ArtifactRef(namespace="inputs", name="source"))
            return
        result = await client.read_artifact(ArtifactRef(namespace="inputs", name="source"))
        assert result.binding_created_at.tzinfo is UTC
        assert result.binding_created_at == result.revision_created_at
        if "." in value:
            assert result.binding_created_at.microsecond == 123456

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
    headers.update(TIMESTAMP_HEADERS)
    if etag is not None:
        headers["etag"] = etag
    return ArtifactHTTPResponse(status, headers, body)
