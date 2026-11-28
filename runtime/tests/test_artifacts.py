from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes.allocation_rpc import AllocationRPCFixture
from fakes.artifact_responses import ArtifactHTTPResponse
from pydantic import ValidationError
from shared_config_defaults import MAX_ARTIFACT_BYTES

from contractor_runtime import artifacts
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactReadTarget,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    ARTIFACT_LIST_CURSOR_PATTERN,
    MAX_ARTIFACT_LIST_CURSOR_LENGTH,
    ArtifactListResult,
    ArtifactRef,
)
from contractor_runtime.toolsets.common.versions import ModelArtifacts, SelectedVersions

SHARED_LIST_CASES = (
    Path(__file__).resolve().parents[2] / "api/testdata/v1alpha1/artifact-list-cases.json"
)
CREATED_AT = "2026-09-01T10:11:12.123456Z"
UPDATED_AT = "2026-09-01T10:11:13.987654Z"
TIMESTAMP_HEADERS = {
    "x-contractor-binding-created-at": CREATED_AT,
    "x-contractor-revision-created-at": UPDATED_AT,
}


@pytest.mark.parametrize("model_adapter", [False, True])
def test_project_read_target_keeps_run_provenance_and_shared_metadata_contract(
    model_adapter,
) -> None:
    async def scenario() -> None:
        ref = ArtifactRef(namespace="analysis", name="report")
        project_ref = ArtifactRef(namespace=ref.namespace, name=ref.name, revision="project-rev")
        metadata = {
            "apiVersion": "contractor/v1alpha1",
            "artifact": project_ref.model_dump(by_alias=True),
            "mediaType": "text/plain",
            "size": 3,
            "contentCacheKey": "project-content",
        }
        transport = FakeTransport(
            [
                ArtifactHTTPResponse(
                    200,
                    {"content-type": "text/plain", "etag": '"run-rev"', **TIMESTAMP_HEADERS},
                    b"run",
                ),
                ArtifactHTTPResponse(
                    200,
                    {"content-type": "text/plain", "etag": '"project-rev"', **TIMESTAMP_HEADERS},
                    b"map",
                ),
                json_response(200, metadata),
                ArtifactHTTPResponse(204, {}, b""),
            ]
        )
        client = ArtifactClient("allocation-1", transport)
        versions = SelectedVersions()
        if model_adapter:
            client = ModelArtifacts(client, versions)
        run = await client.read_artifact(ref)
        cursor = client.observation_cursor
        target = ArtifactReadTarget(scope="project", operation="read_report")
        value = await client.read_artifact(project_ref, max_bytes=3, target=target)
        described = await client.read_artifact_metadata(project_ref, target=target)
        await client.check_artifact_access(ref, target=target)
        assert value.artifact == described.artifact == project_ref
        assert value.data == b"map" and described.size == 3
        assert value.binding_created_at == run.binding_created_at
        assert value.revision_created_at == run.revision_created_at
        assert client.known_exact_refs == (run.artifact,)
        assert client.observed_exact_refs_since(cursor) == ()
        if model_adapter:
            versions.begin_response()
            assert versions.resolve({"namespace": ref.namespace, "name": ref.name}) == run.artifact
            assert versions.base(ref.namespace, ref.name) == "run-rev"
        root = "/allocations/allocation-1/artifacts/analysis/report"
        assert [request.path for request in transport.requests] == [
            root,
            root + "?scope=project&operation=read_report&revision=project-rev",
            root + "/metadata?scope=project&operation=read_report&revision=project-rev",
            root + "/access?scope=project&operation=read_report",
        ]
        assert transport.requests[1].maximum == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scope,operation",
    [
        ("run", "read"),
        ("user", "read"),
        ("project", ""),
        ("project", "a" * 129),
        ("project", "read&scope=run"),
        ("project", None),
    ],
)
def test_artifact_read_target_rejects_unbounded_or_unsupported_options(scope, operation):
    with pytest.raises(ValueError, match="read target"):
        ArtifactReadTarget(scope=scope, operation=operation)


def test_shared_artifact_read_targets():
    cases = json.loads(
        (
            Path(__file__).parents[2] / "api/testdata/v1alpha1/artifact-read-target-cases.json"
        ).read_text()
    )
    assert len(cases["valid"]) >= 5 and len(cases["invalid"]) >= 16
    for value in cases["valid"]:
        target = ArtifactReadTarget(**value)
        assert target.scope == "project" and target.operation == value["operation"]
    for value in cases["invalid"]:
        with pytest.raises((TypeError, ValueError)):
            ArtifactReadTarget(**value)


@pytest.mark.parametrize("status,body", [(200, b""), (200, b"{}"), (204, b"{}")])
def test_artifact_access_requires_empty_204_without_recording_content(status, body):
    async def scenario():
        client = ArtifactClient(
            "allocation", FakeTransport([ArtifactHTTPResponse(status, {}, body)])
        )
        with pytest.raises(ArtifactTransportError, match="access response"):
            await client.check_artifact_access(
                ArtifactRef(namespace="analysis", name="report"),
                target=ArtifactReadTarget(scope="project", operation="read_report"),
            )
        assert client.observation_cursor == 0 and client.known_exact_refs == ()

    asyncio.run(scenario())


def test_project_exact_reads_preserve_shared_bounds_and_missing_revision_failures():
    async def scenario():
        target = ArtifactReadTarget(scope="project", operation="read_report")
        ref = ArtifactRef(namespace="analysis", name="report", revision="wanted")
        headers = {"content-type": "text/plain", "etag": '"wanted"', **TIMESTAMP_HEADERS}
        for changed, limit, error in [
            ({"etag": '"latest"'}, 3, ArtifactTransportError),
            ({"etag": 'W/"wanted"'}, 3, ArtifactTransportError),
            ({"x-contractor-revision-created-at": "invalid"}, 3, ArtifactTransportError),
            ({}, 2, ArtifactResponseLimitError),
        ]:
            client = ArtifactClient(
                "allocation",
                FakeTransport([ArtifactHTTPResponse(200, {**headers, **changed}, b"map")]),
            )
            with pytest.raises(error):
                await client.read_artifact(ref, max_bytes=limit, target=target)
            assert client.observation_cursor == 0
        failure = json_response(
            404,
            {
                "code": "artifact_not_found",
                "message": "not found",
                "retryable": False,
                "requestId": "request-missing",
            },
        )
        client = ArtifactClient("allocation", FakeTransport([failure]))
        with pytest.raises(ArtifactAPIError, match="artifact_not_found"):
            await client.read_artifact(ref, target=target)
        assert len(client.transport.requests) == 1

    asyncio.run(scenario())


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

        listed = await client.list_artifact_page(limit=10)
        assert [ref.revision for ref in listed.artifacts] == [None, None]
        assert listed.next_cursor is None
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
            "/allocations/allocation-1/artifacts?limit=10",
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
        refs = await client.list_artifact_prefix("builder", "memory.", limit=129)
        assert len(refs) == 1 and refs[0].revision is None
        assert transport.requests[0].path.endswith(
            "?namespace=builder&namePrefix=memory.&limit=129"
        )
        for args, kwargs in (
            (("builder", "memory%"), {"limit": 129}),
            (("builder", ""), {"limit": 129}),
            (("", "memory."), {"limit": 129}),
            ((None, "memory."), {"limit": 129}),
            (("builder", None), {"limit": 129}),
            (("builder", "memory."), {"limit": 0}),
            (("builder", "memory."), {"limit": 257}),
            (("builder", "memory."), {"limit": True}),
            (("builder", "memory."), {"limit": "129"}),
        ):
            with pytest.raises(ValueError):
                await client.list_artifact_prefix(*args, **kwargs)
        assert len(transport.requests) == 1
        for invalid in (
            {"artifacts": [{"namespace": "foreign", "name": "memory.one"}]},
            {"artifacts": [{"namespace": "builder", "name": "ordinary"}]},
            {"artifacts": [{"namespace": "builder", "name": "memory.one", "revision": "hidden"}]},
            {"artifacts": [{"namespace": "builder", "name": f"memory.n{i}"} for i in range(130)]},
            {"artifacts": [{"namespace": "builder", "name": "memory.one"}], "nextCursor": "abc"},
        ):
            malformed = ArtifactClient(
                "allocation-1",
                FakeTransport(
                    [json_response(200, {"apiVersion": "contractor/v1alpha1", **invalid})]
                ),
            )
            with pytest.raises(ArtifactTransportError):
                await malformed.list_artifact_prefix("builder", "memory.", limit=129)

    asyncio.run(scenario())


def test_paged_list_validates_query_cursor_and_page_without_widening_scope() -> None:
    async def scenario() -> None:
        page = [{"namespace": "builder", "name": "a"}, {"namespace": "builder", "name": "b"}]
        transport = FakeTransport(
            [
                json_response(
                    200,
                    {"apiVersion": "contractor/v1alpha1", "artifacts": page, "nextCursor": "c1"},
                ),
                json_response(200, {"apiVersion": "contractor/v1alpha1", "artifacts": page[:1]}),
            ]
        )
        client = ArtifactClient("allocation-1", transport)
        first = await client.list_artifact_page("builder", limit=2)
        assert [ref.name for ref in first.artifacts] == ["a", "b"]
        assert first.next_cursor == "c1"
        last = await client.list_artifact_page("builder", limit=2, cursor=first.next_cursor)
        assert last.next_cursor is None
        assert [request.path for request in transport.requests] == [
            "/allocations/allocation-1/artifacts?namespace=builder&limit=2",
            "/allocations/allocation-1/artifacts?namespace=builder&limit=2&cursor=c1",
        ]
        for kwargs in (
            {"limit": 0},
            {"limit": 257},
            {"limit": None},
            {"limit": 1.0},
            {"limit": 1, "cursor": ""},
            {"limit": 1, "cursor": "a=b"},
            {"limit": 1, "cursor": "a" * 513},
            {"limit": 1, "cursor": 7},
            {"namespace": "bad/namespace", "limit": 1},
        ):
            with pytest.raises(ValueError):
                await client.list_artifact_page(**kwargs)
        with pytest.raises(TypeError):
            await client.list_artifact_page("builder")  # type: ignore[call-arg]
        assert len(transport.requests) == 2
        for invalid in (
            {"artifacts": [*page, {"namespace": "builder", "name": "c"}]},
            {"artifacts": [{"namespace": "foreign", "name": "a"}]},
            {"artifacts": [{"namespace": "builder", "name": "a", "revision": "r"}]},
            {"artifacts": page[:1], "nextCursor": "c2"},
            {"artifacts": [], "nextCursor": "c2"},
            {"artifacts": page, "nextCursor": ""},
            {"artifacts": page, "nextCursor": "a" * 513},
            {"artifacts": page, "cursor": "c2"},
        ):
            malformed = ArtifactClient(
                "allocation-1",
                FakeTransport(
                    [json_response(200, {"apiVersion": "contractor/v1alpha1", **invalid})]
                ),
            )
            with pytest.raises(ArtifactTransportError):
                await malformed.list_artifact_page("builder", limit=2)

    asyncio.run(scenario())


def test_shared_list_limits_and_cursor_grammar_match_the_shared_cases() -> None:
    cases = json.loads(SHARED_LIST_CASES.read_text(encoding="utf-8"))
    assert cases["maxLimit"] == artifacts.artifact_list_max_items()
    assert cases["maxCursorLength"] == MAX_ARTIFACT_LIST_CURSOR_LENGTH
    ref = {"namespace": "analysis", "name": "report"}
    for cursor in cases["cursors"]["valid"]:
        assert ARTIFACT_LIST_CURSOR_PATTERN.fullmatch(cursor)
        ArtifactListResult.model_validate(
            {"apiVersion": "contractor/v1alpha1", "artifacts": [ref], "nextCursor": cursor}
        )
    for cursor in cases["cursors"]["invalid"]:
        assert ARTIFACT_LIST_CURSOR_PATTERN.fullmatch(cursor) is None
        with pytest.raises(ValidationError):
            ArtifactListResult.model_validate(
                {"apiVersion": "contractor/v1alpha1", "artifacts": [ref], "nextCursor": cursor}
            )


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
        client.observe_cached_read(result.artifact)
        assert client.observed_exact_refs_since(0) == (result.artifact,)
        assert len(transport.requests) == 2
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
        with pytest.raises(ArtifactTransportError, match="does not match the request"):
            await client.read_artifact(
                ArtifactRef(
                    namespace="analysis",
                    name="report",
                    revision="revision-requested",
                )
            )

    asyncio.run(scenario())


def test_metadata_read_resolves_the_exact_revision_without_the_payload() -> None:
    async def scenario() -> None:
        metadata = {
            "apiVersion": "contractor/v1alpha1",
            "artifact": {"namespace": "inputs", "name": "source", "revision": "revision-2"},
            "mediaType": "application/zip",
            "size": 70_000,
            "contentCacheKey": "version_metadata",
        }
        transport = FakeTransport([json_response(200, metadata), json_response(200, metadata)])
        client = ArtifactClient("allocation-1", transport)
        cursor = client.observation_cursor
        current = await client.read_artifact_metadata(
            ArtifactRef(namespace="inputs", name="source")
        )
        exact = await client.read_artifact_metadata(
            ArtifactRef(namespace="inputs", name="source", revision="revision-2")
        )
        assert current == exact
        assert current.artifact.revision == "revision-2"
        assert current.media_type == "application/zip" and current.size == 70_000
        path = "/allocations/allocation-1/artifacts/inputs/source/metadata"
        assert [(request.method, request.path) for request in transport.requests] == [
            ("GET", path),
            ("GET", path + "?revision=revision-2"),
        ]
        assert all(request.maximum == 1 << 20 for request in transport.requests)
        # Metadata is not a content observation.
        assert client.observed_exact_refs_since(cursor) == ()

        for value in (
            {**metadata, "artifact": {**metadata["artifact"], "name": "other"}},
            {**metadata, "artifact": {**metadata["artifact"], "revision": "revision-1"}},
            {**metadata, "size": MAX_ARTIFACT_BYTES + 1},
            {**metadata, "extra": True},
        ):
            mismatched = ArtifactClient("allocation-1", FakeTransport([json_response(200, value)]))
            with pytest.raises(ArtifactTransportError):
                await mismatched.read_artifact_metadata(
                    ArtifactRef(namespace="inputs", name="source", revision="revision-2")
                )
        missing = ArtifactClient(
            "allocation-1",
            FakeTransport(
                [
                    json_response(
                        404,
                        {
                            "code": "artifact_not_found",
                            "message": "artifact was not found",
                            "retryable": False,
                            "requestId": "request-1",
                        },
                    )
                ]
            ),
        )
        with pytest.raises(ArtifactAPIError) as caught:
            await missing.read_artifact_metadata(ArtifactRef(namespace="inputs", name="missing"))
        assert caught.value.code == "artifact_not_found" and not caught.value.retryable

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
        with pytest.raises(ValueError, match="configured Artifact limit"):
            await client.write_artifact(
                ArtifactRef(namespace="analysis", name="report"),
                data=b"x" * (MAX_ARTIFACT_BYTES + 1),
                media_type="text/plain",
                expected_revision=None,
            )
        with pytest.raises(ValueError, match="expected revision"):
            await client.write_artifact(
                ArtifactRef(namespace="analysis", name="report"),
                data=b"x",
                media_type="text/plain",
                expected_revision=" ",
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
            with pytest.raises(ArtifactTransportError, match="metadata"):
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


class FakeTransport(AllocationRPCFixture):
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
