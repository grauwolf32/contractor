from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import contractor_runtime.toolsets.run_artifacts.tools as run_artifacts
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactListResult,
    ArtifactRef,
    ArtifactWriteResult,
    RuntimeSettings,
)
from contractor_runtime.contracts.artifacts import is_reserved_memory_binding
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.telemetry.metrics import MAX_METRIC_TOOL_CALLS
from contractor_runtime.toolsets.common.artifact_visibility import is_model_hidden_binding
from contractor_runtime.toolsets.run_artifacts.tools import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-tool-secret"


def test_memory_prefix_policy_is_namespace_aware() -> None:
    assert is_reserved_memory_binding("analysis", "memory.note")
    assert not is_reserved_memory_binding("analysis", "report")
    for namespace in ("inputs", "outputs", "skills"):
        assert not is_reserved_memory_binding(namespace, "memory.note")


def test_factory_constructs_only_explicitly_selected_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_write_tool(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unselected write_artifact was instantiated")

    monkeypatch.setattr(run_artifacts, "WriteArtifactTool", unexpected_write_tool)

    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["list_artifacts", "read_artifact"])
        assert set(tools) == {"list_artifacts", "read_artifact"}
        assert "write_artifact" not in tools

        listed = await tools["list_artifacts"]("inputs")
        assert listed == {
            "artifacts": [
                {"namespace": "inputs", "name": "source"},
                {"namespace": "inputs", "name": "memory.allowed_input"},
            ],
            "nextCursor": None,
        }
        read = await tools["read_artifact"]("inputs", "source")
        assert read["artifact"]["revision"] == "revision-read"
        assert base64.b64decode(read["dataBase64"]) == b"payload"
        assert state.metrics.counters["tool_calls"] == 2
        assert all(entry.result_size_bytes is not None for entry in state.metrics.tool_calls)
        assert "dataBase64" not in repr(state.metrics.tool_calls)

    asyncio.run(scenario())


def test_read_artifact_pages_large_payloads() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        chunk = run_artifacts.MAX_READ_CHUNK_BYTES
        client.read_data = b"0123456789abcdef" * (chunk // 16) * 2 + b"tail"
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["read_artifact"])

        first = await tools["read_artifact"]("inputs", "source")
        revision = first["artifact"]["revision"]
        assert first["size"] == len(client.read_data)
        assert first["offset"] == 0 and first["length"] == chunk
        assert len(base64.b64decode(first["dataBase64"])) == chunk
        assert first["hasMore"] and first["nextOffset"] == chunk

        collected = base64.b64decode(first["dataBase64"])
        offset = first["nextOffset"]
        while offset is not None:
            page = await tools["read_artifact"]("inputs", "source", revision, offset=offset)
            collected += base64.b64decode(page["dataBase64"])
            offset = page["nextOffset"]
        assert collected == client.read_data
        assert not page["hasMore"] and page["length"] == 4
        assert client.read_calls == 1

        window = await tools["read_artifact"]("inputs", "source", revision, offset=2, length=3)
        assert base64.b64decode(window["dataBase64"]) == b"234"
        assert window["hasMore"]
        assert client.read_calls == 1

        for arguments in ({"length": 0}, {"length": chunk + 1}, {"offset": -1}):
            with pytest.raises(ToolInputError):
                await tools["read_artifact"]("inputs", "source", **arguments)
        with pytest.raises(ToolInputError, match="offset exceeds"):
            await tools["read_artifact"]("inputs", "source", offset=len(client.read_data) + 1)

    asyncio.run(scenario())


def test_read_artifact_rejects_binary_artifacts_without_returning_bytes() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.read_data = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\xff\xfe binary"
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["read_artifact"])

        with pytest.raises(ToolInputError, match="binary") as caught:
            await tools["read_artifact"]("inputs", "source")
        assert "filesystem or source tools" in str(caught.value)
        assert "UEsDB" not in repr(state.metrics.tool_calls)

    asyncio.run(scenario())


def test_metrics_bound_events_and_redact_credential_bearing_urls() -> None:
    state = WorkerState()
    for index in range(MAX_METRIC_TOOL_CALLS + 3):
        state.metrics.record_tool_call(
            "probe",
            arguments={
                "callback": f"https://example.test/path?access_token={index}",
                "plain": "https://example.test/path",
            },
        )

    assert len(state.metrics.tool_calls) == MAX_METRIC_TOOL_CALLS
    assert state.metrics.truncated
    assert state.metrics.tool_calls[0].call_id == "tool-00000004"
    assert state.metrics.tool_calls[0].arguments == {
        "callback": "[REDACTED_URL]",
        "plain": "https://example.test/path",
    }


def test_write_tool_preserves_exact_revision_and_redacts_payload_metrics() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["write_artifact"])
        encoded = base64.b64encode(f"payload-{SECRET}".encode()).decode()

        result = await tools["write_artifact"](
            "inputs",
            "source",
            "text/plain",
            encoded,
            "revision-old",
        )
        assert result["artifact"]["revision"] == "revision-write"
        assert client.write_expected_revision == "revision-old"
        metric = state.metrics.tool_calls[0]
        assert metric.arguments is not None
        assert metric.arguments["data_base64"] == {
            "redacted": True,
            "size": len(encoded),
        }
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_tool_error_metrics_are_bounded_typed_and_secret_free() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient(
            write_error=ArtifactAPIError(403, "artifact_access_denied", False)
        )
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["write_artifact"])

        with pytest.raises(ArtifactAPIError):
            await tools["write_artifact"](
                "outputs",
                "result",
                "text/plain",
                base64.b64encode(SECRET.encode()).decode(),
            )
        assert state.metrics.counters["tool_errors"] == 1
        assert state.metrics.errors[0].code == "tool_write_artifact_failed"
        assert state.metrics.tool_calls[0].error is not None
        assert state.metrics.tool_calls[0].error.code == "artifact_access_denied"
        assert state.metrics.tool_calls[0].error.retryable is False
        assert "ArtifactAPIError" in state.metrics.tool_calls[0].error.message
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_memory_artifacts_are_hidden_from_generic_list_read_write_and_known_refs() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(
            factory, state, ["list_artifacts", "read_artifact", "write_artifact"]
        )

        assert await tools["list_artifacts"]("analysis") == {
            "artifacts": [{"namespace": "analysis", "name": "report"}],
            "nextCursor": None,
        }
        calls_before = (client.read_calls, client.write_calls)
        list_calls_before = client.list_calls
        for hidden_namespace in ("skills", "contractor-system"):
            with pytest.raises(ValueError, match="purpose-specific"):
                await tools["list_artifacts"](hidden_namespace)
        assert client.list_calls == list_calls_before
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("skills", "likec4", "revision-skill")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"]("skills", "likec4", "application/octet-stream", "")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("analysis", "memory.hidden", "revision-hidden")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"](
                "analysis",
                "memory.hidden",
                "application/json",
                base64.b64encode(b"hidden").decode(),
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("analysis", "http.body.hidden", "revision-http")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"](
                "analysis",
                "http.body.hidden",
                "application/json",
                base64.b64encode(b"hidden").decode(),
            )
        assert (client.read_calls, client.write_calls) == calls_before
        observed = {
            (ref.namespace, ref.name) for tool in tools.values() for ref in tool.known_exact_refs
        }
        assert ("analysis", "memory.hidden") not in observed
        assert ("analysis", "http.body.hidden") not in observed
        assert ("skills", "likec4") not in observed
        assert ("inputs", "memory.allowed_input") in observed

    asyncio.run(scenario())


def test_reserved_skills_namespace_is_hidden_before_generic_client_access() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(
            factory, state, ["list_artifacts", "read_artifact", "write_artifact"]
        )

        calls_before = (client.list_calls, client.read_calls, client.write_calls)
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["list_artifacts"]("skills")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"]("skills", "likec4", "revision-skill")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"]("skills", "likec4", "application/octet-stream", "")
        assert (client.list_calls, client.read_calls, client.write_calls) == calls_before
        assert all(
            ref.namespace != "skills" for tool in tools.values() for ref in tool.known_exact_refs
        )

    asyncio.run(scenario())


SHARED_LIST_CASES = (
    Path(__file__).resolve().parents[2] / "api/testdata/v1alpha1/artifact-list-cases.json"
)


def test_model_hidden_bindings_match_the_shared_cases() -> None:
    cases = json.loads(SHARED_LIST_CASES.read_text(encoding="utf-8"))["bindings"]
    assert cases
    for case in cases:
        assert is_model_hidden_binding(case["namespace"], case["name"]) is case["hidden"], case


class PagingTransport:
    """Private list endpoint that pages a fixed binding set like the Server."""

    def __init__(self, refs: list[ArtifactRef], *, leak: ArtifactRef | None = None) -> None:
        self.refs = sorted(
            (ref for ref in refs if not is_model_hidden_binding(ref.namespace, ref.name)),
            key=lambda ref: (ref.namespace, ref.name),
        )
        self.leak = leak
        self.paths: list[str] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        del headers, body, max_response_bytes
        assert method == "GET"
        self.paths.append(path)
        query = parse_qs(urlsplit(path).query)
        limit = int(query["limit"][0])
        namespace = query.get("namespace", [None])[0]
        start = int(query["cursor"][0][1:]) if "cursor" in query else 0
        selected = [ref for ref in self.refs if namespace in {None, ref.namespace}]
        page = selected[start : start + limit]
        if self.leak is not None:
            page = [self.leak, *page[1:]]
        payload: dict[str, object] = {
            "apiVersion": API_VERSION,
            "artifacts": [ref.model_dump(by_alias=True, exclude_none=True) for ref in page],
        }
        if start + limit < len(selected):
            payload["nextCursor"] = f"p{start + limit}"
        return ArtifactHTTPResponse(
            200, {"content-type": "application/json"}, json.dumps(payload).encode()
        )


def test_list_artifacts_pages_through_every_visible_binding_once() -> None:
    async def scenario() -> None:
        refs = [ArtifactRef(namespace="builder", name=f"report_{i:04d}") for i in range(250)]
        refs += [ArtifactRef(namespace="builder", name=f"http.body.{i:04d}") for i in range(300)]
        refs += [ArtifactRef(namespace="inputs", name="memory.allowed")]
        transport = PagingTransport(refs)
        client = ArtifactClient("allocation-1", transport)
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["list_artifacts"])

        listed: list[tuple[str, str]] = []
        cursor = None
        while True:
            page = await tools["list_artifacts"](cursor=cursor)
            assert set(page) == {"artifacts", "nextCursor"}
            assert len(page["artifacts"]) <= run_artifacts.LIST_PAGE_SIZE
            listed += [(ref["namespace"], ref["name"]) for ref in page["artifacts"]]
            cursor = page["nextCursor"]
            if cursor is None:
                break
        assert len(listed) == len(set(listed)) == 251
        assert ("inputs", "memory.allowed") in listed
        assert not any(name.startswith("http.body.") for _, name in listed)
        assert len(transport.paths) == 3
        assert all(f"limit={run_artifacts.LIST_PAGE_SIZE}" in path for path in transport.paths)
        assert len(state.metrics.tool_calls) == 3

        for invalid in ("", "a b", "a=", "x" * 513, 7):
            with pytest.raises(ToolInputError):
                await tools["list_artifacts"](cursor=invalid)
        for invalid in ("", "bad/namespace", 7):
            with pytest.raises(ToolInputError):
                await tools["list_artifacts"](invalid)
        assert len(transport.paths) == 3

    asyncio.run(scenario())


def test_list_artifacts_rejects_a_hidden_binding_from_the_server() -> None:
    async def scenario() -> None:
        leak = ArtifactRef(namespace="builder", name="memory.leaked")
        transport = PagingTransport([ArtifactRef(namespace="builder", name="report")], leak=leak)
        client = ArtifactClient("allocation-1", transport)
        tools = await create_tools(
            RunArtifactsToolsetFactory(lambda _allocation, _settings: client),
            WorkerState(),
            ["list_artifacts"],
        )
        with pytest.raises(ArtifactTransportError, match="hidden"):
            await tools["list_artifacts"]("builder")

    asyncio.run(scenario())


def test_list_artifacts_description_matches_the_returned_fields() -> None:
    description = run_artifacts.ListArtifactsTool.description
    assert "revision" in description and "nextCursor" in description
    assert "current revision" not in description.split("Returns:")[1]
    assert f"Up to {run_artifacts.LIST_PAGE_SIZE} artifact references" in description
    assert "{" not in description and "}" not in description


async def create_tools(
    factory: RunArtifactsToolsetFactory,
    state: WorkerState,
    selected: list[str],
) -> dict[str, object]:
    settings = RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )
    workspace_path = Path("/tmp/contractor-tool-test/allocation-test")
    result = await factory.create_selected(
        selected=selected,
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="builder",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=workspace_path.parent, path=workspace_path),
        state=state,
    )
    return dict(result)


class FakeArtifactClient:
    def __init__(self, *, write_error: Exception | None = None) -> None:
        self.write_error = write_error
        self.read_data = b"payload"
        self.write_expected_revision: str | None = None
        self.read_calls = 0
        self.write_calls = 0
        self.list_calls = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return (
            ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            ArtifactRef(namespace="inputs", name="memory.allowed_input", revision="revision-input"),
            ArtifactRef(namespace="analysis", name="report", revision="revision-report"),
            ArtifactRef(namespace="analysis", name="memory.hidden", revision="revision-hidden"),
            ArtifactRef(namespace="analysis", name="http.body.hidden", revision="revision-http"),
            ArtifactRef(namespace="skills", name="likec4", revision="revision-skill"),
        )

    async def list_artifact_page(
        self, namespace: str | None = None, *, limit: int, cursor: str | None = None
    ) -> ArtifactListResult:
        # Like the Server: hidden bindings never reach a model-visible page.
        self.list_calls += 1
        assert limit == run_artifacts.LIST_PAGE_SIZE and cursor is None
        if namespace == "inputs":
            refs = [
                ArtifactRef(namespace="inputs", name="source"),
                ArtifactRef(namespace="inputs", name="memory.allowed_input"),
            ]
        elif namespace == "analysis":
            refs = [ArtifactRef(namespace="analysis", name="report")]
        else:
            raise AssertionError(f"unexpected namespace {namespace!r}")
        return ArtifactListResult(api_version=API_VERSION, artifacts=refs)

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.read_calls += 1
        assert ref.namespace == "inputs" and ref.name == "source"
        assert ref.revision in {None, "revision-read"}
        return ArtifactValue(
            artifact=ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            media_type="text/plain",
            data=self.read_data,
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        self.write_calls += 1
        if self.write_error is not None:
            raise self.write_error
        self.write_expected_revision = expected_revision
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=ArtifactRef(
                namespace=target.namespace,
                name=target.name,
                revision="revision-write",
            ),
            mediaType=media_type,
            size=len(data),
        )
