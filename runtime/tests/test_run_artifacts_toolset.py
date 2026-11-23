from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fakes.artifact_reads import CountingPayload, count_derivations
from fakes.worker_state import WorkerState

import contractor_runtime.toolsets.run_artifacts.tools as run_artifacts
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
    ArtifactMetadataResult,
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
        assert read["text"] == "payload"
        assert state.metrics.counters["tool_calls"] == 2
        assert all(entry.result_size_bytes is not None for entry in state.metrics.tool_calls)
        assert "dataBase64" not in repr(state.metrics.tool_calls)

    asyncio.run(scenario())


def test_read_artifact_pages_large_payloads() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.read_data = ("ä😀" * 50_000 + "\ntail\n").encode()
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        read = (await create_tools(factory, state, ["read_artifact"]))["read_artifact"]
        page = await read("inputs", "source")
        revision = page["artifact"]["revision"]
        collected = page["text"]
        while page["truncated"]:
            page = await read(cursor=page["nextCursor"])
            collected += page["text"]
        assert collected.encode() == client.read_data
        assert client.read_calls == 1
        assert "dataBase64" not in page
        assert (await read("inputs", "source"))["artifact"]["revision"] == revision
        assert client.read_calls == 1 and client.metadata_calls == 2
        for arguments in (
            {"max_lines": 0},
            {"max_lines": 401},
            {"start_line": -1},
            {"line_offset": -1},
        ):
            with pytest.raises(ToolInputError):
                await read("inputs", "source", **arguments)

    asyncio.run(scenario())


def test_text_pages_of_a_32_mib_artifact_validate_utf8_once(monkeypatch) -> None:
    async def scenario() -> None:
        line = "äöü € 😀 0123456789\n".encode()
        payload = CountingPayload(line * (32 * 1024 * 1024 // len(line) + 1))
        counted = count_derivations(monkeypatch)
        client = FakeArtifactClient()
        client.read_data = payload
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        read = (await create_tools(factory, WorkerState(), ["read_artifact"]))["read_artifact"]
        for start_line in range(1, 4001, 400):
            page = await read(
                cursor=read._cursors.issue(
                    ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
                    start_line=start_line,
                    max_lines=400,
                )
            )
            assert page["text"].encode() == line * 400
            assert page["size"] == len(payload)
        assert client.read_calls == 1
        assert (counted.validations, counted.indexes, payload.decodes) == (1, 1, 0)

        client.read_data = CountingPayload(payload[:-1] + b"\xff")
        await read.close()
        for _ in range(2):
            with pytest.raises(ToolInputError, match="not valid UTF-8"):
                await read(
                    cursor=read._cursors.issue(
                        ArtifactRef(namespace="inputs", name="source", revision="revision-read")
                    )
                )
        assert client.read_calls == 2 and counted.validations == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "media_type",
    ["application/zip", "application/vnd.contractor.agent-skill+zip", "font/woff2"],
)
def test_read_artifact_rejects_binary_media_types_before_download(media_type: str) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.read_media_type = media_type
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["read_artifact"])

        with pytest.raises(ToolInputError, match="binary") as caught:
            await tools["read_artifact"]("inputs", "source")
        assert media_type in str(caught.value)
        assert client.metadata_calls == 1 and client.read_calls == 0

    asyncio.run(scenario())


def test_read_artifact_rejects_binary_artifacts_without_returning_bytes() -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        client.read_data = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\xff\xfe binary"
        state = WorkerState()
        factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)
        tools = await create_tools(factory, state, ["read_artifact"])

        with pytest.raises(ToolInputError, match="not valid UTF-8"):
            await tools["read_artifact"]("inputs", "source")
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


@asynccontextmanager
async def local_source(tmp_path, data=b"payload"):
    from test_projectfs_storage import limits, local_settings

    from contractor_runtime.projectfs import DirectWorkspaceSession, LocalWorkspaceProvider

    provider = LocalWorkspaceProvider(local_settings(tmp_path / "owned"))
    storage = await provider.create("allocation-1")
    root = Path(storage.root) / "run_workdir"
    root.mkdir()
    (root / "report.json").write_bytes(data)
    session = DirectWorkspaceSession(
        mode="direct",
        storage=storage,
        limits=limits(),
        directories=set(),
        text_files={},
        binary_paths=set(),
    )
    try:
        yield session.reader_view(), root
    finally:
        await session.close()
        await provider.cleanup(storage)


def test_write_tool_preserves_file_bytes_revision_and_content_free_metrics(tmp_path) -> None:
    async def scenario():
        data = ("é😀 " + SECRET + "\r\n  ").encode() + b"\x00\xff"
        async with local_source(tmp_path, data) as (source, _root):
            client = FakeArtifactClient()
            state = WorkerState()
            factory = RunArtifactsToolsetFactory(lambda *_: client)
            tools = await create_tools(factory, state, ["write_artifact"], source=source)
            result = await tools["write_artifact"](
                "inputs",
                "source",
                "report.json",
                "application/octet-stream",
            )
            assert result["artifact"]["revision"] == "revision-write"
            assert client.write_expected_revision is None
            assert client.written_data == data
            assert state.metrics.tool_calls[0].arguments["source_path"] == "report.json"
            assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_file_publication_enforces_artifact_ceiling_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(run_artifacts, "artifact_payload_max_bytes", lambda: 4)

    async def scenario():
        async with local_source(tmp_path, b"12345") as (source, _root):
            client = FakeArtifactClient()
            tools = await create_tools(
                RunArtifactsToolsetFactory(lambda *_: client),
                WorkerState(),
                ["write_artifact"],
                source=source,
            )
            with pytest.raises(ToolInputError, match="Artifact limit"):
                await tools["write_artifact"]("builder", "result", "report.json", "text/plain")
            assert client.write_calls == 0

    asyncio.run(scenario())


def test_cancelled_file_publication_never_writes_and_workspace_joins_read(tmp_path, monkeypatch):
    import threading

    from contractor_runtime.projectfs.local_io import RootedLocalFilesystem

    async def scenario():
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = RootedLocalFilesystem.read

        def read(*args, **kwargs):
            entered.set()
            try:
                assert release.wait(30)
                return original(*args, **kwargs)
            finally:
                finished.set()

        async with local_source(tmp_path) as (source, _root):
            monkeypatch.setattr(RootedLocalFilesystem, "read", read)
            client = FakeArtifactClient()
            tools = await create_tools(
                RunArtifactsToolsetFactory(lambda *_: client),
                WorkerState(),
                ["write_artifact"],
                source=source,
            )
            pending = asyncio.create_task(
                tools["write_artifact"]("builder", "result", "report.json", "text/plain")
            )
            try:
                assert await asyncio.to_thread(entered.wait, 30)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert client.write_calls == 0
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
        assert finished.is_set()
        assert client.write_calls == 0

    asyncio.run(scenario())


def test_tool_error_metrics_are_bounded_typed_and_secret_free(tmp_path) -> None:
    async def scenario():
        async with local_source(tmp_path, SECRET.encode()) as (source, _root):
            client = FakeArtifactClient(
                write_error=ArtifactAPIError(403, "artifact_access_denied", False)
            )
            state = WorkerState()
            factory = RunArtifactsToolsetFactory(lambda *_: client)
            tools = await create_tools(factory, state, ["write_artifact"], source=source)
            with pytest.raises(ArtifactAPIError):
                await tools["write_artifact"]("outputs", "result", "report.json", "text/plain")
            assert state.metrics.counters["tool_errors"] == 1
            assert state.metrics.tool_calls[0].error.code == "artifact_access_denied"
            assert state.metrics.tool_calls[0].error.retryable is False
            assert client.write_calls == 1
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
        for hidden_namespace in ("skills",):
            with pytest.raises(ValueError, match="purpose-specific"):
                await tools["list_artifacts"](hidden_namespace)
        assert client.list_calls == list_calls_before
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"](
                cursor=tools["read_artifact"]._cursors.issue(
                    ArtifactRef(namespace="skills", name="likec4", revision="revision-skill")
                )
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"]("skills", "likec4", "application/octet-stream", "")
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"](
                cursor=tools["read_artifact"]._cursors.issue(
                    ArtifactRef(
                        namespace="analysis", name="memory.hidden", revision="revision-hidden"
                    )
                )
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"](
                "analysis",
                "memory.hidden",
                "application/json",
                "missing.json",
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["read_artifact"](
                cursor=tools["read_artifact"]._cursors.issue(
                    ArtifactRef(
                        namespace="analysis", name="http.body.hidden", revision="revision-http"
                    )
                )
            )
        with pytest.raises(ValueError, match="purpose-specific"):
            await tools["write_artifact"](
                "analysis",
                "http.body.hidden",
                "application/json",
                "missing.json",
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
            await tools["read_artifact"](
                cursor=tools["read_artifact"]._cursors.issue(
                    ArtifactRef(namespace="skills", name="likec4", revision="revision-skill")
                )
            )
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


def test_shared_model_hidden_bindings_match_the_shared_cases() -> None:
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
    *,
    source=None,
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
        resources=state.resources,
        project_workspace=source,
    )
    return dict(result)


class FakeArtifactClient:
    def __init__(self, *, write_error: Exception | None = None) -> None:
        self.write_error = write_error
        self.read_data = b"payload"
        self.read_media_type = "text/plain"
        self.write_expected_revision: str | None = None
        self.metadata_calls = 0
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

    async def read_artifact_metadata(self, ref: ArtifactRef) -> ArtifactMetadataResult:
        self.metadata_calls += 1
        assert ref.namespace == "inputs" and ref.name == "source"
        assert ref.revision in {None, "revision-read"}
        return ArtifactMetadataResult(
            contentCacheKey="test_metadata_version",
            apiVersion=API_VERSION,
            artifact=ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            mediaType=self.read_media_type,
            size=len(self.read_data),
        )

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.read_calls += 1
        assert ref.namespace == "inputs" and ref.name == "source"
        assert ref.revision == "revision-read"
        return ArtifactValue(
            artifact=ArtifactRef(namespace="inputs", name="source", revision="revision-read"),
            media_type=self.read_media_type,
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
        self.written_data = data
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


@pytest.mark.parametrize("kind", ["missing", "escape", "absolute", "symlink", "oversize"])
def test_file_publication_rejects_unavailable_sources_before_write(tmp_path, kind):
    async def scenario():
        async with local_source(tmp_path) as (source, root):
            selected = "report.json"
            if kind == "missing":
                selected = "missing.json"
            elif kind == "escape":
                selected = "../report.json"
            elif kind == "absolute":
                selected = str(root / "report.json")
            elif kind == "symlink":
                (root / "link.json").symlink_to(root / "report.json")
                selected = "link.json"
            else:
                (root / "report.json").write_bytes(b"x" * 1025)
            client = FakeArtifactClient()
            tools = await create_tools(
                RunArtifactsToolsetFactory(lambda *_: client),
                WorkerState(),
                ["write_artifact"],
                source=source,
            )
            with pytest.raises(ToolInputError, match="source_path"):
                await tools["write_artifact"]("builder", "result", selected, "text/plain")
            assert client.write_calls == 0

    asyncio.run(scenario())


def test_file_publication_rejects_inline_content_even_with_a_valid_path(tmp_path):
    from types import SimpleNamespace

    from contractor_runtime.worker.function_tools import ContractorFunctionTool

    async def scenario():
        async with local_source(tmp_path) as (source, _root):
            client = FakeArtifactClient()
            tools = await create_tools(
                RunArtifactsToolsetFactory(lambda *_: client),
                WorkerState(),
                ["write_artifact"],
                source=source,
            )
            tool = ContractorFunctionTool(
                tools["write_artifact"], logical_invocation_id=lambda _: "logical"
            )
            fields = tool._get_declaration().parameters_json_schema["properties"]
            assert set(fields) == {
                "namespace",
                "name",
                "source_path",
                "media_type",
            }
            for content in ({"data_base64": "YQ=="}, {"text": "a"}, {"expected_revision": None}):
                with pytest.raises(ToolInputError, match="undeclared field"):
                    await tool.run_async(
                        args={
                            "namespace": "builder",
                            "name": "result",
                            "source_path": "report.json",
                            "media_type": "text/plain",
                            **content,
                        },
                        tool_context=SimpleNamespace(invocation_id="sdk"),
                    )
            assert client.write_calls == 0

    asyncio.run(scenario())


def test_file_publication_reconciles_lost_reply_with_exact_source_bytes(tmp_path):
    class LostReplyClient(FakeArtifactClient):
        async def write_artifact(self, target, *, data, media_type, expected_revision):
            self.write_calls += 1
            self.saved = ArtifactValue(
                target.model_copy(update={"revision": "committed-r1"}),
                media_type,
                data,
                datetime.now(UTC),
                datetime.now(UTC),
            )
            if self.write_calls == 1:
                raise ArtifactTransportError("response lost after commit")
            raise ArtifactAPIError(409, "artifact_conflict", False)

        async def read_artifact(self, ref, *, max_bytes=None):
            self.read_calls += 1
            assert len(self.saved.data) <= max_bytes
            return self.saved

    async def scenario():
        data = "é😀\\n  ".encode()
        async with local_source(tmp_path, data) as (source, _root):
            client = LostReplyClient()
            tools = await create_tools(
                RunArtifactsToolsetFactory(lambda *_: client),
                WorkerState(),
                ["write_artifact"],
                source=source,
            )
            result = await tools["write_artifact"]("builder", "result", "report.json", "text/plain")
            assert result["artifact"]["revision"] == "committed-r1"
            assert client.saved.data == data
            assert client.write_calls == 2 and client.read_calls == 1

    asyncio.run(scenario())
