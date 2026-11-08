from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from test_projectfs_zip import archive, settings, workspace_inputs
from test_security_findings_toolset import FakeFindingClient, FakeToolContext, _settings, _workspace
from test_taint_annotations import LANGUAGE_CASES, MemoryWriter, make_tools

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.artifacts import ArtifactMetadataResult, ArtifactWriteResult
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.projectfs import MemoryWorkspaceProvider, hydrate_workspace
from contractor_runtime.projectfs.storage import WorkspaceSourceFile
from contractor_runtime.toolsets.common.trace_source import trace_source
from contractor_runtime.toolsets.security_findings.trace import TraceFindingsToolsetFactory
from contractor_runtime.toolsets.taint_annotations.tools import TaintAnnotationsToolsetFactory

SOURCE = "def first():\n    sink(value)\n\ndef second():\n    sink(other)\n"
REF = ArtifactRef(namespace="inputs", name="source", revision="1")
NOW = datetime(2026, 11, 8, tzinfo=UTC)


def capture(original, current, path="app.py"):
    return WorkspaceSourceFile("src/" + path, path, REF, original, current)


@pytest.mark.parametrize(
    "path,marker,original",
    [
        ("app.py", "#", "def first():\n    sink(x)\n"),
        ("app.js", "//", "function first() {\n    sink(x);\n}\n"),
        ("app.lua", "--", "function first()\n    sink(x)\nend\n"),
    ],
)
def test_mapping_uses_inserted_annotation_lines_only(path, marker, original):
    current = f"{marker} @trace target=test\n{marker} @sink kind=test arg=x\n" + original
    proof = trace_source(capture(original, current, path), 4, 4)
    assert proof["source"]["startLine"] == 2
    assert proof["source"]["endLine"] == 2
    assert proof["workspace"]["text"] == "".join(current.splitlines(keepends=True)[:4])
    assert trace_source(capture(original, current, path), 1, 2)["source"]["startLine"] == 1


@pytest.mark.parametrize(
    "original,current,start,end",
    [
        ("a\n", "a changed\n", 1, 1),
        ("a\nb\n", "a\n", 1, 1),
        ("a\n", "# ordinary comment\na\n", 2, 2),
        ("a\n", "a\n# @trace target=end\n", 2, 2),
        ("a\n", "a\n", 0, 1),
        ("a\n", "a\n", 1, 2),
        ("a\n", "a\n", 2, 1),
    ],
)
def test_mapping_rejects_unsupported_edits_and_locations(original, current, start, end):
    with pytest.raises(ToolInputError):
        trace_source(capture(original, current), start, end)


def test_existing_annotations_and_crlf_are_preserved():
    original = "# @trace target=old\r\ndef first():\r\n    sink(x)\r\n"
    current = "# @trace target=new\r\n" + original
    proof = trace_source(capture(original, current), 4, 4)
    assert proof["source"]["startLine"] == 3
    assert proof["workspace"]["text"] == current


def test_large_file_maps_linearly_and_enforces_snippet_limits():
    original = "same();\n" * 99_998 + "def first():\n    sink(x)\n"
    current = original.replace("def first():", "# @trace target=test\ndef first():")
    assert (
        trace_source(capture(original, current), 100_001, 100_001)["source"]["startLine"] == 100_000
    )
    with pytest.raises(ToolInputError):
        trace_source(capture(original, current), 1, 401)
    long = "x" * (128 * 1024 + 1)
    with pytest.raises(ToolInputError):
        trace_source(capture(long, long), 1, 1)


class PublishingTransport(FakeFindingClient):
    async def request(self, method, path, **kwargs):
        response = await super().request(method, path, **kwargs)
        if method == "POST" and self.requests[-1].get("publishProblem"):
            body = json.loads(response.body)
            body["findingId"] = "finding-trace"
            return ArtifactHTTPResponse(
                response.status_code, response.headers, json.dumps(body).encode()
            )
        return response


class MemoryArtifacts(ArtifactClient):
    def __init__(self, transport, values):
        super().__init__("allocation-1", transport)
        self.values = {(v.artifact.namespace, v.artifact.name): v for v in values}
        self.writes = 0
        self.lose_response = False

    async def read_artifact(self, ref, *, max_bytes=None, **kwargs):
        value = self.values.get((ref.namespace, ref.name))
        if value is None:
            raise ArtifactAPIError(404, "artifact_not_found", False)
        assert ref.revision is None or ref.revision == value.artifact.revision
        assert max_bytes is None or len(value.data) <= max_bytes
        return value

    async def read_artifact_metadata(self, ref, **kwargs):
        value = await self.read_artifact(ref)
        return ArtifactMetadataResult(
            apiVersion="contractor/v1alpha1",
            artifact=value.artifact,
            mediaType=value.media_type,
            size=len(value.data),
            contentCacheKey="trace_test_key",
        )

    async def write_artifact(self, target, *, data, media_type, expected_revision):
        assert expected_revision is None
        key = (target.namespace, target.name)
        if key in self.values:
            raise ArtifactAPIError(409, "artifact_conflict", False)
        self.writes += 1
        exact = target.model_copy(update={"revision": "proof-1"})
        self.values[key] = ArtifactValue(exact, media_type, data, NOW, NOW)
        if self.lose_response:
            self.lose_response = False
            raise ArtifactTransportError("response lost")
        return ArtifactWriteResult(
            apiVersion="contractor/v1alpha1", artifact=exact, mediaType=media_type, size=len(data)
        )


async def prepare(target="backend", source=SOURCE):
    spec, reader = workspace_inputs([("source", target, archive({"app.py": source.encode()}))])
    spec.mode = "overlay"
    session = await hydrate_workspace(
        provider=MemoryWorkspaceProvider(settings("memory")),
        spec=spec,
        artifact_reader=reader,
        allocation_id="allocation-1",
        timeout_seconds=5,
    )
    state = WorkerState()
    transport = PublishingTransport()
    client = MemoryArtifacts(transport, reader.values.values())
    tools = await TraceFindingsToolsetFactory(lambda *_: client).create_selected(
        selected=["finding"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="analysis",
        runtime_settings=_settings(),
        workspace=_workspace(),
        resources=state.resources,
        project_workspace=session.source_view(),
    )
    annotations = await TaintAnnotationsToolsetFactory().create_selected(
        selected=["annotate_trace", "annotate_sink"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="analysis",
        runtime_settings=_settings(),
        workspace=_workspace(),
        resources=state.resources,
        project_workspace=session.writer_view(),
    )
    return session, tools["finding"], annotations, client, transport


def test_actual_annotation_finding_provider_and_replay():
    async def scenario():
        session, tool, annotations, client, transport = await prepare()
        result = await annotations["annotate_trace"](
            path="backend/app.py", symbol="first", target="route"
        )
        assert result["changed"] is True
        # Repeated annotation changes neither bytes nor the mapping.
        assert not (
            await annotations["annotate_trace"](
                path="backend/app.py", symbol="first", target="route"
            )
        )["changed"]
        declaration = FunctionTool(tool)._get_declaration()
        assert declaration.name == "finding"
        context = FakeToolContext("invocation-1")
        arguments = dict(
            title="Missing guard",
            description="Source-supported problem",
            file="backend/app.py",
            line=3,
            publish_problem=True,
            incidental=True,
            proposed_checks=[{"objective": "Verify", "method": "source"}],
        )
        client.lose_response = True
        await FunctionTool(tool).run_async(args=arguments, tool_context=context)
        assert tool._publisher.metrics.counters["tool_calls"] == 3
        first = transport.requests[0]
        assert first["proposal"]["locations"] == [{"file": "app.py", "line": 2}]
        assert first["publishProblem"] is True and first["incidental"] is True
        assert first["proposal"]["proposed_checks"] == [{"objective": "Verify", "method": "source"}]
        proof_value = next(v for v in client.values.values() if v.media_type == "application/json")
        proof = json.loads(proof_value.data)
        assert "# @trace" in proof["workspace"]["text"]
        assert proof["source"]["startLine"] == 2
        assert proof["source"]["artifact"] in first["evidenceRefs"]
        assert proof_value.artifact.model_dump() in first["evidenceRefs"]
        # Replay must use the retained proof even after the workspace is rolled back.
        await session.rollback_changes()
        await FunctionTool(tool).run_async(args=arguments, tool_context=context)
        assert transport.requests[1] == first and client.writes == 1
        assert tool._publisher.metrics.counters["tool_calls"] == 4
        with pytest.raises(ToolInputError, match="different arguments"):
            await tool(**{**arguments, "line": 2}, tool_context=context)
        assert len(transport.requests) == 2
        assert tool._publisher.metrics.counters["tool_calls"] == 5
        await tool.close()
        for annotation in annotations.values():
            await annotation.close()
        await session.close()

    asyncio.run(scenario())


def test_hydration_base_survives_checkpoint_import_and_rollback():
    async def scenario():
        session, tool, annotations, _client, transport = await prepare("")
        await annotations["annotate_trace"](path="app.py", symbol="first", target="first")
        bundle = await session.prepare_export()
        await session.commit_export(bundle)
        await session.import_state(bundle.state)
        await annotations["annotate_sink"](path="app.py", symbol="second", kind="call", arg="other")
        cap = await session.source_view().capture_source_file("app.py")
        assert cap.original == SOURCE
        current_line = cap.current.splitlines().index("    sink(other)") + 1
        assert trace_source(cap, current_line, current_line)["source"]["startLine"] == 5
        await session.rollback_changes()
        cap = await session.source_view().capture_source_file("app.py")
        assert cap.original == SOURCE
        assert cap.current.count("# @") == 1
        await tool(
            title="Problem",
            description="Evidence",
            file="app.py",
            range={"start_line": 2, "end_line": 3},
            tool_context=FakeToolContext("invocation-range"),
        )
        assert transport.requests[-1]["proposal"]["locations"] == [
            {"file": "app.py", "range": {"start_line": 1, "end_line": 2}}
        ]
        # Genuine edits fail before saving a proof or submitting a finding.
        await session.write_text("app.py", SOURCE.replace("sink(value)", "changed(value)"))
        with pytest.raises(ToolInputError):
            await tool(
                title="Problem",
                description="Evidence",
                file="app.py",
                line=2,
                tool_context=FakeToolContext("invocation-invalid"),
            )
        assert len(transport.requests) == 1
        await tool.close()
        for annotation in annotations.values():
            await annotation.close()
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation", ["unknown", "nested-unknown", "null", "missing", "case-alias", "duplicate"]
)
def test_saved_proof_is_strict_and_cannot_trigger_new_submission(mutation):
    async def scenario():
        session, tool, annotations, client, transport = await prepare("")
        arguments = dict(title="Problem", description="Evidence", file="app.py", line=2)
        context = FakeToolContext("strict-proof")
        await tool(**arguments, tool_context=context)
        key = next(
            key for key, value in client.values.items() if value.media_type == "application/json"
        )
        value = client.values[key]
        proof = json.loads(value.data)
        if mutation == "unknown":
            proof["other"] = 1
        elif mutation == "nested-unknown":
            proof["source"]["other"] = 1
        elif mutation == "null":
            proof["source"]["artifact"] = None
        elif mutation == "missing":
            del proof["source"]["artifact"]
        elif mutation == "case-alias":
            proof["source"]["File"] = proof["source"].pop("file")
        raw = json.dumps(proof).encode()
        if mutation == "duplicate":
            raw = raw.replace(b'"schema":', b'"schema": "bad", "schema":', 1)
        client.values[key] = ArtifactValue(value.artifact, value.media_type, raw, NOW, NOW)
        with pytest.raises(ToolInputError, match="proof is invalid"):
            await tool(**arguments, tool_context=context)
        assert len(transport.requests) == 1
        await tool.close()
        for annotation in annotations.values():
            await annotation.close()
        await session.close()

    asyncio.run(scenario())


def test_ambiguous_proof_read_failure_preserves_transport_error_and_replay():
    async def scenario():
        session, tool, annotations, client, transport = await prepare("")
        await annotations["annotate_trace"](path="app.py", symbol="first", target="route")
        read = client.read_artifact
        unavailable = True

        async def failed_read(ref, **kwargs):
            if unavailable and client.writes and ref.name.startswith("trace-proof-"):
                raise ArtifactTransportError("proof acknowledgement is unavailable")
            return await read(ref, **kwargs)

        client.read_artifact = failed_read
        client.lose_response = True
        context = FakeToolContext("ambiguous-proof")
        arguments = dict(title="Problem", description="Evidence", file="app.py", line=3)
        with pytest.raises(ArtifactTransportError):
            await tool(**arguments, tool_context=context)
        assert client.writes == 1 and transport.requests == []
        unavailable = False
        await session.rollback_changes()
        await tool(**arguments, tool_context=context)
        assert client.writes == 1
        assert transport.requests[0]["proposal"]["locations"] == [{"file": "app.py", "line": 2}]
        await tool.close()
        for annotation in annotations.values():
            await annotation.close()
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("path,original,symbol,_expected", LANGUAGE_CASES)
def test_trace_accepts_real_annotations_for_every_language(
    tmp_path, path, original, symbol, _expected
):
    async def scenario():
        writer = MemoryWriter({path: original})
        tools, _ = await make_tools(tmp_path, writer)
        for name, args in [
            ("annotate_trace", {"target": "route", "args": "value:tainted", "calls": "db.query"}),
            ("annotate_validate", {"arg": "value", "kind": "guard"}),
            ("annotate_sink", {"arg": "value", "kind": "db.query"}),
        ]:
            await tools[name](path=path, symbol=symbol, **args)
        current = await writer.read_text(path)
        proof = trace_source(
            capture(original, current, path), len(current.splitlines()), len(current.splitlines())
        )
        assert proof["source"]["endLine"] == len(original.splitlines())
        marker = "#" if "# @trace" in current else "--" if "-- @trace" in current else "//"
        wrong = "//" if marker != "//" else "#"
        with pytest.raises(ToolInputError):
            trace_source(
                capture(original, current.replace(marker + " @", wrong + " @"), path), 1, 1
            )
        for tool in tools.values():
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "original,current",
    [
        (
            'query = """\nSELECT id FROM users\n"""\nexecute(query)\n',
            'query = """\n# @trace target=route\nSELECT id FROM users\n"""\nexecute(query)\n',
        ),
        (SOURCE, "// @trace target=route\n" + SOURCE),
        (SOURCE, "# @trace arbitrary=route\n" + SOURCE),
        (SOURCE, "# @trace target=route args=value:other\n" + SOURCE),
        (SOURCE, "# @trace target=route calls=a,a\n" + SOURCE),
        (SOURCE, "# @trace target=route extra=value\n" + SOURCE),
        (SOURCE, SOURCE.replace("    sink(value)", "    # @trace target=route\n    sink(value)")),
        (SOURCE, "# @trace target=route\n\n" + SOURCE),
        (SOURCE, "# @trace target=coding:utf-7\n" + SOURCE),
    ],
)
def test_invalid_annotation_edits_fail_before_proof_write_and_intake(original, current):
    async def scenario():
        session, tool, annotations, client, transport = await prepare("", source=original)
        await session.write_text("app.py", current)
        # State import is not a trusted annotation receipt either.
        bundle = await session.prepare_export()
        await session.rollback_changes()
        await session.import_state(bundle.state)
        with pytest.raises(ToolInputError):
            await tool(
                title="Problem",
                description="Evidence",
                file="app.py",
                line=len(current.splitlines()),
                tool_context=FakeToolContext("invalid-annotation"),
            )
        assert client.writes == 0
        assert transport.requests == []
        await tool.close()
        for annotation in annotations.values():
            await annotation.close()
        await session.close()

    asyncio.run(scenario())


def test_unchanged_file_does_not_require_a_language_or_parser(monkeypatch):
    def unexpected_parse(*_args):
        raise RuntimeError("unchanged sources must not be parsed")

    monkeypatch.setattr(
        "contractor_runtime.toolsets.common.trace_source.load_parser", unexpected_parse
    )
    assert (
        trace_source(capture("unparseable", "unparseable", "input.unknown"), 1, 1)["source"][
            "startLine"
        ]
        == 1
    )
    with pytest.raises(ToolInputError, match="language"):
        trace_source(
            capture("unparseable", "# @trace target=route\nunparseable", "input.unknown"), 2, 2
        )


@pytest.mark.parametrize(
    "original",
    [
        "void first(int values[static 2]) {}\n",
        "class Handler {\npublic:\n    void first() {}\n};\n",
    ],
)
def test_ambiguous_headers_use_a_valid_c_or_cpp_declaration(original):
    current = original.replace(
        "void first", "// @trace target=route\n" + "    " * ("class" in original) + "void first"
    )
    proof = trace_source(
        capture(original, current, "app.h"), len(current.splitlines()), len(current.splitlines())
    )
    assert proof["source"]["endLine"] == len(original.splitlines())


@pytest.mark.parametrize(
    "path,original,inserted",
    [
        (
            "app.py",
            'query = """\nSELECT id FROM users\n"""\ndef first(): pass\n',
            "# @trace target=route\n",
        ),
        (
            "app.js",
            "const query = `\nSELECT id FROM users\n`;\nfunction first() {}\n",
            "// @trace target=route\n",
        ),
        (
            "app.go",
            "package app\nvar query = `\nSELECT id FROM users\n`\nfunc first() {}\n",
            "// @trace target=route\n",
        ),
        (
            "app.rb",
            "query = <<SQL\nSELECT id FROM users\nSQL\ndef first\nend\n",
            "# @trace target=route\n",
        ),
        (
            "app.lua",
            "query = [[\nSELECT id FROM users\n]]\nfunction first() end\n",
            "-- @trace target=route\n",
        ),
        (
            "app.sh",
            "query=$(cat <<SQL\nSELECT id FROM users\nSQL\n)\nfirst() { :; }\n",
            "# @trace target=route\n",
        ),
    ],
)
def test_annotations_inside_multiline_literals_are_not_comments(path, original, inserted):
    current = original.replace("SELECT", inserted + "SELECT")
    with pytest.raises(ToolInputError):
        trace_source(
            capture(original, current, path), len(current.splitlines()), len(current.splitlines())
        )
