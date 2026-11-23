from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest
from fakes.source_selectors import artifact_sources
from fakes.worker_state import WorkerState

from contractor_runtime import threads
from contractor_runtime.toolsets.common import finding_anchors as anchors

REF = {"namespace": "evidence", "name": "source", "revision": "1"}


def archive(file, source):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as handle:
        handle.writestr(zipfile.ZipInfo(file, date_time=(1980, 1, 1, 0, 0, 0)), source)
    return out.getvalue()


def derive(file, source, line, *, local=False):
    return anchors.derive_zip(
        archive(file, source), REF, [{"file": file, "line": line}], qualify_local=local
    )


def test_shared_runtime_anchor_corpus():
    data = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-anchor-cases.json").read_text()
    )
    for case in data["cases"]:
        result = anchors.derive_zip(
            base64.b64decode(case["zipBase64"]),
            REF,
            [{"file": case["file"], "line": case["line"]}],
            qualify_local=case["qualifyLocal"],
        )
        assert result == case["derived"], case["name"]
    for record in data["unavailable"]:
        assert anchors.unavailable(record["reason"]) == record


@pytest.mark.parametrize(
    "file,source,line,shifted,shiftline",
    [
        (
            "a.py",
            "def f(x):\n    sink(x)\n",
            2,
            "# ordinary note\ndef f( x ):\n    # ordinary note\n    sink( x )\n",
            4,
        ),
        (
            "a.js",
            "function f(x) {\n  sink(x);\n}\n",
            2,
            "// ordinary note\nfunction f( x ) {\n// ordinary note\n  sink( x );\n}\n",
            4,
        ),
        (
            "a.ts",
            "function f(x: string) {\n  sink(x);\n}\n",
            2,
            "// ordinary note\nfunction f( x: string ) {\n// ordinary note\n  sink( x );\n}\n",
            4,
        ),
    ],
)
def test_profiles_ignore_only_syntax_trivia(file, source, line, shifted, shiftline):
    first = derive(file, source, line)["anchors"][0]
    second = derive(file, shifted, shiftline)["anchors"][0]
    for key in ("profile", "grammar", "nodeDigest", "signatureDigest", "path"):
        assert first[key] == second[key], key
    assert first["fileDigest"] != second["fileDigest"]
    changed = derive(file, source.replace("sink(x)", "sink(other)"), line)["anchors"][0]
    assert first["nodeDigest"] != changed["nodeDigest"]
    assert "context" not in first


@pytest.mark.parametrize(
    "file,source,line",
    [
        ("a.py", "def f():\n    sink('a b')\n    sink('a b')\n", 2),
        ("a.js", "function f() {\n sink('a b');\n sink('a b');\n}\n", 2),
        ("a.ts", "function f(): void {\n sink('a b');\n sink('a b');\n}\n", 2),
    ],
)
def test_repeated_nodes_and_literal_contents_remain_distinct(file, source, line):
    first = derive(file, source, line)["anchors"][0]
    repeat = derive(file, source, line + 1)["anchors"][0]
    assert first["nodeDigest"] == repeat["nodeDigest"]
    assert first["path"] != repeat["path"]
    changed = derive(file, source.replace("a b", "a  b"), line)["anchors"][0]
    assert first["nodeDigest"] != changed["nodeDigest"]


@pytest.mark.parametrize(
    "file,source,line,reason",
    [
        ("a.py", "x(); y()\n", 1, "anchor_ambiguous"),
        ("a.js", "x(); y();\n", 1, "anchor_ambiguous"),
        ("a.ts", "x(); y();\n", 1, "anchor_ambiguous"),
        ("a.py", "def f(): return 1\n", 1, "anchor_ambiguous"),
        ("a.ts", "function f(): string;\n", 1, "anchor_ambiguous"),
        ("a.py", "def f(\n", 1, "parse_error"),
        ("a.js", "function f(\n", 1, "parse_error"),
        ("a.ts", "function f(\n", 1, "parse_error"),
        ("a.go", "package p\n", 1, "anchor_language_unsupported"),
    ],
)
def test_unavailable_never_guesses_a_node(file, source, line, reason):
    assert derive(file, source, line) == anchors.unavailable(reason)


@pytest.mark.parametrize(
    "file,prefix,source",
    [
        ("a.py", "# type: ignore\n", "sink(x)\n"),
        ("a.js", "// @ts-ignore\n", "sink(x);\n"),
        ("a.ts", "// @ts-expect-error\n", "sink(x);\n"),
    ],
)
def test_semantic_comments_are_retained(file, prefix, source):
    first = derive(file, source, 1)["anchors"][0]
    second = derive(file, prefix + source, 2)["anchors"][0]
    assert first["signatureDigest"] != second["signatureDigest"]


def test_optional_preparation_limits_and_invalid_archives():
    assert anchors.derive_zip(
        b"x" * (anchors.MAX_SOURCE_BYTES + 1), REF, [], qualify_local=False
    ) == anchors.unavailable("input_limit")
    assert anchors.derive_zip(
        b"bad zip", REF, [{"file": "a.py", "line": 1}], qualify_local=False
    ) == anchors.unavailable("source_invalid")
    assert derive("../a.py", "sink(x)\n", 1) == anchors.unavailable("source_invalid")
    assert derive("a.py", "x" * (anchors.MAX_FILE_BYTES + 1), 1) == anchors.unavailable(
        "input_limit"
    )
    assert anchors.derive_zip(
        archive("a.py", "sink(x)\n"), REF, [{"file": "a.py", "line": 1}] * 33, qualify_local=False
    ) == anchors.unavailable("input_limit")


def test_cancellation_joins_parser_before_release():
    async def scenario():
        entered = threading.Event()
        release = threading.Event()
        exited = threading.Event()

        def parse():
            entered.set()
            release.wait()
            exited.set()

        task = asyncio.create_task(threads.to_thread_until_done(parse, name="anchor-cancel-test"))
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "file,source",
    [
        ("a.py", "def f():\n    sink(x)\n"),
        ("a.js", "function f() {\n  sink(x);\n}\n"),
        ("a.ts", "function f(): void {\n  sink(x);\n}\n"),
    ],
)
@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_coordinate_profiles_use_shared_source_lines(file, source, newline):
    result = derive(file, source.replace("\n", newline), 2)
    # A grammar may reject lone-CR source. It must not silently select another
    # coordinate when the parser's row counter differs from read_file.
    if result["availability"] == "available":
        assert result["anchors"][0]["startLine"] == 2
        assert (
            result["anchors"][0]["nodeDigest"]
            == derive(file, source, 2)["anchors"][0]["nodeDigest"]
        )
    else:
        assert result["reason"] in {"parse_error", "anchor_ambiguous"}


def test_selected_files_parse_once_in_publication_order(monkeypatch):
    calls = []
    real = anchors.get_parser

    def parser(language):
        calls.append(language)
        return real(language)

    monkeypatch.setattr(anchors, "get_parser", parser)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as handle:
        handle.writestr("a.py", "sink(x)\nsink(y)\n")
        handle.writestr("b.js", "sink(z);\n")
    result = anchors.derive_zip(
        out.getvalue(),
        REF,
        [{"file": "a.py", "line": 1}, {"file": "b.js", "line": 1}, {"file": "a.py", "line": 2}],
        qualify_local=False,
    )
    assert result["availability"] == "available"
    assert calls == ["python", "javascript"]
    assert [a["file"] for a in result["anchors"]] == ["a.py", "b.js", "a.py"]


def test_parser_budget_stops_without_partial_anchors(monkeypatch):
    original = anchors._Budget

    class Exhausted(original):
        def __init__(self):
            super().__init__()
            self.nodes = 100_000

    monkeypatch.setattr(anchors, "_Budget", Exhausted)
    assert derive("a.py", "sink(x)\n", 1) == anchors.unavailable("input_limit")


def test_shared_code_publisher_uses_hidden_exact_anchor_metadata():
    from test_finding_facades import context, finding_tool
    from test_security_findings_toolset import FakeFindingClient

    from contractor_runtime.artifacts import ArtifactHTTPResponse, ArtifactTransportError
    from contractor_runtime.contracts import ArtifactRef
    from contractor_runtime.toolsets.security_findings.facades import CodeFindingsToolsetFactory

    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-anchor-cases.json").read_text()
    )
    case = next(c for c in cases["cases"] if c["name"] == "python-local-key")
    source = base64.b64decode(case["zipBase64"])

    class API(FakeFindingClient):
        def __init__(self):
            super().__init__()
            self.reads = 0
            self.bodies = []

        async def request(self, method, path, *, headers, body, max_response_bytes):
            if method == "GET":
                if "/metadata?" in path:
                    return ArtifactHTTPResponse(
                        200,
                        {"content-type": "application/json"},
                        json.dumps(
                            {
                                "apiVersion": "contractor/v1alpha1",
                                "artifact": REF,
                                "mediaType": "application/zip",
                                "size": len(source),
                                "contentCacheKey": "test_source_version",
                            }
                        ).encode(),
                    )
                self.reads += 1
                return ArtifactHTTPResponse(
                    200,
                    {
                        "content-type": "application/zip",
                        "etag": '"1"',
                        "x-contractor-binding-created-at": "2026-10-22T00:00:00Z",
                        "x-contractor-revision-created-at": "2026-10-22T00:00:00Z",
                    },
                    source,
                )
            self.bodies.append(body)
            result = await super().request(
                method, path, headers=headers, body=body, max_response_bytes=max_response_bytes
            )
            if len(self.bodies) == 1:
                raise ArtifactTransportError("simulated lost response after commit")
            return result

    async def scenario():
        api = API()
        state = WorkerState()
        state.resources.selected_versions.observe(ArtifactRef.model_validate(REF))
        tool = await finding_tool(CodeFindingsToolsetFactory, api, state)
        result = await tool(
            title="Hardcoded key",
            description="An independent code observation",
            file=case["file"],
            line=case["line"],
            cwe="CWE-321",
            sources=artifact_sources([ArtifactRef.model_validate(REF)]),
            tool_context=context(),
        )
        assert set(result) == {"proposal_id", "receipt_id", "client_key", "finding_id"}
        assert api.requests[0]["derived_anchors"] == case["derived"]
        assert api.requests[0]["evidenceRefs"] == [REF]
        assert len(api.bodies) == 2 and api.bodies[0] == api.bodies[1]
        assert api.reads == 1
        assert "BEGIN PRIVATE KEY" not in repr(api.requests[0]["derived_anchors"])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "file,source,line,changed",
    [
        ("a.py", "@first\ndef f():\n    sink(x)\n", 3, "@second\ndef f():\n    sink(x)\n"),
        (
            "a.js",
            "const first = () => {\n sink(x);\n};\n",
            2,
            "const second = () => {\n sink(x);\n};\n",
        ),
        (
            "a.ts",
            "const first = (x: string) => {\n sink(x);\n};\n",
            2,
            "const second = (x: string) => {\n sink(x);\n};\n",
        ),
        ("a.js", "'use strict';\nsink(x);\n", 2, "'use asm';\nsink(x);\n"),
        ("a.ts", "'use strict';\nsink(x);\n", 2, "'use asm';\nsink(x);\n"),
    ],
)
def test_signatures_preserve_binding_decorators_and_directives(file, source, line, changed):
    first = derive(file, source, line)["anchors"][0]
    second = derive(file, changed, line)["anchors"][0]
    assert first["nodeDigest"] == second["nodeDigest"]
    assert first["signatureDigest"] != second["signatureDigest"]


def test_missing_cached_grammar_never_downloads_during_publication(monkeypatch):
    monkeypatch.setattr(anchors, "downloaded_languages", lambda: [])

    def unexpected(_language):
        raise RuntimeError("publication attempted to load an absent grammar")

    monkeypatch.setattr(anchors, "get_parser", unexpected)
    assert derive("a.py", "sink(x)\n", 1) == anchors.unavailable("anchor_parser_unavailable")


def padded_archive(size, marker):
    """A stored source ZIP of exactly ``size`` bytes with one selectable file."""

    def build(padding):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as handle:
            info = zipfile.ZipInfo("a.py", date_time=(1980, 1, 1, 0, 0, 0))
            handle.writestr(info, f"sink({marker})\nsink(y)\n")
            handle.writestr(zipfile.ZipInfo("pad.bin", date_time=(1980, 1, 1, 0, 0, 0)), padding)
        return out.getvalue()

    data = build(b"")
    data = build(b"p" * (size - len(data)))
    assert len(data) == size
    return data


class SourceAPI:
    """Private API transport: exact source metadata/payload reads plus finding intake."""

    def __init__(self, sources):
        from test_security_findings_toolset import FakeFindingClient

        self.sources = dict(sources)
        self.intake = FakeFindingClient()
        self.metadata_reads = []
        self.payload_reads = []
        self.reported_size = {}
        self.missing = set()
        self.payload_error = None
        self.gate = None

    @property
    def requests(self):
        return self.intake.requests

    async def request(self, method, path, *, headers, body, max_response_bytes):
        from urllib.parse import parse_qs, urlsplit

        from contractor_runtime.artifacts import ArtifactHTTPResponse

        if method != "GET":
            return await self.intake.request(
                method, path, headers=headers, body=body, max_response_bytes=max_response_bytes
            )
        parsed = urlsplit(path)
        revision = parse_qs(parsed.query)["revision"][0]
        if parsed.path.endswith("/metadata"):
            self.metadata_reads.append(revision)
            if revision in self.missing:
                error = {
                    "code": "artifact_not_found",
                    "message": "artifact unavailable",
                    "retryable": False,
                    "requestId": "anchor-source-test",
                }
                return ArtifactHTTPResponse(
                    404, {"content-type": "application/json"}, json.dumps(error).encode()
                )
            size = self.reported_size.get(revision, len(self.sources.get(revision, b"")))
            metadata = {
                "apiVersion": "contractor/v1alpha1",
                "artifact": {**REF, "revision": revision},
                "mediaType": "application/zip",
                "size": size,
                "contentCacheKey": "source_version_" + revision,
            }
            return ArtifactHTTPResponse(
                200, {"content-type": "application/json"}, json.dumps(metadata).encode()
            )
        self.payload_reads.append(revision)
        if self.gate is not None:
            await self.gate.wait()
        if self.payload_error is not None:
            raise self.payload_error
        return ArtifactHTTPResponse(
            200,
            {
                "content-type": "application/zip",
                "etag": f'"{revision}"',
                "x-contractor-binding-created-at": "2026-10-22T00:00:00Z",
                "x-contractor-revision-created-at": "2026-10-22T00:00:00Z",
            },
            self.sources[revision],
        )


async def code_finding_tool(api):
    from test_security_findings_toolset import _settings, _workspace

    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.toolsets.security_findings.facades import CodeFindingsToolsetFactory

    client = ArtifactClient("allocation-test", api)
    tools = await CodeFindingsToolsetFactory(lambda *_: client).create_selected(
        selected=["finding"],
        allocation_id="allocation-test",
        run_id="run-test",
        namespace="worker",
        runtime_settings=_settings(),
        workspace=_workspace(),
        resources=WorkerState().resources,
    )
    return tools["finding"], client


async def submit_code_finding(tool, revision, call):
    from test_finding_facades import context

    from contractor_runtime.contracts import ArtifactRef

    tool._publisher.resources.selected_versions.observe(
        ArtifactRef.model_validate({**REF, "revision": revision})
    )
    return await tool(
        title="Sink reached",
        description="An independent code observation.",
        file="a.py",
        line=1,
        sources=artifact_sources([ArtifactRef.model_validate({**REF, "revision": revision})]),
        tool_context=context(call_id=call),
    )


def test_thirty_findings_reuse_one_exact_source_download_and_parse_each(monkeypatch):
    from contractor_runtime.contracts import ArtifactRef

    parses = []
    real_parser = anchors.get_parser

    def parser(language):
        parses.append(language)
        return real_parser(language)

    monkeypatch.setattr(anchors, "get_parser", parser)

    async def scenario():
        api = SourceAPI({"1": archive("a.py", "sink(x)\nsink(y)\n")})
        tool, client = await code_finding_tool(api)
        exact = ArtifactRef.model_validate(REF)
        for index in range(30):
            cursor = client.observation_cursor
            await submit_code_finding(tool, "1", f"call-{index}")
            # Each publication observes the exact source content, served or downloaded.
            assert client.observed_exact_refs_since(cursor) == (exact,)
        assert api.payload_reads == ["1"]
        assert api.metadata_reads == ["1"] * 30
        # Derivation and parsing still happen once per publication (FND-R169).
        assert parses == ["python"] * 30
        derived = [request["derived_anchors"] for request in api.requests]
        assert len(derived) == 30 and derived[0]["availability"] == "available"
        assert all(value == derived[0] for value in derived)

    asyncio.run(scenario())


def test_concurrent_findings_share_one_exact_source_download():
    async def scenario():
        api = SourceAPI({"1": archive("a.py", "sink(x)\n")})
        api.gate = asyncio.Event()
        tool, _ = await code_finding_tool(api)
        calls = [
            asyncio.create_task(submit_code_finding(tool, "1", f"call-{index}"))
            for index in range(30)
        ]
        for _ in range(1000):
            if len(api.metadata_reads) == 30:
                break
            await asyncio.sleep(0)
        assert len(api.metadata_reads) == 30 and api.payload_reads == ["1"]
        api.gate.set()
        await asyncio.gather(*calls)
        assert api.payload_reads == ["1"]
        assert len(api.requests) == 30
        assert all(r["derived_anchors"]["availability"] == "available" for r in api.requests)

    asyncio.run(scenario())


def test_source_reuse_keeps_one_maximum_entry_and_is_released_with_the_tools():
    async def scenario():
        size = anchors.MAX_SOURCE_BYTES
        api = SourceAPI({"1": padded_archive(size, "a"), "2": padded_archive(size, "b")})
        api.reported_size["3"] = size + 1
        tool, _ = await code_finding_tool(api)
        for index, revision in enumerate(["1", "1", "2", "2", "1"]):
            await submit_code_finding(tool, revision, f"call-{index}")
        # One retained 8 MiB entry: another exact ref replaces and releases it.
        assert api.payload_reads == ["1", "2", "1"]
        await submit_code_finding(tool, "3", "oversized")
        assert api.requests[-1]["derived_anchors"] == anchors.unavailable("input_limit")
        assert api.payload_reads == ["1", "2", "1"]
        assert all(
            request["derived_anchors"]["availability"] == "available"
            for request in api.requests[:-1]
        )
        await tool.close()
        for index in range(2):
            await submit_code_finding(tool, "1", f"released-{index}")
        assert api.payload_reads == ["1", "2", "1", "1", "1"]

    asyncio.run(scenario())


def test_source_reuse_preserves_missing_mismatch_and_transport_outcomes():
    from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError

    async def scenario():
        api = SourceAPI({"1": archive("a.py", "sink(x)\n"), "2": archive("a.py", "sink(z)\n")})
        tool, _ = await code_finding_tool(api)
        await submit_code_finding(tool, "1", "first")
        assert api.payload_reads == ["1"] and len(api.requests) == 1
        # Retained bytes never bypass the finding's own metadata read.
        api.missing.add("1")
        with pytest.raises(ArtifactAPIError):
            await submit_code_finding(tool, "1", "missing")
        api.missing.clear()
        assert api.payload_reads == ["1"] and len(api.requests) == 1
        # Metadata that disagrees with the retained bytes downloads again and
        # fails exactly as a fresh download that disagrees with its metadata.
        api.reported_size["1"] = len(api.sources["1"]) + 1
        with pytest.raises(ArtifactTransportError, match="differs from exact content"):
            await submit_code_finding(tool, "1", "mismatch")
        del api.reported_size["1"]
        assert api.payload_reads == ["1", "1"] and len(api.requests) == 1
        await submit_code_finding(tool, "1", "consistent")
        assert api.payload_reads == ["1", "1"] and len(api.requests) == 2
        # A failed download retains nothing.
        api.payload_error = ArtifactTransportError("source response lost")
        with pytest.raises(ArtifactTransportError):
            await submit_code_finding(tool, "2", "lost")
        api.payload_error = None
        await submit_code_finding(tool, "2", "retried")
        assert api.payload_reads == ["1", "1", "2", "2"] and len(api.requests) == 3

    asyncio.run(scenario())


def test_cancelled_or_expired_source_download_retains_nothing():
    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.contracts import ArtifactRef
    from contractor_runtime.toolsets.security_findings.anchors import ExactSourceCache

    async def scenario():
        api = SourceAPI({"1": archive("a.py", "sink(x)\n"), "2": archive("b.py", "sink(y)\n")})
        api.gate = asyncio.Event()
        client = ArtifactClient("allocation-test", api)
        cache = ExactSourceCache()
        first = await client.read_artifact_metadata(ArtifactRef.model_validate(REF))
        leader = asyncio.create_task(cache.read(client, first))
        waiter = asyncio.create_task(cache.read(client, first))
        for _ in range(10):
            await asyncio.sleep(0)
        assert api.payload_reads == ["1"] and not waiter.done()
        leader.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        # The waiter is not failed by the cancelled download; it reads itself.
        assert api.payload_reads == ["1", "1"] and not waiter.done()
        api.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await leader
        assert (await waiter).data == api.sources["1"]
        cursor = client.observation_cursor
        assert (await cache.read(client, first)).data == api.sources["1"]
        assert api.payload_reads == ["1", "1"]
        assert client.observed_exact_refs_since(cursor) == (first.artifact,)
        # A deadline that expires during the download leaves no partial entry.
        api.gate = asyncio.Event()
        second = await client.read_artifact_metadata(
            ArtifactRef.model_validate({**REF, "revision": "2"})
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0):
                await cache.read(client, second)
        api.gate.set()
        assert (await cache.read(client, second)).data == api.sources["2"]
        assert (await cache.read(client, second)).data == api.sources["2"]
        assert api.payload_reads == ["1", "1", "2", "2"]

    asyncio.run(scenario())
