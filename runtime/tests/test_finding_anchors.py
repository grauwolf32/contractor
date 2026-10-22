from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

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

    from contractor_runtime.allocation import WorkerState
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
        tool = await finding_tool(CodeFindingsToolsetFactory, api, WorkerState())
        result = await tool(
            title="Hardcoded key",
            description="An independent code observation",
            file=case["file"],
            line=case["line"],
            cwe="CWE-321",
            evidence_refs=[ArtifactRef.model_validate(REF)],
            tool_context=context(),
        )
        assert set(result) == {"proposal_id", "receipt_id", "client_key"}
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
