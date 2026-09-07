from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from google.adk.tools import FunctionTool

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.run_artifacts.tools import ReadArtifactTool
from contractor_runtime.toolsets.security_findings.collection import (
    COLLECTION_MEDIA_TYPE,
    MAX_ARCHIVE_BYTES,
    FindingsError,
    decode_collection,
)
from contractor_runtime.toolsets.security_findings.reader import ListFindingsTool
from contractor_runtime.toolsets.security_findings.tools import (
    FindingTool,
    FindingV2Tool,
    SecurityFindingsToolsetFactory,
    SecurityFindingsV2ToolsetFactory,
)
from contractor_runtime.worker.instrumentation import _safe_tool_response
from contractor_runtime.workspace import AllocationWorkspace

FIXTURE = (
    Path(__file__).parents[2] / "internal/auditdomain/testdata/finding-collection-v1.fixture.json"
)
TIMESTAMPS = {
    "x-contractor-binding-created-at": "2026-09-06T09:00:00Z",
    "x-contractor-revision-created-at": "2026-09-06T09:00:00Z",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _fixture():
    value = json.loads(FIXTURE.read_bytes())
    return value["collection"], {key: text.encode() for key, text in value["contents"].items()}


def _zip(members, *, compression=zipfile.ZIP_STORED, mode=stat.S_IFREG | 0o644):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, body in members:
            info = zipfile.ZipInfo(path, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = mode << 16
            info.compress_type = compression
            archive.writestr(info, body)
    return output.getvalue()


def _package(collection, contents, *, metadata=None):
    body = _json(collection) if metadata is None else metadata
    docs = [("collection", "collection.json", "application/json", body)]
    for document in collection["documents"]:
        docs.append(
            (
                document["id"],
                "documents/" + document["id"],
                document["media_type"],
                contents[document["id"]],
            )
        )
    docs.sort(key=lambda item: item[1])
    manifest = {
        "schema": "contractor.audit.package.v1",
        "kind": "finding-collection",
        "package_id": "collection-" + _digest(body)[7:],
        "members": [
            {
                "id": id_,
                "path": path,
                "media_type": mime,
                "size": len(data),
                "digest": _digest(data),
            }
            for id_, path, mime, data in docs
        ],
    }
    return _zip([("manifest.json", _json(manifest)), *[(path, data) for _, path, _, data in docs]])


def _replace_proposal(collection, contents, index, transform):
    entry = collection["entries"][index]
    old_id = entry["proposal_document_id"]
    body = transform(contents.pop(old_id))
    document = next(doc for doc in collection["documents"] if doc["id"] == old_id)
    document["digest"], document["size_bytes"] = _digest(body), len(body)
    scope, ref = document["scope"], document["ref"]
    id_ = (
        "doc-"
        + _digest(
            _json(
                [
                    scope["kind"],
                    scope["id"],
                    ref["namespace"],
                    ref["name"],
                    ref["revision"],
                    document["digest"],
                    document["media_type"],
                    len(body),
                ]
            )
        )[7:]
    )
    document["id"] = entry["proposal_document_id"] = id_
    contents[id_] = body
    collection["documents"].sort(key=lambda doc: doc["id"])


class Transport:
    """Real ArtifactClient wire behavior in one allocation, with injected faults."""

    def __init__(self, payload):
        self.requests = []
        self.bindings = {("inputs", "findings"): ("input-revision", COLLECTION_MEDIA_TYPE, payload)}
        self.versions = {
            ("inputs", "findings", "input-revision"): (
                "input-revision",
                COLLECTION_MEDIA_TYPE,
                payload,
            )
        }
        self.puts = 0
        self.lose_response = False
        self.fail_at = None
        self.deny_get = False
        self.wrong_etag = False

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.requests.append((method, path, dict(headers), max_response_bytes))
        parsed = urlsplit(path)
        assert parsed.path.startswith("/allocations/consumer-allocation/artifacts/")
        namespace, name = parsed.path.split("/")[-2:]
        target = (namespace, name)
        if method == "GET":
            if self.deny_get:
                return ArtifactHTTPResponse(403, {}, b"")
            revision = parse_qs(parsed.query).get("revision", [None])[0]
            record = (
                self.bindings.get(target)
                if revision is None
                else self.versions.get((*target, revision))
            )
            if record is None:
                return ArtifactHTTPResponse(404, {}, b"")
            revision, media_type, data = record
            if self.wrong_etag and namespace != "inputs":
                revision = "wrong-exact-revision"
            return ArtifactHTTPResponse(
                200,
                {
                    "content-type": media_type,
                    "etag": '"' + revision + '"',
                    **TIMESTAMPS,
                },
                data,
            )
        assert method == "PUT" and namespace.startswith("findings-")
        assert headers["If-None-Match"] == "*" and "If-Match" not in headers
        assert not parsed.query
        self.puts += 1
        if self.puts == self.fail_at:
            return ArtifactHTTPResponse(503, {}, b"")
        if target in self.bindings:
            return ArtifactHTTPResponse(412, {}, b"")
        revision = "consumer-revision-" + str(self.puts)
        record = (revision, headers["Content-Type"], body)
        self.bindings[target] = record
        self.versions[(*target, revision)] = record
        if self.lose_response:
            self.lose_response = False
            raise ArtifactTransportError("response lost")
        return ArtifactHTTPResponse(
            201,
            {
                "content-type": "application/json",
                "etag": '"' + revision + '"',
                **TIMESTAMPS,
            },
            _json(
                {
                    "apiVersion": "contractor/v1alpha1",
                    "artifact": {
                        "namespace": namespace,
                        "name": name,
                        "revision": revision,
                    },
                    "mediaType": headers["Content-Type"],
                    "size": len(body),
                }
            ),
        )


async def _tools(transport, selected=("list_findings",), *, version=2):
    state = WorkerState()
    client = ArtifactClient("consumer-allocation", transport)
    factory_class = (
        SecurityFindingsV2ToolsetFactory if version == 2 else SecurityFindingsToolsetFactory
    )
    factory = factory_class(lambda _allocation, _settings: client)
    tools = await factory.create_selected(
        selected=selected,
        allocation_id="consumer-allocation",
        run_id="consumer-run",
        namespace="reader",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="secret",
            artifactApiUrl="https://cp.example/private/v1",
            requestTimeoutSeconds=5,
        ),
        workspace=AllocationWorkspace(root=Path("/tmp/findings"), path=Path("/tmp/findings/a")),
        state=state,
    )
    return tools, client, state


def test_shared_go_python_fixture_and_exact_consumer_reads():
    async def scenario():
        collection, contents = _fixture()
        payload = _package(collection, contents)
        frozen = json.loads(FIXTURE.read_bytes())
        assert _digest(payload) == frozen["package_digest"]
        assert "collection-" + _digest(_json(collection))[7:] == frozen["package_id"]
        decoded = decode_collection(payload)
        assert decoded.metadata == collection and decoded.contents == contents
        transport = Transport(payload)
        tools, client, state = await _tools(transport)
        result = await tools["list_findings"]()
        assert len(result["items"]) == 2 and result["next_cursor"] is None
        reader = ReadArtifactTool(client, state.metrics, ())
        for index, item in enumerate(result["items"]):
            entry = collection["entries"][index]
            assert item["receipt_id"] == entry["receipt_id"]
            assert item["reviews"] == entry["reviews"]
            assert item["has_hypothesis"] == (index == 1)
            for document in [item["proposal"], *item["evidence"]]:
                ref = document["ref"]
                assert ref["namespace"] == "findings-" + frozen["package_digest"][7:]
                assert ref["revision"].startswith("consumer-revision-")
                assert document["source"]["ref"]["revision"] == "rev-1"
                assert document["source"]["scope"]["kind"] == "run"
                read = await reader(**ref)
                assert base64.b64decode(read["dataBase64"]) == contents[ref["name"]]
        first, second = result["items"]
        assert first["evidence"][0]["digest"] == second["evidence"][0]["digest"]
        assert first["evidence"][0]["ref"] != second["evidence"][0]["ref"]
        assert "generic finding" not in repr(state.metrics.tool_calls)
        assert all(
            "run-a" not in request[1] and "run-b" not in request[1]
            for request in transport.requests
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("version", "selected"),
    [
        (1, ("finding",)),
        (2, ("finding",)),
        (2, ("list_findings",)),
        (2, ("finding", "list_findings")),
    ],
)
def test_selected_tools_and_adk_descriptions(version, selected, tmp_path):
    async def scenario():
        transport = Transport(_package(*_fixture()))
        tools, _, _ = await _tools(transport, selected, version=version)
        assert set(tools) == set(selected)
        if "list_findings" not in selected:
            assert transport.requests == []
        for name, tool in tools.items():
            declaration = (
                FunctionTool(tool)._get_declaration().model_dump(mode="json", by_alias=True)
            )
            assert declaration["name"] == name
            assert declaration["description"]
            parameters = declaration["parametersJsonSchema"]["properties"]
            assert "tool_context" not in parameters
            if name == "list_findings":
                assert set(parameters) == {"subject_kind", "subject_key", "limit", "cursor"}
            elif version == 1:
                assert (
                    type(tool) is FindingTool and "security finding" in declaration["description"]
                )
            else:
                assert type(tool) is FindingV2Tool and "reproduce" in declaration["description"]
        registry = built_in_factories(tmp_path)
        assert await registry.toolsets[f"security-findings@{version}"].probe() == (
            frozenset({"finding"}) if version == 1 else frozenset({"finding", "list_findings"})
        )

    asyncio.run(scenario())


def test_reader_filters_cursors_previews_and_defensive_copies():
    async def scenario():
        collection, contents = _fixture()

        def change(body):
            proposal = json.loads(body)
            proposal["title"] = "界" * 300
            proposal["description"] = "🦆" * 1000
            return _json(proposal)

        _replace_proposal(collection, contents, 0, change)
        transport = Transport(_package(collection, contents))
        tools, _, _ = await _tools(transport)
        reader = tools["list_findings"]
        page = await reader(limit=1)
        item = page["items"][0]
        assert item["title_preview"] == {"text": "界" * 170, "truncated": True}
        assert item["description_preview"] == {"text": "🦆" * 512, "truncated": True}
        cursor = page["next_cursor"]
        second = await reader(cursor=cursor, limit=100)
        assert [item["receipt_id"] for item in second["items"]] == ["receipt-b"]
        assert second["next_cursor"] is None
        item["proposal"]["ref"]["revision"] = "corrupted"
        item["subject"]["key"] = "corrupted"
        filtered = await reader(subject_kind="function", subject_key="subject-a")
        assert filtered["items"][0]["subject"]["key"] == "subject-a"
        assert filtered["items"][0]["proposal"]["ref"]["revision"] != "corrupted"
        assert await reader(subject_kind="Function") == {"items": [], "next_cursor": None}
        assert await reader(subject_kind="function", subject_key="absent") == {
            "items": [],
            "next_cursor": None,
        }
        for arguments in [
            {"subject_key": "x"},
            {"subject_kind": ""},
            {"subject_key": ""},
            {"limit": True},
            {"limit": 0},
            {"limit": 101},
            {"limit": 1.5},
        ]:
            with pytest.raises(FindingsError, match="arguments_invalid"):
                await reader(**arguments)
        raw_cursor = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        invalid_cursors = [cursor + "=", "!", "", "a" * 1367]
        for key, replacement in [
            ("version", True),
            ("version", 2),
            ("collection_digest", "sha256:" + "0" * 64),
            ("after_receipt_id", "unknown"),
            ("subject_kind", "configuration"),
        ]:
            value = {**raw_cursor, key: replacement}
            invalid_cursors.append(base64.urlsafe_b64encode(_json(value)).rstrip(b"=").decode())
        invalid_cursors.append(
            base64.urlsafe_b64encode(_json(raw_cursor) + b" ").rstrip(b"=").decode()
        )
        for invalid in invalid_cursors:
            with pytest.raises(FindingsError, match="cursor_invalid"):
                await reader(cursor=invalid)
        with pytest.raises(FindingsError, match="cursor_invalid"):
            await reader(cursor=cursor, subject_kind="function")
        transport.bindings[("inputs", "findings")] = ("changed", COLLECTION_MEDIA_TYPE, b"bad")
        assert (await reader(limit=1))["next_cursor"] == cursor
        await reader.close()
        with pytest.raises(FindingsError, match="reader_unavailable"):
            await reader()

    asyncio.run(scenario())


def test_empty_collection_is_success():
    async def scenario():
        collection, _ = _fixture()
        collection["entries"], collection["documents"] = [], []
        transport = Transport(_package(collection, {}))
        tools, _, _ = await _tools(transport)
        assert await tools["list_findings"]() == {"items": [], "next_cursor": None}
        assert transport.puts == 0

    asyncio.run(scenario())


def test_replay_loss_interruption_conflict_and_access_errors():
    async def scenario():
        transport = Transport(_package(*_fixture()))
        transport.fail_at = 3
        with pytest.raises(FindingsError, match="document_unavailable"):
            await _tools(transport, ("finding", "list_findings"))
        assert len(transport.bindings) == 3  # Input and two retained intermediates.
        transport.lose_response = True
        tools, _, _ = await _tools(transport)
        first = await tools["list_findings"]()
        again, _, _ = await _tools(transport)
        assert await again["list_findings"]() == first
        document = first["items"][0]["proposal"]["ref"]
        target = document["namespace"], document["name"]
        transport.bindings[target] = ("changed", "application/json", b"different bytes")
        with pytest.raises(FindingsError, match="document_conflict"):
            await _tools(transport)
        assert transport.bindings[target][2] == b"different bytes"
        transport.deny_get = True
        with pytest.raises(FindingsError, match="collection_unavailable"):
            await _tools(transport)
        wrong = Transport(_package(*_fixture()))
        wrong.wrong_etag = True
        with pytest.raises(FindingsError, match="document_unavailable"):
            await _tools(wrong)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "damage",
    [
        "unknown-field",
        "wrong-case",
        "duplicate-key",
        "null-array",
        "unsorted",
        "extra-document",
        "bad-document-id",
        "bad-snapshot",
        "fractional-revision",
        "empty-holds",
        "wrong-schema",
        "invalid-source",
        "missing-link",
        "bad-digest",
        "bad-size",
        "proposal-unknown",
        "proposal-evidence",
        "proposal-duplicate",
        "proposal-null",
        "proposal-limit",
    ],
)
def test_invalid_collection_fails_before_any_materialization(damage):
    async def scenario():
        collection, contents = _fixture()
        metadata = None
        if damage == "unknown-field":
            collection["unknown"] = True
        elif damage == "wrong-case":
            collection["Schema"] = collection.pop("schema")
        elif damage == "duplicate-key":
            metadata = _json(collection).replace(b'"entries":', b'"entries":[],"entries":')
        elif damage == "null-array":
            collection["entries"][0]["evidence"] = None
        elif damage == "unsorted":
            collection["entries"].reverse()
        elif damage == "extra-document":
            collection["entries"].pop()
        elif damage == "bad-document-id":
            collection["documents"][0]["scope"]["id"] = "different-run"
        elif damage == "bad-snapshot":
            collection["snapshot_at"] = "2026-02-30T12:00:00Z"
        elif damage == "fractional-revision":
            collection["entries"][1]["reviews"][0]["revision"] = 1.5
        elif damage == "empty-holds":
            collection["entries"][0]["audit_holds"] = []
        elif damage == "wrong-schema":
            collection["schema"] = "contractor.findings.collection.v2"
        elif damage == "invalid-source":
            collection["sources"] = [{"kind": "run", "id": "not-selected"}]
        elif damage == "missing-link":
            collection["entries"][0]["evidence"][0]["document_id"] = "doc-missing"
        elif damage == "bad-digest":
            contents[collection["documents"][0]["id"]] = b"changed"
        elif damage == "bad-size":
            collection["documents"][0]["size_bytes"] = True
        else:

            def change(body):
                proposal = json.loads(body)
                if damage == "proposal-unknown":
                    proposal["unknown"] = "value"
                elif damage == "proposal-evidence":
                    proposal["evidence_ids"] = ["different"]
                elif damage == "proposal-duplicate":
                    return body.replace(b'"title":', b'"title":"duplicate","title":')
                elif damage == "proposal-null":
                    proposal["preconditions"] = None
                elif damage == "proposal-limit":
                    proposal["description"] = "x" * 65537
                return _json(proposal)

            _replace_proposal(collection, contents, 0, change)
        transport = Transport(_package(collection, contents, metadata=metadata))
        with pytest.raises(FindingsError):
            await _tools(transport, ("finding", "list_findings"))
        assert transport.puts == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "damage",
    [
        "traversal",
        "symlink",
        "executable",
        "duplicate",
        "missing",
        "unexpected",
        "compression",
        "crc",
        "oversized",
        "manifest",
    ],
)
def test_archive_boundaries(damage):
    payload = _package(*_fixture())
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    if damage == "traversal":
        members[-1] = ("../escape", members[-1][1])
    elif damage == "duplicate":
        members.append(members[-1])
    elif damage == "missing":
        members.pop()
    elif damage == "unexpected":
        members.append(("extra.txt", b"x"))
    elif damage == "manifest":
        members[0] = ("manifest.json", members[0][1] + b" ")
    mode = stat.S_IFLNK | 0o644 if damage == "symlink" else stat.S_IFREG | 0o644
    if damage == "executable":
        mode |= 0o111
    compression = zipfile.ZIP_BZIP2 if damage == "compression" else zipfile.ZIP_STORED
    if damage == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = _zip(members, mode=mode, compression=compression)
    else:
        payload = _zip(members, mode=mode, compression=compression)
    if damage == "crc":
        payload = payload.replace(b"Observation a", b"Observation z")
    elif damage == "oversized":
        payload = b"x" * (MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(FindingsError):
        decode_collection(payload)


def test_byte_bounded_pagination_never_drops_items(monkeypatch):
    async def scenario():
        tools, _, _ = await _tools(Transport(_package(*_fixture())))
        reader = tools["list_findings"]
        first = await reader(limit=1)
        monkeypatch.setattr(
            "contractor_runtime.toolsets.security_findings.reader.MAX_PAGE_BYTES", len(_json(first))
        )
        page = await reader(limit=100)
        assert page == first
        second = await reader(cursor=page["next_cursor"])
        assert second["next_cursor"] is None
        assert second["items"][0]["receipt_id"] == "receipt-b"
        monkeypatch.setattr(
            "contractor_runtime.toolsets.security_findings.reader.MAX_PAGE_BYTES", 100
        )
        with pytest.raises(FindingsError, match="limit_exceeded"):
            await reader()

    asyncio.run(scenario())


def test_page_item_bound_and_cursor_can_change_limit():
    async def scenario():
        metrics = WorkerState().metrics
        items = tuple(
            {"receipt_id": f"receipt-{i:03}", "subject": {"kind": "code", "key": "x"}}
            for i in range(256)
        )
        reader = ListFindingsTool(items, "sha256:" + "a" * 64, metrics, ())
        cursor, ids, limits = None, [], [20, 100, 37, 100]
        for limit in limits:
            page = await reader(cursor=cursor, limit=limit)
            assert len(page["items"]) <= limit
            ids.extend(item["receipt_id"] for item in page["items"])
            cursor = page["next_cursor"]
        assert cursor is None and ids == [item["receipt_id"] for item in items]

    asyncio.run(scenario())


def test_collection_supports_audit_holds_and_noncanonical_proposals():
    collection, contents = _fixture()
    collection["sources"] = [{"kind": "audit", "id": "audit-a"}]
    collection["entries"][0]["audit_holds"] = ["audit-a"]
    _replace_proposal(collection, contents, 0, lambda body: b" \n" + body + b"\n")
    payload = _package(collection, contents)
    assert decode_collection(payload).metadata == collection
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        compressed = _zip(
            [(info.filename, archive.read(info)) for info in archive.infolist()],
            compression=zipfile.ZIP_DEFLATED,
        )
    assert decode_collection(compressed).metadata == collection


def test_artifact_client_enforces_reader_byte_limit_even_with_unbounded_transport():
    async def scenario():
        transport = Transport(b"too large")
        client = ArtifactClient("consumer-allocation", transport)
        ref = ArtifactRef(namespace="inputs", name="findings")
        with pytest.raises(ArtifactTransportError, match="byte limit"):
            await client.read_artifact(ref, max_bytes=3)
        assert transport.requests[-1][3] == 3
        for bound in (0, True, 64 * 1024 * 1024 + 1):
            with pytest.raises(ValueError, match="byte limit"):
                await client.read_artifact(ref, max_bytes=bound)

    asyncio.run(scenario())


def test_reader_preparation_distinguishes_byte_limit_from_missing_input():
    async def scenario():
        with pytest.raises(FindingsError, match="limit_exceeded"):
            await _tools(Transport(b"x" * (MAX_ARCHIVE_BYTES + 1)))
        missing = Transport(b"")
        missing.bindings.clear()
        with pytest.raises(FindingsError, match="collection_unavailable"):
            await _tools(missing)

    asyncio.run(scenario())


def test_reader_errors_keep_distinct_codes_in_model_visible_envelope(monkeypatch):
    async def scenario():
        tools, _, _ = await _tools(Transport(_package(*_fixture())))
        reader = tools["list_findings"]
        for arguments, expected in [
            ({"subject_key": "subject-a"}, "findings_arguments_invalid"),
            ({"cursor": "unknown"}, "findings_cursor_invalid"),
        ]:
            with pytest.raises(FindingsError) as raised:
                await reader(**arguments)
            envelope = _safe_tool_response("list_findings", raised.value)
            assert envelope["error"]["code"] == expected
            assert envelope["ok"] is False
        monkeypatch.setattr(
            "contractor_runtime.toolsets.security_findings.reader.MAX_PAGE_BYTES", 1
        )
        with pytest.raises(FindingsError) as raised:
            await reader()
        assert _safe_tool_response("list_findings", raised.value)["error"]["code"] == (
            "findings_limit_exceeded"
        )

    asyncio.run(scenario())
