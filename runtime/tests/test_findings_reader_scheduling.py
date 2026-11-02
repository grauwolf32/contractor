from __future__ import annotations

import asyncio
import copy
import json
import math
import threading

import pytest
from test_findings_reader_toolset import Transport, _digest, _fixture, _json, _package, _tools

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.security_findings import reader
from contractor_runtime.toolsets.security_findings.collection import (
    MAX_ARCHIVE_BYTES,
    FindingsError,
    document_id,
)


def _many_documents(count):
    collection, original = _fixture()
    template = collection["entries"][0]
    proposal = json.loads(original[template["proposal_document_id"]])
    evidence_count = count - 1
    links = min(256, evidence_count)
    proposal["evidence_ids"] = [f"evidence-{index:03}" for index in range(links)]
    proposal["description"] = "bounded description " * 3000
    documents, contents = [], {}

    def add(name, body, media_type, namespace="evidence"):
        document = {
            "scope": {"kind": "run", "id": "run-a"},
            "ref": {"namespace": namespace, "name": name, "revision": "rev-large"},
            "digest": _digest(body),
            "media_type": media_type,
            "size_bytes": len(body),
        }
        document["id"] = document_id(document)
        documents.append(document)
        contents[document["id"]] = body
        return document["id"]

    proposal_id = add("proposal", _json(proposal), "application/json", "finding-proposals")
    evidence = [
        add(f"evidence-{i:04}", (f"evidence {i:04} " * 100).encode(), "text/plain")
        for i in range(evidence_count)
    ]
    collection["entries"] = []
    for index in range(math.ceil(evidence_count / links)):
        entry = copy.deepcopy(template)
        entry.update(
            {
                "receipt_id": f"receipt-{index:03}",
                "proposal_id": f"proposal-{index:03}",
                "invocation_id": f"invocation-{index:03}",
                "proposal_document_id": proposal_id,
                "evidence": [
                    {
                        "evidence_id": f"evidence-{j:03}",
                        "document_id": evidence[(index * links + j) % evidence_count],
                    }
                    for j in range(links)
                ],
            }
        )
        collection["entries"].append(entry)
    collection["documents"] = sorted(documents, key=lambda document: document["id"])
    return _package(collection, contents)


def _maximum_archive():
    collection, contents = _fixture()
    link = collection["entries"][0]["evidence"][0]
    document = next(d for d in collection["documents"] if d["id"] == link["document_id"])
    document["ref"]["revision"] = "rev-large"
    size = 1024
    for _ in range(5):
        contents.pop(document["id"])
        data = b"x" * size
        document.update({"digest": _digest(data), "size_bytes": size})
        document["id"] = document_id(document)
        contents[document["id"]] = data
        link["document_id"] = document["id"]
        collection["documents"].sort(key=lambda d: d["id"])
        payload = _package(collection, contents)
        difference = MAX_ARCHIVE_BYTES - len(payload)
        if difference == 0:
            return payload
        size += difference
    raise RuntimeError("maximum-sized archive fixture did not converge")


@pytest.mark.parametrize("size", [256, 1023, "maximum-archive"])
def test_real_reader_prepares_and_pages_large_collections_off_loop(monkeypatch, size):
    payload = _maximum_archive() if size == "maximum-archive" else _many_documents(size)

    async def scenario():
        loop_thread = threading.get_ident()
        seen = []

        def observe(owner, name):
            original = getattr(owner, name)

            def wrapped(*args):
                assert threading.get_ident() != loop_thread
                seen.append(name)
                return original(*args)

            monkeypatch.setattr(owner, name, wrapped)

        observe(reader, "decode_collection")
        observe(reader, "_project_items")
        observe(reader, "_verify_document")
        observe(reader.ListFindingsTool, "_page")
        transport = Transport(payload)
        tools, client, _ = await _tools(transport)
        expected_documents = 4 if size == "maximum-archive" else size
        assert transport.puts == expected_documents
        tool = tools["list_findings"]
        receipt_ids, cursor = [], None
        while True:
            page = await tool(limit=100, cursor=cursor)
            assert len(_json(page)) <= reader.MAX_PAGE_BYTES
            receipt_ids.extend(item["receipt_id"] for item in page["items"])
            for item in page["items"]:
                for document in [item["proposal"], *item["evidence"]]:
                    ref = ArtifactRef.model_validate(document["ref"])
                    assert (ref.namespace, ref.name, ref.revision) in transport.versions
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert receipt_ids == sorted({item["receipt_id"] for item in tool._items})
        # Reuse existing documents through the real create/read/exact-read path.
        replayed, _, _ = await _tools(transport)
        assert "_verify_document" in seen
        assert "decode_collection" in seen and "_project_items" in seen and "_page" in seen
        ref = ArtifactRef.model_validate(tool._items[0]["evidence"][0]["ref"])
        retained = await client.read_artifact(ref, max_bytes=MAX_ARCHIVE_BYTES)
        assert _digest(retained.data) == tool._items[0]["evidence"][0]["digest"]
        await replayed["list_findings"].close()
        await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", ["documents", "archive"])
def test_collection_limit_plus_one_never_materializes_any_document(overflow):
    payload = _many_documents(1024) if overflow == "documents" else b"x" * (MAX_ARCHIVE_BYTES + 1)

    async def scenario():
        transport = Transport(payload)
        with pytest.raises(FindingsError, match="findings_limit_exceeded"):
            await _tools(transport)
        assert transport.puts == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["decode", "verify", "projection", "page"])
@pytest.mark.parametrize("cancel", [False, True])
def test_findings_cpu_work_yields_loop_and_keeps_ownership_until_settled(
    monkeypatch, phase, cancel
):
    async def scenario():
        transport = Transport(_package(*_fixture()))
        tool = None
        if phase in {"verify", "page"}:
            tools, _, _ = await _tools(transport)
            tool = tools["list_findings"]
        targets = {
            "decode": (reader, "decode_collection"),
            "verify": (reader, "_verify_document"),
            "projection": (reader, "_project_items"),
            "page": (reader.ListFindingsTool, "_page"),
        }
        owner, name = targets[phase]
        original = getattr(owner, name)
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release, finished = threading.Event(), threading.Event()

        def blocked(*args):
            assert threading.get_ident() != loop_thread
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(5), "test did not release findings CPU work"
                return original(*args)
            finally:
                finished.set()

        monkeypatch.setattr(owner, name, blocked)
        pending = asyncio.create_task(tool() if phase == "page" else _tools(transport))
        ready = asyncio.create_task(started.wait())
        closing = None
        try:
            await asyncio.wait({ready, pending}, return_when=asyncio.FIRST_COMPLETED)
            if pending.done():
                pending.result()
            await ready
            requests = len(transport.requests)
            if phase == "page":
                closing = asyncio.create_task(tool.close())
            for _ in range(3):
                if cancel:
                    pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done() and not finished.is_set()
                if closing is not None:
                    assert not closing.done()
                assert len(transport.requests) == requests
        finally:
            release.set()
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert len(transport.requests) == requests
        else:
            result = await pending
            if phase != "page":
                await result[0]["list_findings"].close()
        assert finished.is_set()
        if closing is not None:
            await closing
            with pytest.raises(FindingsError, match="findings_reader_unavailable"):
                await tool()
        elif tool is not None:
            await tool.close()

    asyncio.run(scenario())
