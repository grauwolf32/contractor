"""Immutable encoded views, bounded pages, current authority and joined cleanup."""

import asyncio
import json
import threading
from pathlib import Path

import pytest
from fakes.sitemap import SitemapTransport, document, identity

import contractor_runtime.toolsets.sitemap.index as index_module
import contractor_runtime.toolsets.sitemap.session as session_module
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient, ArtifactTransportError
from contractor_runtime.sitemap import MAX_SITEMAP_BYTES
from contractor_runtime.toolsets.sitemap.index import MAX_PAGE_BYTES, MAX_VIEW_BYTES, ViewIndex
from contractor_runtime.toolsets.sitemap.session import SitemapSession


def test_shared_sitemap_document_wire_shape() -> None:
    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/project-sitemap-cases.json").read_text()
    )
    assert ViewIndex(json.dumps(cases["empty"]).encode()).entries == {}
    assert ViewIndex(json.dumps(cases["observed"]).encode()).entries
    assert ViewIndex(json.dumps(cases["reported"]).encode()).entries
    for raw in cases["invalid"]:
        with pytest.raises(ArtifactTransportError):
            ViewIndex(raw.encode())


def test_shared_absent_current_map_is_local_empty_after_authority_and_not_run_provenance():
    async def scenario():
        transport = SitemapTransport()
        transport.read_error = (404, "artifact_not_found", False)
        artifacts = ArtifactClient("allocation", transport)
        session = SitemapSession(artifacts)
        options = dict(
            cursor=None,
            refresh=False,
            limit=100,
            method=None,
            origin=None,
            path_prefix=None,
            include_suppressed=False,
        )
        first = await session.list(**options)
        assert first == {"revision": "absent", "entries": [], "nextCursor": None}
        cases = json.loads(
            (Path(__file__).parents[2] / "api/testdata/project-sitemap-cases.json").read_text()
        )
        assert json.loads(session._view.data) == cases["empty"]
        assert artifacts.known_exact_refs == () and artifacts.observation_cursor == 0
        assert len(transport.requests) == 2 and "/access?" in transport.requests[-1][1]
        # A later publication does not silently replace the empty view.
        transport.read_error = None
        assert await session.list(**options) == first
        populated = await session.list(**{**options, "refresh": True})
        assert populated["entries"] and populated["revision"] == "revision-original"
        assert artifacts.known_exact_refs == () and artifacts.observation_cursor == 0
        assert session._view is not None
        await session.close()

    asyncio.run(scenario())


def test_sitemap_domain_rejects_non_sitemap_media_on_shared_artifact_read():
    class WrongMedia(SitemapTransport):
        async def request(self, method, path, **kwargs):
            from dataclasses import replace

            response = await super().request(method, path, **kwargs)
            return replace(response, headers={**response.headers, "content-type": "text/plain"})

    async def scenario():
        session = SitemapSession(ArtifactClient("allocation", WrongMedia()))
        with pytest.raises(ArtifactTransportError, match="invalid media type"):
            await session.list(
                cursor=None,
                refresh=False,
                limit=100,
                method=None,
                origin=None,
                path_prefix=None,
                include_suppressed=False,
            )
        assert session._view is None
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        (403, "artifact_access_denied", False),
        (409, "project_deleting", False),
        (404, "not_found", False),
        (503, "storage_transaction_conflict", True),
    ],
)
def test_current_map_missing_fallback_never_hides_authority_or_project_errors(failure):
    async def scenario():
        transport = SitemapTransport()
        transport.read_error = failure
        session = SitemapSession(ArtifactClient("allocation", transport))
        with pytest.raises(ArtifactAPIError) as caught:
            await session.list(
                cursor=None,
                refresh=False,
                limit=100,
                method=None,
                origin=None,
                path_prefix=None,
                include_suppressed=False,
            )
        assert caught.value.code == failure[1]
        assert session._view is None and len(transport.requests) == 1
        await session.close()

    asyncio.run(scenario())


def test_empty_view_still_requires_current_authority_before_output():
    class RevokedAfterRead(SitemapTransport):
        async def request(self, method, path, **kwargs):
            response = await super().request(method, path, **kwargs)
            if "/map?" in path:
                self.access_error = (403, "artifact_access_denied", False)
            return response

    async def scenario():
        transport = RevokedAfterRead()
        transport.read_error = (404, "artifact_not_found", False)
        session = SitemapSession(ArtifactClient("allocation", transport))
        with pytest.raises(ArtifactAPIError, match="artifact_access_denied"):
            await session.list(
                cursor=None,
                refresh=False,
                limit=100,
                method=None,
                origin=None,
                path_prefix=None,
                include_suppressed=False,
            )
        await session.close()

    asyncio.run(scenario())


def test_generated_sitemap_wire_schema_matches_owner() -> None:
    import importlib.util

    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "sitemap_schema_generator", root / "scripts/generate_sitemap_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUTPUT.read_text() == module.generated()
    assert module.WIRE_OUTPUT.read_text() == module.generated_wire()


def test_cached_pages_survive_publication_and_eviction_but_not_refresh_or_revocation() -> None:
    async def scenario():
        transport = SitemapTransport(document(205))
        session = SitemapSession(ArtifactClient("allocation", transport))
        options = dict(
            cursor=None,
            refresh=False,
            limit=100,
            method=None,
            origin=None,
            path_prefix=None,
            include_suppressed=False,
        )
        first = await session.list(**options)
        assert len(first["entries"]) == 100 and first["revision"] == "revision-original"
        assert first["nextCursor"]
        assert set(first) == {"revision", "entries", "nextCursor"}
        original = set(row["id"] for row in first["entries"])
        # The scripted server no longer has the old bytes. A continuation must
        # not fetch them or turn to the latest publication.
        transport.document = document(1)
        transport.revision = "latest-revision"
        second = await session.list(**{**options, "cursor": first["nextCursor"]})
        third = await session.list(**{**options, "cursor": second["nextCursor"]})
        original.update(row["id"] for row in second["entries"] + third["entries"])
        assert len(original) == 205 and third["nextCursor"] is None
        assert sum("/map?" in path for _, path, _, _ in transport.requests) == 1
        assert second["revision"] == "revision-original"
        access_paths = [path for _, path, _, _ in transport.requests if "/access" in path]
        assert (
            access_paths
            == [
                "/allocations/allocation/artifacts/sitemap/map/access"
                "?scope=project&operation=list_sitemap_entries"
            ]
            * 3
        )
        for tampered in [first["nextCursor"] + "x", "garbage"]:
            with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
                await session.list(**{**options, "cursor": tampered})
        with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
            await session.list(**{**options, "cursor": first["nextCursor"], "method": "POST"})
        transport.access_error = (403, "sitemap_access_denied", False)
        with pytest.raises(ArtifactAPIError, match="sitemap_access_denied"):
            await session.list(**{**options, "cursor": first["nextCursor"]})
        transport.access_error = None
        latest = await session.list(**{**options, "refresh": True})
        assert latest["revision"] == "latest-revision"
        with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
            await session.list(**{**options, "cursor": first["nextCursor"]})
        await session.close()
        assert session._view is None and not any(session._key)
        with pytest.raises(ArtifactAPIError):
            await session.list(**options)

    asyncio.run(scenario())


def test_read_pages_concrete_parts_suppression_and_exact_snapshot_request() -> None:
    async def scenario():
        value = document(1, 3, 100)
        entry = value["groups"][0]
        marker = {
            "id": "removal",
            "members": [entry["members"][0]["id"]],
            "provenance": {"sourceId": "public", "kind": "manual"},
        }
        value["markers"] = [marker]
        transport = SitemapTransport(value)
        session = SitemapSession(ArtifactClient("allocation", transport))
        page = await session.read(
            entry["id"], cursor=None, refresh=False, limit=1, include_suppressed=False
        )
        rows = list(page["entries"])
        while page["nextCursor"]:
            page = await session.read(
                entry["id"],
                cursor=page["nextCursor"],
                refresh=False,
                limit=1,
                include_suppressed=False,
            )
            rows += page["entries"]
        assert len([row for row in rows if row["kind"] == "member"]) == 2
        assert all(row.get("memberId") != marker["members"][0] for row in rows)
        assert any(row["kind"] == "marker" for row in rows)
        assert not {"source", "rule", "gap"} & {row["kind"] for row in rows}
        requests_before_snapshot = len(transport.requests)
        await session.snapshot([entry["id"]], [], False, "stable-key")
        assert len(transport.requests) == requests_before_snapshot + 1
        _, path, headers, body = transport.requests[-1]
        request = json.loads(body)
        assert path.endswith("/snapshots") and headers["Idempotency-Key"] == "stable-key"
        assert request["selection"]["revision"] == "revision-original"
        assert set(request["selection"]) == {
            "revision",
            "groups",
            "members",
            "requireReferencedInputs",
        }
        assert "document" not in request
        await session.close()

    asyncio.run(scenario())


def test_metadata_pages_bound_work_bytes_and_cursor_authority_without_entry_repetition():
    value = document(1)
    value["sources"] = [
        {
            "provenance": {"sourceId": f"source-{i}", "kind": "manual"},
            "generation": 1,
            "policyDigest": "a" * 64,
        }
        for i in range(128)
    ]
    value["rules"] = [
        {
            "sourceId": f"source-{i % 128}",
            "origin": "https://example.test",
            "method": "GET",
            "path": f"/metadata/{i}",
            "pointer": "/" + "x" * 8000,
            "parameters": [],
            "structural": [],
        }
        for i in range(256)
    ]
    reasons = [
        "structural",
        "members",
        "groups",
        "summaries",
        "examples",
        "regrouping",
        "incomplete",
        "parameters",
        "source-unavailable",
        "ambiguous",
        "projection",
    ]
    value["gaps"] = [
        {"sourceId": source["provenance"]["sourceId"], "reason": reason, "count": 1}
        for source in value["sources"]
        for reason in reasons
    ]

    async def scenario():
        transport = SitemapTransport(value)
        session = SitemapSession(ArtifactClient("allocation", transport))
        options = dict(
            cursor=None,
            refresh=False,
            limit=100,
            method=None,
            origin=None,
            path_prefix=None,
            include_suppressed=False,
            scope="metadata",
        )
        first = await session.list(**options)
        cursor = first["nextCursor"]
        with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
            await session.list(**{**options, "cursor": cursor, "scope": "entries"})
        transport.access_error = (403, "sitemap_access_denied", False)
        with pytest.raises(ArtifactAPIError, match="sitemap_access_denied"):
            await session.list(**{**options, "cursor": cursor})
        transport.access_error = None
        # Publication does not change the already materialized sequence.
        transport.document = document(0)
        transport.revision = "new-revision"
        view = session._view
        original_load, loads = view.load, 0

        def counted_load(span):
            nonlocal loads
            loads += 1
            return original_load(span)

        view.load = counted_load
        rows = list(first["entries"])
        byte_limited_page = False
        while cursor:
            loads = 0
            page = await session.list(**{**options, "cursor": cursor})
            assert 0 < len(page["entries"]) <= 100 and loads <= 101
            assert len(json.dumps(page).encode()) <= MAX_PAGE_BYTES
            assert page["revision"] == first["revision"]
            byte_limited_page |= len(page["entries"]) < 100 and page["nextCursor"] is not None
            rows.extend(page["entries"])
            cursor = page["nextCursor"]
        assert byte_limited_page
        assert rows == (
            [{"kind": "source", "value": item} for item in value["sources"]]
            + [{"kind": "rule", "value": item} for item in value["rules"]]
            + [{"kind": "gap", "value": item} for item in value["gaps"]]
        )
        loads = 0
        entry = await session.read(
            identity("group-0"), cursor=None, refresh=False, limit=100, include_suppressed=False
        )
        assert len(entry["entries"]) == 1 and loads == 1
        assert entry["entries"][0]["kind"] == "member"
        await session.list(**{**options, "refresh": True})
        with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
            await session.list(**{**options, "cursor": first["nextCursor"]})
        await session.close()
        with pytest.raises(ArtifactAPIError, match="sitemap_cursor_expired"):
            await session.list(**options)

    asyncio.run(scenario())


def test_maximum_encoded_view_realistic_index_memory_and_bounded_page_work(monkeypatch) -> None:
    value = document(10000, 5, 500)
    encoded = json.dumps(value, separators=(",", ":")).encode()
    del value
    assert len(encoded) < MAX_SITEMAP_BYTES
    encoded += b" " * (MAX_SITEMAP_BYTES - len(encoded))
    view = ViewIndex(encoded)
    assert len(view.entries) == 10000 and len(view.member_ids) == 50000
    assert view.retained_bytes + MAX_PAGE_BYTES < MAX_VIEW_BYTES
    assert len(view.data) == MAX_SITEMAP_BYTES
    print(
        f"sitemap view encoded={len(view.data)} retained_graph={view.retained_bytes} "
        f"groups={len(view.entries)} members={len(view.member_ids)}"
    )

    async def bounded_pages():
        transport = SitemapTransport()
        session = SitemapSession(ArtifactClient("allocation", transport))
        session._view = view
        session._identity = "maximum-view"
        session._revision = "exact-revision"
        loads = 0
        original_load = view.load

        def count_load(span):
            nonlocal loads
            loads += 1
            return original_load(span)

        view.load = count_load
        try:
            page = await session.list(
                cursor=None,
                refresh=False,
                limit=100,
                method="POST",
                origin=None,
                path_prefix=None,
                include_suppressed=False,
            )
            assert page["entries"] == [] and page["nextCursor"]
            assert loads <= 100 * 7
            assert len(json.dumps(page).encode()) <= MAX_PAGE_BYTES
            assert all("/access" in path for _, path, _, _ in transport.requests)
            loads = 0
            page = await session.read(
                view.entry_ids[0], cursor=None, refresh=False, limit=2, include_suppressed=False
            )
            assert len(page["entries"]) == 2 and loads <= 2
        finally:
            view.load = original_load
            await session.close()

    asyncio.run(bounded_pages())
    with pytest.raises(ArtifactTransportError, match="encoded bound"):
        ViewIndex(encoded + b" ")
    # The total graph guard must be capable of rejecting a real indexed view.
    monkeypatch.setattr(index_module, "MAX_VIEW_BYTES", view.retained_bytes)
    with pytest.raises(ArtifactTransportError, match="total memory bound"):
        ViewIndex(encoded)


def test_cancelled_indexing_and_release_join_the_single_worker(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = session_module.ViewIndex

    def held(data):
        started.set()
        if not release.wait(30):
            raise RuntimeError("test did not release index worker")
        try:
            return original(data)
        finally:
            finished.set()

    monkeypatch.setattr(session_module, "ViewIndex", held)

    async def scenario():
        session = SitemapSession(ArtifactClient("allocation", SitemapTransport()))
        reading = asyncio.create_task(
            session.read(
                identity("group-0"), cursor=None, refresh=False, limit=100, include_suppressed=False
            )
        )
        assert await asyncio.to_thread(started.wait, 30)
        reading.cancel()
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await reading
        await closing
        assert finished.is_set() and session._view is None

    try:
        asyncio.run(scenario())
    finally:
        release.set()
