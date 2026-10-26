"""Production factory callers and separation of API outcomes from infrastructure failures."""

import asyncio
import json
from pathlib import Path

import pytest
from fakes.sitemap import SitemapTransport, document, identity

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.sitemap.tools import SitemapToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


async def tools_for(transport, selected, *, settings=None, dispatcher=None):
    state = WorkerState()
    factory = SitemapToolsetFactory(
        lambda _allocation, _settings: ArtifactClient("allocation", transport), dispatcher
    )
    tools = await factory.create_selected(
        selected=selected,
        allocation_id="allocation",
        run_id="consumer",
        namespace="research",
        runtime_settings=settings
        or RuntimeSettings(artifact_api_url="https://control.invalid", request_timeout_seconds=10),
        workspace=AllocationWorkspace("allocation", Path("/unused")),
        state=state,
    )
    return tools, state


def test_selected_factory_capability_and_exact_private_mutation():
    async def scenario():
        transport = SitemapTransport()
        factory = SitemapToolsetFactory()
        assert "import_sitemap_source" not in await factory.probe()
        tools, state = await tools_for(transport, ["upsert_sitemap_entry"])
        assert set(tools) == {"upsert_sitemap_entry"}
        assert "Returns:" in tools["upsert_sitemap_entry"].__doc__
        result = await tools["upsert_sitemap_entry"](
            "absent", url="https://example.test/manual", method="GET", idempotency_key="stable-key"
        )
        assert result["artifact"]["revision"] == "accepted-revision"
        method, path, headers, body = transport.requests[-1]
        assert method == "POST" and path.endswith("/sitemap/mutations")
        assert headers["Idempotency-Key"] == "stable-key"
        assert json.loads(body) == {
            "apiVersion": "contractor/v1alpha1",
            "mutation": {
                "operation": "declare",
                "expectedRevision": "absent",
                "url": "https://example.test/manual",
                "method": "GET",
            },
        }
        assert len(transport.requests) == 1  # editor selection does not fetch a reader view
        assert state.metrics.counters["tool_calls"] == 1
        await tools["upsert_sitemap_entry"].close()
        with pytest.raises(ValueError):
            await tools_for(transport, ["import_sitemap_source"])

    asyncio.run(scenario())


def test_keyed_transport_reconciliation_does_not_retry_definite_errors_or_target_calls():
    async def scenario():
        transport = SitemapTransport()
        transport.ambiguous_once = True
        tools, _ = await tools_for(transport, ["remove_sitemap_entry"])
        tool = tools["remove_sitemap_entry"]
        await tool("revision-original", [identity("member")], "marker", idempotency_key="key")
        assert len(transport.requests) == 2
        assert transport.requests[0] == transport.requests[1]
        transport.requests.clear()
        transport.post_error = (409, "sitemap_conflict", False)
        result = await tool(
            "stale-revision", [identity("member")], "marker", idempotency_key="different-key"
        )
        assert result == {
            "status": "failed",
            "ok": False,
            "error": {"code": "sitemap_conflict", "retryable": False},
        }
        assert len(transport.requests) == 1
        transport.requests.clear()
        transport.post_error = (503, "artifact_history_maintenance_pending", True)
        result = await tool(
            "revision-original", [identity("member")], "marker", idempotency_key="retry-key"
        )
        assert result["error"] == {
            "code": "artifact_history_maintenance_pending",
            "retryable": True,
        }
        assert len(transport.requests) == 1  # shared storage owner owns the next bounded attempt
        await tool.close()

    asyncio.run(scenario())


def test_known_short_encoded_credentials_never_enter_persistence_or_metrics():
    async def scenario():
        transport = SitemapTransport()
        settings = RuntimeSettings.model_validate(
            {
                "artifactApiUrl": "https://control.invalid",
                "requestTimeoutSeconds": 10,
                "httpOriginTarget": {
                    "url": "https://example.test",
                    "basicAuth": {"username": "alice", "password": "tiny"},
                },
            }
        )
        tools, state = await tools_for(transport, ["upsert_sitemap_entry"], settings=settings)
        for path in ["/tiny", "/t%69ny", "/t%2569ny", "/alice"]:
            with pytest.raises(ToolInputError, match="private credential"):
                await tools["upsert_sitemap_entry"](
                    "absent", url="https://example.test" + path, method="GET"
                )
        assert transport.requests == []
        recorded = repr(state.metrics.tool_calls)
        for secret in ["tiny", "t%69ny", "t%2569ny", "alice"]:
            assert secret not in recorded
        await tools["upsert_sitemap_entry"].close()

    asyncio.run(scenario())


def test_cached_page_authority_error_is_a_tool_result_but_bad_wire_is_exception():
    async def scenario():
        transport = SitemapTransport(document(105))
        tools, _ = await tools_for(
            transport, ["list_sitemap_entries", "read_sitemap_entry", "snapshot_sitemap_context"]
        )
        first = await tools["list_sitemap_entries"]()
        transport.access_error = (409, "allocation_write_fenced", False)
        denied = await tools["list_sitemap_entries"](cursor=first["nextCursor"])
        assert denied["ok"] is False and denied["error"]["code"] == "allocation_write_fenced"
        transport.access_error = None
        transport.document = {"version": "invalid"}
        with pytest.raises(ArtifactTransportError):
            await tools["list_sitemap_entries"](refresh=True)
        assert tools["list_sitemap_entries"]._session._view is None
        for tool in tools.values():
            await tool.close()

    asyncio.run(scenario())


def test_import_dispatch_seam_is_explicit_and_uses_same_allocation_client():
    async def scenario():
        calls = []

        async def dispatcher(client, selection, key):
            access = await client.access()
            calls.append((selection, key, access.manual_source_id))
            return {"ok": True, "gaps": []}

        transport = SitemapTransport()
        tools, _ = await tools_for(transport, ["import_sitemap_source"], dispatcher=dispatcher)
        result = await tools["import_sitemap_source"](
            {
                "format": "test-only",
                "ref": {"namespace": "inputs", "name": "report", "revision": "exact"},
            },
            "import-key",
        )
        assert (
            result == {"ok": True, "gaps": [], "status": "completed"}
            and calls[0][1] == "import-key"
        )
        assert all("foreign" not in path for _, path, _, _ in transport.requests)
        await tools["import_sitemap_source"].close()

    asyncio.run(scenario())


def test_close_joins_inflight_mutation_and_denies_later_calls():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class Held(SitemapTransport):
            async def request(self, method, path, **kwargs):
                if method == "POST":
                    started.set()
                    await release.wait()
                return await super().request(method, path, **kwargs)

        transport = Held()
        tools, _ = await tools_for(transport, ["upsert_sitemap_entry"])
        tool = tools["upsert_sitemap_entry"]
        writing = asyncio.create_task(
            tool("absent", url="https://example.test/write", method="GET")
        )
        await started.wait()
        closing = asyncio.create_task(tool.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        assert (await writing)["status"] == "completed"
        await closing
        calls = len(transport.requests)
        denied = await tool("absent", url="https://example.test/late", method="GET")
        assert denied["status"] == "failed" and denied["error"]["code"] == "sitemap_cursor_expired"
        assert len(transport.requests) == calls

    asyncio.run(scenario())
