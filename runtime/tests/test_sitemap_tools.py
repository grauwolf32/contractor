"""Production factory callers and separation of API outcomes from infrastructure failures."""

import asyncio
import json
from pathlib import Path

import pytest
from fakes.sitemap import SitemapTransport, document, identity
from fakes.worker_state import WorkerState

from contractor_runtime.artifacts import ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.sitemap.session import manual_source_id
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
        resources=state.resources,
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
        result = await tools["upsert_sitemap_entry"].contractor_trusted_call(
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
        trusted = tool.contractor_trusted_call
        await trusted("revision-original", [identity("member")], "marker", idempotency_key="key")
        assert len(transport.requests) == 2
        assert transport.requests[0] == transport.requests[1]
        transport.requests.clear()
        transport.post_error = (409, "sitemap_conflict", False)
        result = await trusted(
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
        result = await trusted(
            "revision-original", [identity("member")], "marker", idempotency_key="retry-key"
        )
        assert result["error"] == {
            "code": "artifact_history_maintenance_pending",
            "retryable": True,
        }
        assert len(transport.requests) == 1  # shared storage owner owns the next bounded attempt
        await tool.close()

    asyncio.run(scenario())


def test_known_short_encoded_credentials_remain_metadata_but_not_tool_metrics():
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
        paths = ["/tiny", "/t%69ny", "/t%2569ny", "/alice", "/token/" + "a" * 64]
        for path in paths:
            result = await tools["upsert_sitemap_entry"].contractor_trusted_call(
                "absent", url="https://example.test" + path, method="GET"
            )
            assert result["status"] == "completed"
            request = json.loads(transport.requests[-1][3])
            assert request["mutation"]["url"] == "https://example.test" + path
        note = "token=tiny; password=alice"
        tags = ["tiny", "alice", "authorization"]
        result = await tools["upsert_sitemap_entry"].contractor_trusted_call(
            "revision-original", member_ids=[identity("member")], note=note, tags=tags
        )
        assert result["status"] == "completed"
        request = json.loads(transport.requests[-1][3])
        assert request["mutation"]["note"] == note and request["mutation"]["tags"] == tags
        assert len(transport.requests) == len(paths) + 1
        recorded = repr(state.metrics.tool_calls)
        for secret in [*paths, note, *tags]:
            assert secret not in recorded
        await tools["upsert_sitemap_entry"].close()

    asyncio.run(scenario())


def test_explicit_sitemap_arguments_keep_the_nesting_bound():
    async def scenario():
        transport = SitemapTransport()
        tools, _ = await tools_for(transport, ["configure_sitemap_grouping"])
        rules = []
        for _ in range(33):
            rules = [rules]
        with pytest.raises(ToolInputError, match="nesting bound"):
            await tools["configure_sitemap_grouping"].contractor_trusted_call(
                "revision-original", rules
            )
        assert transport.requests == []
        await tools["configure_sitemap_grouping"].close()

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


def test_empty_map_metadata_supplies_complete_source_precondition_to_editor():
    from google.adk.tools import FunctionTool

    async def scenario():
        source = {
            "provenance": {
                "sourceId": manual_source_id("research"),
                "kind": "manual",
            },
            "generation": 2,
            "policyDigest": "b" * 64,
        }
        value = document(0)
        value["sources"] = [source]
        transport = SitemapTransport(value)
        tools, _ = await tools_for(
            transport, ["list_sitemap_entries", "configure_sitemap_grouping"]
        )
        listing = tools["list_sitemap_entries"]
        declaration = FunctionTool(listing)._get_declaration()
        assert declaration.parameters_json_schema["properties"]["scope"]["enum"] == [
            "entries",
            "metadata",
        ]
        assert "scope" not in declaration.parameters_json_schema.get("required", [])
        assert (await listing())["entries"] == []
        first = await listing(scope="metadata", limit=1)
        assert first["entries"] == [{"kind": "source", "value": source}]
        second = await listing(scope="metadata", limit=1, cursor=first["nextCursor"])
        assert second["entries"] == [{"kind": "gap", "value": value["gaps"][0]}]
        assert second["nextCursor"] is None
        rules = [
            {
                "origin": "https://example.test",
                "method": "GET",
                "path": "/new",
                "pointer": "",
                "parameters": [],
                "structural": [],
            }
        ]
        before = len(transport.requests)
        result = await tools["configure_sitemap_grouping"](rules)
        assert result["status"] == "completed"
        assert len(transport.requests) == before + 1
        command = json.loads(transport.requests[-1][3])["mutation"]
        assert command["source"]["expected"] == source
        assert command["source"]["rules"] == rules
        assert set(command["source"]) == {"rules", "expected"}
        for tool in tools.values():
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"scope": "unknown"},
        {"scope": "metadata", "method": "GET"},
        {"scope": "metadata", "origin": "https://example.test"},
        {"scope": "metadata", "path_prefix": "/"},
        {"scope": "metadata", "include_suppressed": True},
    ],
)
def test_metadata_scope_rejects_entry_filters_before_fetching(arguments):
    async def scenario():
        transport = SitemapTransport()
        tools, _ = await tools_for(transport, ["list_sitemap_entries"])
        with pytest.raises(ToolInputError):
            await tools["list_sitemap_entries"](**arguments)
        assert transport.requests == []
        await tools["list_sitemap_entries"].close()

    asyncio.run(scenario())


def test_import_dispatch_seam_is_explicit_and_uses_same_allocation_client():
    async def scenario():
        calls = []

        async def dispatcher(client, selection, key):
            calls.append((selection, key, client))
            return {"ok": True, "gaps": []}

        transport = SitemapTransport()
        tools, _ = await tools_for(transport, ["import_sitemap_source"], dispatcher=dispatcher)
        result = await tools["import_sitemap_source"].contractor_trusted_call(
            "test-only",
            {"namespace": "inputs", "name": "report", "revision": "exact"},
            "import-key",
        )
        assert (
            result == {"ok": True, "gaps": [], "status": "completed"}
            and calls[0][1] == "import-key"
        )
        assert calls[0][2] is tools["import_sitemap_source"]._client
        assert transport.requests == []
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
            tool.contractor_trusted_call("absent", url="https://example.test/write", method="GET")
        )
        await started.wait()
        closing = asyncio.create_task(tool.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        assert (await writing)["status"] == "completed"
        await closing
        calls = len(transport.requests)
        denied = await tool.contractor_trusted_call(
            "absent", url="https://example.test/late", method="GET"
        )
        assert denied["status"] == "failed" and denied["error"]["code"] == "sitemap_cursor_expired"
        assert len(transport.requests) == calls

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["list_sitemap_entries", "read_sitemap_entry"])
def test_cached_page_rechecks_its_exact_operation_when_only_editor_and_collection_remain(operation):
    async def scenario():
        transport = SitemapTransport(document(3, 3))
        transport.collect_observations = True
        transport.access_operations = {
            "list_sitemap_entries",
            "read_sitemap_entry",
            "configure_sitemap_grouping",
        }
        tools, _ = await tools_for(transport, [operation, "configure_sitemap_grouping"])
        arguments = {"limit": 1}
        if operation == "read_sitemap_entry":
            arguments["entry_id"] = identity("group-0")
        first = await tools[operation](**arguments)
        assert first["status"] == "completed" and first["nextCursor"]
        transport.access_operations = {"configure_sitemap_grouping"}
        rejected = await tools[operation](**arguments, cursor=first["nextCursor"])
        assert rejected == {
            "status": "failed",
            "ok": False,
            "error": {"code": "artifact_access_denied", "retryable": False},
        }
        assert transport.requests[-1][1].endswith("/access?scope=project&operation=" + operation)
        assert sum("/map?" in path for _, path, _, _ in transport.requests) == 1
        for tool in tools.values():
            await tool.close()

    asyncio.run(scenario())


def test_model_edits_require_review_and_keep_old_pages_without_edit_authority():
    from contractor_runtime.worker.function_tools import ContractorFunctionTool

    async def scenario():
        transport = SitemapTransport(document(105))
        tools, _ = await tools_for(transport, ["list_sitemap_entries", "upsert_sitemap_entry"])
        edit, listing = tools["upsert_sitemap_entry"], tools["list_sitemap_entries"]
        schema = (
            ContractorFunctionTool(edit, logical_invocation_id=lambda _: "logical")
            ._get_declaration()
            .parameters_json_schema
        )
        assert "expected_revision" not in schema["properties"]
        with pytest.raises(ToolInputError, match="Read/list"):
            await edit(url="https://example.test/new", method="GET")
        assert transport.requests == []
        first = await listing(limit=1)
        result = await edit(url="https://example.test/new", method="GET")
        assert result["artifact"]["revision"] == "accepted-revision"
        assert (
            json.loads(transport.requests[-1][3])["mutation"]["expectedRevision"]
            == first["revision"]
        )
        # Reading an old continuation cannot silently authorize another edit.
        await listing(limit=1, cursor=first["nextCursor"])
        with pytest.raises(ToolInputError, match="refresh"):
            await edit(url="https://example.test/other", method="GET")
        await listing(refresh=True)
        transport.post_error = (409, "sitemap_conflict", False)
        rejected = await edit(url="https://example.test/other", method="GET")
        assert rejected["error"]["code"] == "sitemap_conflict"
        with pytest.raises(ToolInputError, match="refresh"):
            await edit(url="https://example.test/other", method="GET")

    asyncio.run(scenario())


def test_shared_manual_editor_identity_matches_fixture():
    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/sitemap-editor-source-id.json").read_text()
    )
    for case in cases:
        assert manual_source_id(case["namespace"], case["deployment"]) == case["sourceId"]


def test_ambiguous_model_edit_reuses_frozen_precondition_after_refresh():
    class LostReplies(SitemapTransport):
        lost = True

        async def request(self, method, path, **kwargs):
            if method == "POST" and self.lost:
                self.requests.append((method, path, dict(kwargs["headers"]), kwargs["body"]))
                raise ArtifactTransportError("Acknowledgement unavailable")
            return await super().request(method, path, **kwargs)

    async def scenario():
        transport = LostReplies()
        tools, _ = await tools_for(transport, ["list_sitemap_entries", "upsert_sitemap_entry"])
        await tools["list_sitemap_entries"]()
        edit = tools["upsert_sitemap_entry"]
        with pytest.raises(ArtifactTransportError):
            await edit(url="https://example.test/new", method="GET", idempotency_key="same")
        original = transport.requests[-1]
        transport.lost = False
        transport.revision = "newer-current"
        await tools["list_sitemap_entries"](refresh=True)
        with pytest.raises(ToolInputError, match="pending"):
            await edit(url="https://example.test/different", method="GET", idempotency_key="other")
        await edit(url="https://example.test/new", method="GET", idempotency_key="same")
        assert transport.requests[-1] == original

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["api", "transport"])
@pytest.mark.parametrize(
    "name,arguments",
    [
        ("upsert_sitemap_entry", {"url": "https://example.test/new", "method": "GET"}),
        ("remove_sitemap_entry", {"member_ids": [identity("member")], "marker_id": "marker"}),
        ("restore_sitemap_entry", {"marker_id": "marker"}),
        ("configure_sitemap_grouping", {"rules": []}),
    ],
)
def test_keyless_model_edit_reconciles_original_intent_after_refresh(name, arguments, failure):
    class Failing(SitemapTransport):
        lose_replies = False

        async def request(self, method, path, **kwargs):
            if method == "POST" and self.lose_replies:
                self.requests.append((method, path, dict(kwargs["headers"]), kwargs["body"]))
                raise ArtifactTransportError("Acknowledgement unavailable")
            return await super().request(method, path, **kwargs)

    async def scenario():
        transport = Failing()
        tools, _ = await tools_for(
            transport, sorted({"list_sitemap_entries", "upsert_sitemap_entry", name})
        )
        edit, listing = tools[name], tools["list_sitemap_entries"]
        try:
            await listing()
            if failure == "api":
                transport.post_error = (503, "storage_transaction_conflict", True)
                assert (await edit(**arguments))["error"]["retryable"] is True
            else:
                transport.lose_replies = True
                with pytest.raises(ArtifactTransportError):
                    await edit(**arguments)
                assert transport.requests[-2] == transport.requests[-1]
            original = transport.requests[-1]
            transport.post_error, transport.lose_replies = None, False
            transport.revision = "newer-current"
            await listing(refresh=True)
            with pytest.raises(ToolInputError, match="pending"):
                await edit(**arguments, idempotency_key="different")
            with pytest.raises(ToolInputError, match="pending"):
                await tools["upsert_sitemap_entry"](
                    url="https://example.test/different", method="GET"
                )
            assert (await edit(**arguments))["status"] == "completed"
            assert transport.requests[-1] == original
            await listing(refresh=True)
            await edit(**arguments)
            assert transport.requests[-1][2]["Idempotency-Key"] != original[2]["Idempotency-Key"]
        finally:
            await edit.close()

    asyncio.run(scenario())


def test_project_sitemap_read_does_not_select_run_evidence():
    async def scenario():
        transport = SitemapTransport()
        tools, state = await tools_for(transport, ["list_sitemap_entries"])
        listing, versions = tools["list_sitemap_entries"], state.resources.selected_versions
        selector = {"namespace": "sitemap", "name": "map"}
        try:
            await listing()
            with pytest.raises(ToolInputError, match="not selected"):
                versions.resolve(selector)
            run_ref = ArtifactRef(**selector, revision="reviewed-run-revision")
            versions.observe(run_ref)
            await listing(refresh=True)
            versions.begin_response()
            assert versions.resolve(selector) == run_ref
            assert versions.base("sitemap", "map") == run_ref.revision
            assert listing.known_exact_refs == ()
        finally:
            await listing.close()

    asyncio.run(scenario())
