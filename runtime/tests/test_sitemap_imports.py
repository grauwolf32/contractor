"""Exact report commands and one bounded queue shared by every automatic source."""

import asyncio
import json
import threading

import pytest
from fakes.sitemap import SitemapTransport
from fakes.sitemap_reports import ReportTransport, report_collector
from test_sitemap_collection import no_timer, observation
from test_sitemap_tools import tools_for

from contractor_runtime.allocation.preparation import _SourceObservationSink
from contractor_runtime.artifacts import ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.factories import built_in_factories
from contractor_runtime.sitemap import SitemapClient
from contractor_runtime.sitemap_collection import MAX_BYTES, MAX_ITEMS, SitemapCollector
from contractor_runtime.toolsets.common import report_observations
from contractor_runtime.toolsets.sitemap.tools import import_sitemap_report


def report_ref(name="report"):
    return ArtifactRef(namespace="research", name=name, revision="exact")


def test_shared_installed_import_uses_exact_selection_and_reconciles_only_explicit_calls(tmp_path):
    async def scenario():
        factory = built_in_factories(tmp_path).toolsets["sitemap@1"]
        assert "import_sitemap_source" in await factory.probe()
        transport = ReportTransport()
        transport.ambiguous_once = True
        tools, _ = await tools_for(
            transport, ["import_sitemap_source"], dispatcher=import_sitemap_report
        )
        tool = tools["import_sitemap_source"]
        selection = {"format": "ffuf-report@1", "artifact": report_ref().model_dump()}
        try:
            result = await tool(selection["format"], selection["artifact"], "import-key")
            assert result["status"] == "completed"
            assert len(transport.requests) == 2
            assert transport.requests[0] == transport.requests[1]
            method, path, headers, body = transport.requests[0]
            assert method == "POST" and path.endswith("/sitemap/observations")
            assert headers["Idempotency-Key"] == "import-key"
            assert json.loads(body) == {
                "apiVersion": "contractor/v1alpha1",
                "input": {
                    "kind": "ffuf",
                    "artifact": selection["artifact"],
                    "observations": [
                        {
                            "url": "https://example.test/users?expand=profile",
                            "method": "GET",
                            "status": 200,
                            "outcome": "response",
                            "at": "2026-10-02T12:00:00Z",
                        }
                    ],
                },
                "automatic": False,
            }
            assert len(transport.reads) == 1 and transport.reads[0][1] == 512 << 10
            transport.requests.clear()
            transport.post_error = (400, "invalid_request", False)
            failed = await tool(selection["format"], selection["artifact"], "invalid-report")
            assert failed["status"] == "failed" and failed["error"]["code"] == "invalid_request"
            assert len(transport.requests) == 1
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "selection",
    [
        {"format": "raw-stdout@1", "artifact": report_ref().model_dump()},
        {"format": "ffuf-report@1", "artifact": "research/report@exact"},
        {"format": "ffuf-report@1", "artifact": {"namespace": "research", "name": "report"}},
        {
            "format": "katana-report@1",
            "artifact": {"namespace": "skills", "name": "private", "revision": "exact"},
        },
    ],
)
def test_shared_import_rejects_nonexact_unsupported_or_hidden_selection_before_io(selection):
    async def scenario():
        transport = SitemapTransport()
        tools, _ = await tools_for(
            transport, ["import_sitemap_source"], dispatcher=import_sitemap_report
        )
        try:
            with pytest.raises(ToolInputError):
                await tools["import_sitemap_source"](selection["format"], selection["artifact"])
            assert not transport.requests
        finally:
            await tools["import_sitemap_source"].close()

    asyncio.run(scenario())


def test_shared_one_queue_preserves_source_order_and_imports_only_supported_report_refs():
    async def scenario():
        transport = ReportTransport()
        collector = report_collector(transport, wait=no_timer)
        await collector.receive_source("http", [observation(1), observation(2)])
        await collector.receive_report("scan_ffuf", report_ref())
        other = _SourceObservationSink(collector, "custom-provider")
        await other.receive([observation(3), observation(4)])
        await collector.receive_report("scan_nuclei", report_ref("unselected"))
        await collector.receive_report("scan_katana", report_ref("crawl"))
        await collector.receive_source("http", [observation(5)])
        assert not transport.requests and len(collector._pending) == 7
        await collector.close()
        frames = [json.loads(body) for _, _, _, body in transport.requests]
        assert [frame.get("input", {}).get("kind", "report") for frame in frames] == [
            "http",
            "ffuf",
            "custom-provider",
            "katana",
            "http",
        ]
        assert [
            frame["input"]["artifact"]["name"] for frame in frames if "artifact" in frame["input"]
        ] == ["report", "crawl"]
        assert all(frame["automatic"] is True for frame in frames)
        assert len(transport.reads) == 2
        assert all(method == "POST" for method, _, _, _ in transport.requests)
        assert not collector._pending and collector._publisher is None

    asyncio.run(scenario())


def test_shared_large_caido_frame_admits_bounded_chunks_and_skips_only_oversized_rows(caplog):
    async def scenario():
        transport = SitemapTransport()
        collector = SitemapCollector(
            SitemapClient(ArtifactClient("allocation", transport)), wait=no_timer
        )
        rows = [
            {**observation(i), "url": "https://target.example/" + "a" * 7000 + str(i)}
            for i in range(MAX_ITEMS)
        ]
        await collector.receive_source("caido", rows)
        assert not transport.requests
        assert collector._bytes <= MAX_BYTES and len(collector._pending) <= MAX_ITEMS
        await collector.close()
        admitted = [
            row
            for _, _, _, body in transport.requests
            for row in json.loads(body)["input"]["observations"]
        ]
        assert len(transport.requests) == 2 and 0 < len(admitted) < MAX_ITEMS
        assert admitted == rows[: len(admitted)]
        assert all(len(body) <= MAX_BYTES for _, _, _, body in transport.requests)
        assert "buffer full" in caplog.text

        transport = SitemapTransport()
        collector = SitemapCollector(
            SitemapClient(ArtifactClient("allocation", transport)), wait=no_timer
        )
        await collector.receive_source(
            "caido", [{**observation(), "url": "x" * MAX_BYTES}, observation(1000)]
        )
        await collector.close()
        assert json.loads(transport.requests[0][3])["input"]["observations"] == [observation(1000)]

    asyncio.run(scenario())


def test_shared_sources_share_one_pending_capacity_while_report_publication_is_blocked(caplog):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Blocked(ReportTransport):
            async def request(self, method, path, **kwargs):
                response = await super().request(method, path, **kwargs)
                if method == "POST":
                    entered.set()
                    await release.wait()
                return response

        transport = Blocked()
        collector = report_collector(transport, wait=no_timer)
        await collector.receive_report("scan_ffuf", report_ref())
        checkpoint = asyncio.create_task(collector.checkpoint())
        await entered.wait()
        rows = [observation(i) for i in range(50)]
        await collector.receive_source("http", rows)
        rows[0]["url"] = "https://changed.example/"
        await collector.receive_source("caido", [observation(i) for i in range(49)])
        await collector.receive_report("scan_katana", report_ref("crawl"))
        await collector.receive_report("scan_ffuf", report_ref("overflow"))
        assert len(collector._pending) == MAX_ITEMS
        assert collector._bytes <= MAX_BYTES and len(transport.requests) == 1
        assert "buffer full" in caplog.text
        release.set()
        await checkpoint
        await collector.close()
        assert len(transport.requests) == 4
        frames = [json.loads(body) for _, _, _, body in transport.requests]
        assert frames[1]["input"]["observations"][0] == observation(0)
        assert frames[-1]["input"]["artifact"]["name"] == "crawl"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code", ["invalid_request", "artifact_not_found", "artifact_access_denied"]
)
def test_shared_bad_automatic_report_does_not_disable_later_http_or_valid_report(code, caplog):
    async def scenario():
        class FirstReportFails(ReportTransport):
            async def request(self, method, path, **kwargs):
                self.post_error = (400, code, False) if not self.requests else None
                return await super().request(method, path, **kwargs)

        transport = FirstReportFails()
        collector = report_collector(transport, wait=no_timer)
        await collector.receive_report("scan_ffuf", report_ref("bad"))
        await collector.receive_source("http", [observation()])
        await collector.receive_report("scan_katana", report_ref("good"))
        await collector.close()
        assert len(transport.requests) == 3 and not collector._disabled
        assert "sitemap collection failed" in caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize("code", ["artifact_history_protected", "sitemap_access_denied"])
def test_shared_global_report_publication_refusal_keeps_existing_collector_stop(code):
    async def scenario():
        transport = ReportTransport()
        transport.post_error = (409, code, False)
        collector = report_collector(transport, wait=no_timer)
        await collector.receive_report("scan_ffuf", report_ref())
        await collector.receive_source("http", [observation()])
        await collector.close()
        assert len(transport.requests) == 1 and collector._disabled

    asyncio.run(scenario())


def test_shared_automatic_import_has_three_total_attempts_with_one_unchanged_body_and_key():
    async def scenario():
        class Lost(ReportTransport):
            async def request(self, method, path, **kwargs):
                result = await super().request(method, path, **kwargs)
                if method == "POST":
                    self.reports.clear()  # Prepared retry must not reread now-evicted bytes.
                    raise ArtifactTransportError("acknowledgement lost")
                return result

        delays = []

        async def advance(delay):
            delays.append(delay)

        transport = Lost()
        collector = report_collector(transport, wait=no_timer, sleep=advance)
        await collector.receive_report("scan_ffuf", report_ref())
        await collector.close()
        assert len(transport.requests) == 3
        assert transport.requests == [transport.requests[0]] * 3
        assert delays == [0.25, 0.5] and collector._disabled
        assert len(transport.reads) == 1

    asyncio.run(scenario())


def test_shared_cancelled_close_joins_current_publisher_without_chaining_more_sources():
    async def scenario():
        first_entered, later_entered = asyncio.Event(), asyncio.Event()
        release_first, release_later = asyncio.Event(), asyncio.Event()
        now = [0.0]

        class Blocked(ReportTransport):
            async def request(self, method, path, **kwargs):
                response = await super().request(method, path, **kwargs)
                if method != "POST":
                    return response
                if len(self.requests) == 1:
                    first_entered.set()
                    await release_first.wait()
                else:
                    later_entered.set()
                    await release_later.wait()
                return response

        transport = Blocked()
        collector = report_collector(
            transport,
            wait=no_timer,
            clock=lambda: now[0],
        )
        await collector.receive_report("scan_ffuf", report_ref())
        await collector.receive_source("http", [observation()])
        await collector.receive_source("caido", [observation()])
        closing = asyncio.create_task(collector.close())
        await first_entered.wait()
        closing.cancel()
        await asyncio.sleep(0)
        now[0] = 6.0
        release_first.set()
        later = asyncio.create_task(later_entered.wait())
        await asyncio.wait({closing, later}, return_when=asyncio.FIRST_COMPLETED)
        chained = later.done()
        # Release a wrongly chained publisher too, so this regression fails
        # rather than hanging or leaking its own test tasks.
        release_later.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await collector._join_publisher()
        later.cancel()
        await asyncio.gather(later, return_exceptions=True)
        assert not chained and len(transport.requests) == 1
        assert collector._closed and collector._publisher is None and not collector._pending

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["malformed", "too-large", "wrong-media"])
def test_shared_explicit_invalid_report_returns_classified_error_without_publication(failure):
    async def scenario():
        transport = ReportTransport()
        if failure == "malformed":
            transport.reports["report"] = b'{"unexpected":true}'
        elif failure == "too-large":
            transport.reports["report"] = b"x" * ((512 << 10) + 1)
        else:
            transport.media_type = "text/plain"
        tools, _ = await tools_for(
            transport, ["import_sitemap_source"], dispatcher=import_sitemap_report
        )
        tool = tools["import_sitemap_source"]
        try:
            result = await tool("ffuf-report@1", report_ref().model_dump(), "bad-report")
            assert result["status"] == "failed"
            assert result["error"] == {"code": "invalid_request", "retryable": False}
            assert len(transport.reads) == 1 and not transport.requests
        finally:
            await tool.close()

    asyncio.run(scenario())


def test_shared_explicit_retry_uses_prepared_batch_but_new_invocation_rereads_evicted_source():
    async def scenario():
        class Evicted(ReportTransport):
            async def request(self, method, path, **kwargs):
                if method == "POST":
                    self.reports.clear()
                return await super().request(method, path, **kwargs)

        transport = Evicted()
        transport.ambiguous_once = True
        tools, _ = await tools_for(
            transport, ["import_sitemap_source"], dispatcher=import_sitemap_report
        )
        tool = tools["import_sitemap_source"]
        try:
            result = await tool("ffuf-report@1", report_ref().model_dump(), "same-key")
            assert result["status"] == "completed"
            assert len(transport.reads) == 1 and len(transport.requests) == 2
            assert transport.requests[0] == transport.requests[1]
            result = await tool("ffuf-report@1", report_ref().model_dump(), "same-key")
            assert result["status"] == "failed"
            assert result["error"]["code"] == "artifact_not_found"
            assert len(transport.reads) == 2 and len(transport.requests) == 2
        finally:
            await tool.close()

    asyncio.run(scenario())


def test_shared_malformed_automatic_report_drops_only_that_item_before_http_and_valid_report():
    async def scenario():
        transport = ReportTransport()
        transport.reports["bad"] = b"{}"
        collector = report_collector(transport, wait=no_timer)
        await collector.receive_report("scan_ffuf", report_ref("bad"))
        await collector.receive_source("http", [observation()])
        await collector.receive_report("scan_katana", report_ref("good"))
        await collector.close()
        assert len(transport.reads) == 2 and not collector._disabled
        assert [json.loads(body)["input"]["kind"] for _, _, _, body in transport.requests] == [
            "http",
            "katana",
        ]

    asyncio.run(scenario())


def test_shared_parser_cancellation_joins_off_thread_work(monkeypatch):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        original = report_observations.project_report

        def held(*args):
            entered.set()
            release.wait()
            return original(*args)

        monkeypatch.setattr(report_observations, "project_report", held)
        task = asyncio.create_task(
            report_observations.prepare_report(
                ArtifactClient("allocation", ReportTransport()), "ffuf-report@1", report_ref()
            )
        )
        try:
            await asyncio.to_thread(entered.wait)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_shared_collector_keeps_gaps_without_rows_and_aggregates_one_source_batch():
    async def scenario():
        transport = SitemapTransport()
        collector = SitemapCollector(
            SitemapClient(ArtifactClient("allocation", transport)), wait=no_timer
        )
        await collector.receive_source("caido", [], gaps=[{"reason": "projection", "count": 1}])
        await collector.receive_source(
            "caido", [observation()], gaps=[{"reason": "projection", "count": 2}]
        )
        await collector.close()
        assert len(transport.requests) == 1
        assert json.loads(transport.requests[0][3])["input"] == {
            "kind": "caido",
            "observations": [observation()],
            "gaps": [{"reason": "projection", "count": 3}],
        }
        await SitemapClient(ArtifactClient("allocation", transport)).collect(
            {"kind": "future-source", "observations": []}, "future", automatic=False
        )
        frame = json.loads(transport.requests[-1][3])
        assert frame["input"]["kind"] == "future-source" and frame["automatic"] is False

    asyncio.run(scenario())
