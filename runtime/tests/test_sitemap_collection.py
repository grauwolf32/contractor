from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from http_cookie_store import HTTPToolsetFactory

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.sitemap_collection import MAX_BYTES, MAX_ITEMS, SitemapCollector
from contractor_runtime.toolsets.common.http_observations import notify_http_observations
from contractor_runtime.workspace import AllocationWorkspace


def observation(number=0):
    return {
        "url": f"https://target.example/{number}",
        "method": "GET",
        "outcome": "response",
        "status": 200,
        "at": "2026-10-27T00:00:00Z",
        "sessionGeneration": "session-one",
    }


async def no_timer(_):
    await asyncio.Event().wait()


class Client:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.failure_once = False
        self.block = False
        self.entered, self.resume = asyncio.Event(), asyncio.Event()

    async def collect(self, value, key):
        self.calls.append((copy.deepcopy(value), key))
        self.entered.set()
        if self.block:
            await self.resume.wait()
        if self.failure:
            error = self.failure
            if self.failure_once:
                self.failure = None
            raise error


def test_receive_never_waits_for_storage_and_bounds_both_batches(caplog):
    async def scenario():
        client = Client()
        client.block = True
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive([observation(n) for n in range(MAX_ITEMS)], ())
        await client.entered.wait()
        # Publication is blocked; another complete pending batch is admitted.
        await collector.receive([observation(n + MAX_ITEMS) for n in range(MAX_ITEMS)], ())
        await collector.receive([observation(999)], ())
        assert len(collector._pending) == MAX_ITEMS and collector._bytes <= MAX_BYTES
        assert "buffer full" in caplog.text
        client.resume.set()
        await collector.close()
        assert len(client.calls) == 2 and collector._publisher is None and collector._timer is None
        assert collector._closed and not collector._pending
        for payload, _ in client.calls:
            assert len(payload["observations"]) == MAX_ITEMS
            assert "lost" not in payload
        await collector.receive([observation()], ())
        assert len(client.calls) == 2

    asyncio.run(scenario())


def test_ambiguous_publication_reuses_exact_input_and_key():
    async def scenario():
        client = Client()
        client.failure = ArtifactTransportError("response lost")
        client.failure_once = True
        delays = []

        async def advance(delay):
            delays.append(delay)

        collector = SitemapCollector(client, sleep=advance, wait=no_timer)
        await collector.receive([observation()], ("private-match-only",))
        assert not client.calls
        await collector.checkpoint()
        await collector.close()
        assert len(client.calls) == 2 and client.calls[0] == client.calls[1]
        assert delays == [0.25]

    asyncio.run(scenario())


def test_failure_is_logged_without_changing_read_or_completion(caplog):
    async def scenario():
        client = Client()
        client.failure = ArtifactAPIError(409, "artifact_history_protected", False)
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive([observation()], ())
        await collector.checkpoint()
        await collector.checkpoint()
        await collector.close()
        assert len(client.calls) == 1 and collector._disabled
        assert "artifact_history_protected" in [getattr(r, "code", "") for r in caplog.records]
        assert not hasattr(collector, "status")

    asyncio.run(scenario())


def test_unknown_remote_failure_text_is_never_logged(caplog):
    async def scenario():
        client = Client()
        client.failure = ArtifactAPIError(503, "private-canary/remote-text", False)
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive([observation()], ())
        await collector.close()
        assert "private-canary" not in caplog.text
        assert "sitemap_unavailable" in caplog.text
        assert all(getattr(r, "code", "") != client.failure.code for r in caplog.records)

    asyncio.run(scenario())


def test_retry_budget_is_not_renewed_and_retry_after_is_honoured():
    async def scenario():
        now, delays = [0.0], []

        async def advance(delay):
            now[0] += delay
            delays.append(delay)

        for code, count in [
            ("storage_transaction_conflict", 1),
            ("artifact_history_maintenance_pending", 3),
        ]:
            client = Client()
            client.failure = ArtifactAPIError(503, code, True)
            collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
            await collector.receive([observation()], ())
            await collector.checkpoint()
            await collector.checkpoint()
            await collector.close()
            assert len(client.calls) == count and len({key for _, key in client.calls}) == 1
        client = Client()
        client.failure = ArtifactAPIError(503, "artifact_history_maintenance_pending", True)
        client.failure.retry_after_seconds = 2
        client.failure_once = True
        collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
        await collector.receive([observation()], ())
        await collector.close()
        assert len(client.calls) == 2 and delays[-1] == 2

    asyncio.run(scenario())


def test_checkpoint_does_not_give_the_second_batch_a_fresh_budget():
    async def scenario():
        now, delays = [0.0], []
        collector = None

        class AdvancingClient(Client):
            async def collect(self, value, key):
                await super().collect(value, key)
                if len(self.calls) == 1:
                    now[0] += 14
                    await collector.receive([observation(999)], ())
                else:
                    error = ArtifactAPIError(503, "artifact_history_maintenance_pending", True)
                    error.retry_after_seconds = 2
                    raise error

        async def advance(delay):
            delays.append(delay)
            now[0] += delay

        client = AdvancingClient()
        collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
        await collector.receive([observation()], ())
        await collector.checkpoint()
        await collector.close()
        assert len(client.calls) == 2 and not delays and collector._disabled

    asyncio.run(scenario())


def test_timer_and_close_join_the_actual_publisher():
    async def scenario():
        tick = asyncio.Event()

        async def timer(_):
            await tick.wait()

        client = Client()
        client.block = True
        collector = SitemapCollector(client, wait=timer)
        await collector.receive([observation()], ())
        tick.set()
        await client.entered.wait()
        closing = asyncio.create_task(collector.close())
        await asyncio.sleep(0)
        assert not closing.done()
        client.resume.set()
        await closing
        assert len(client.calls) == 1 and collector._publisher is None and collector._timer is None

    asyncio.run(scenario())


@pytest.mark.parametrize("runtime_kind", ["adk", "tool"])
def test_shared_resource_flush_does_not_consume_worker_state_cleanup_budget(tmp_path, runtime_kind):
    from datetime import UTC, datetime, timedelta

    from fakes.clock import advance_clock, run_with_advanceable_clock
    from fakes.model import scripted_model, text_result
    from test_adk_runtime import build_context, stage_request
    from test_tool_worker import ArtifactStore, FixtureTool, request, tool_spec, worker

    from contractor_runtime.worker.runtime import AdkWorkerRuntime

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def checkpoint():
            entered.set()
            await release.wait()

        if runtime_kind == "adk":
            state = WorkerState()
            runtime = AdkWorkerRuntime(
                build_context(tmp_path, state, {}), scripted_model([text_result("done")])
            )
            await runtime.start()
            invocation = stage_request()
        else:
            spec = tool_spec()
            runtime = await worker(tmp_path, ArtifactStore(), FixtureTool(), spec=spec)
            state = runtime._state
            invocation = request(spec)
        state.before_completion = checkpoint
        task = asyncio.create_task(runtime.invoke(invocation))
        await entered.wait()
        # Advance the injected event-loop clock past State's five-second budget.
        # A separate resource deadline must not make successful execution fail.
        advance_clock(6)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        outcome = await task
        assert outcome.failure is None
        snapshot = await runtime.agent_state_snapshot()
        assert snapshot.state.last_completed_invocation.phase == "succeeded"
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    run_with_advanceable_clock(scenario())


def test_http_capture_subscriber_failure_preserves_target_result_and_physical_hops(tmp_path: Path):
    async def scenario():
        received, target = [], []

        class Sink:
            async def receive(self, observations, secrets):
                received.extend(observations)
                raise RuntimeError("storage down")

        async def handler(request):
            target.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/end"}, request=request)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/plain"},
                content=b"target result",
                request=request,
            )

        state = WorkerState()
        state.http_observation_sink = Sink()
        factory = HTTPToolsetFactory(
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        tools = await factory.create_selected(
            selected=["http_request"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example", requestTimeoutSeconds=10
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            state=state,
        )
        try:
            response = await tools["http_request"](
                "https://target.example/start", follow_redirects=True
            )
            assert response["status"] == 200 and response["body_preview"] == "target result"
            assert "sitemap" not in json.dumps(response)
            assert [o["status"] for o in received] == [302, 200]
            assert len(target) == 2
            received.clear()
            await tools["http_request"]("https://target.example/start", follow_redirects=False)
            assert [o["url"] for o in received] == ["https://target.example/start"]
            assert len(target) == 3
        finally:
            await tools["http_request"].close()

    asyncio.run(scenario())


def test_subscriber_join_preserves_caller_cancellation():
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        class Sink:
            async def receive(self, observations, secrets):
                entered.set()
                await finish.wait()

        task = asyncio.create_task(notify_http_observations(Sink(), [{}], ()))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("proxy", [False, True])
def test_http_subscriber_attributes_physical_retry_timeout_and_failure(
    tmp_path, monkeypatch, proxy
):
    import contractor_runtime.toolsets.http.tools as http_tools
    from contractor_runtime.adapters import AdapterHandles
    from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
    from contractor_runtime.toolsets.http.tools import HTTPToolError

    async def scenario():
        observations, calls = [], []

        class Sink:
            async def receive(self, rows, secrets):
                observations.extend(rows)

        async def no_delay(_):
            return

        monkeypatch.setattr(http_tools, "_retry_sleep", no_delay)

        async def handler(request):
            path = request.url.path
            calls.append(path)
            if path == "/retry" and calls.count(path) == 1:
                return httpx.Response(503, request=request)
            if path == "/timeout":
                raise httpx.ReadTimeout("test timeout", request=request)
            if path == "/down":
                raise httpx.ConnectError("test failure", request=request)
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        factory = HTTPToolsetFactory(lambda: client)
        state = WorkerState()
        state.http_observation_sink = Sink()
        tools = await factory.create_selected(
            selected=["http_request"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example", requestTimeoutSeconds=10
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            state=state,
            adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(client))
            if proxy
            else AdapterHandles(),
        )
        try:
            response = await tools["http_request"]("https://target.example/retry")
            assert response["status"] == 200
            assert [row["status"] for row in observations] == [503, 200]
            for path, outcome in [("/timeout", "timeout"), ("/down", "transport-error")]:
                observations.clear()
                with pytest.raises(HTTPToolError, match="http_request_failed"):
                    await tools["http_request"]("https://target.example" + path, method="POST")
                assert len(observations) == 1 and observations[0]["outcome"] == outcome
                assert observations[0]["status"] == 0 and calls.count(path) == 1
            observations.clear()
            with pytest.raises(HTTPToolError):
                await tools["http_request"]("invalid URL")
            assert not observations
        finally:
            await tools["http_request"].close()
            await client.aclose()

    asyncio.run(scenario())
