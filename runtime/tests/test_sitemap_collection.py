from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import grpc
import httpx
import pytest
from fakes.sitemap import SitemapTransport
from fakes.worker_state import WorkerState
from http_cookie_store import HTTPToolsetFactory

from contractor_runtime.allocation.preparation import _SourceObservationSink
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.sitemap import SitemapClient
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
        await collector.receive_source("http", [observation(n) for n in range(MAX_ITEMS)])
        await client.entered.wait()
        # Publication is blocked; another complete pending batch is admitted.
        await collector.receive_source(
            "http", [observation(n + MAX_ITEMS) for n in range(MAX_ITEMS)]
        )
        await collector.receive_source("http", [observation(999)])
        assert len(collector._pending) == MAX_ITEMS and collector._bytes <= MAX_BYTES
        assert "buffer full" in caplog.text
        client.resume.set()
        await collector.close()
        assert len(client.calls) == 2 and collector._publisher is None and collector._timer is None
        assert collector._closed and not collector._pending
        for payload, _ in client.calls:
            assert len(payload["observations"]) == MAX_ITEMS
            assert "lost" not in payload
        await collector.receive_source("http", [observation()])
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
        await collector.receive_source("http", [observation()])
        assert not client.calls
        await collector.checkpoint()
        await collector.close()
        assert len(client.calls) == 2 and client.calls[0] == client.calls[1]
        assert delays == [0.25]

    asyncio.run(scenario())


def test_observation_bytes_still_bound_batches_without_matching_secrets():
    async def scenario():
        client = Client()
        collector = SitemapCollector(client, wait=no_timer)
        row = {**observation(), "url": "https://target.example/" + "a" * 7000}
        await collector.receive_source("http", [row] * 35)
        assert collector._bytes <= MAX_BYTES and not client.calls
        await collector.receive_source("http", [row] * 5)
        await collector.close()
        assert len(client.calls) == 2
        assert sum(len(payload["observations"]) for payload, _ in client.calls) == 40
        for payload, _ in client.calls:
            assert set(payload) == {"kind", "observations"} and payload["kind"] == "http"
            encoded = json.dumps(
                {"apiVersion": "contractor/v1alpha1", "input": payload, "automatic": True},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            assert len(encoded) <= MAX_BYTES

    asyncio.run(scenario())


def test_http_collection_sends_only_attempt_metadata_and_preserves_credential_like_url(tmp_path):
    async def scenario():
        transport = SitemapTransport()
        collector = SitemapCollector(
            SitemapClient(ArtifactClient("allocation", transport)), wait=no_timer
        )
        url = "https://target.example/token/tiny/" + "a" * 64 + "?password=tiny"

        async def handler(request):
            assert str(request.url) == url
            assert request.headers["x-api-key"] == "header-only-token"
            assert request.content == b"request-body-only"
            return httpx.Response(
                200,
                headers={"set-cookie": "session=response-cookie-only; Path=/"},
                content=b"response-body-only",
                request=request,
            )

        state = WorkerState()
        state.resources.http_observation_sink = _SourceObservationSink(collector, "http")
        factory = HTTPToolsetFactory(
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        tools = await factory.create_selected(
            selected=["http_request"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example",
                requestTimeoutSeconds=10,
                httpOriginTarget={
                    "url": "https://target.example",
                    "basicAuth": {"username": "alice", "password": "tiny"},
                },
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            resources=state.resources,
        )
        try:
            result = await tools["http_request"](
                url,
                method="POST",
                headers={"X-API-Key": "header-only-token"},
                body_type="text",
                body="request-body-only",
            )
            assert result["status"] == 200
            await collector.checkpoint()
            assert len(transport.requests) == 1
            method, path, _, body = transport.requests[0]
            assert method == "POST" and path.endswith("/sitemap/observations")
            request = json.loads(body)
            assert set(request) == {"apiVersion", "input", "automatic"}
            assert request["automatic"] is True
            assert set(request["input"]) == {"kind", "observations"}
            assert request["input"]["kind"] == "http"
            rows = request["input"]["observations"]
            assert len(rows) == 1 and rows[0]["url"] == url
            assert set(rows[0]) == {"url", "method", "outcome", "status", "at"}
            for excluded in (
                "header-only-token",
                "request-body-only",
                "response-body-only",
                "response-cookie-only",
            ):
                assert excluded.encode() not in body
            assert url not in repr(state.metrics.tool_calls)
        finally:
            await tools["http_request"].close()
            await collector.close()

    asyncio.run(scenario())


def test_collection_keeps_http_sessions_and_request_tags_without_identity_metadata(tmp_path):
    async def scenario():
        client = Client()
        collector = SitemapCollector(client, wait=no_timer)
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(200, request=request)

        state = WorkerState()
        state.resources.http_observation_sink = _SourceObservationSink(collector, "http")
        factory = HTTPToolsetFactory(
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        tools = await factory.create_selected(
            selected=["http_request", "http_session_set", "http_session_clear"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example",
                requestTimeoutSeconds=10,
                httpOriginTarget={"url": "https://target.example"},
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            resources=state.resources,
        )
        try:
            url = "https://target.example/resource"
            await tools["http_request"](url)
            for identity in ("alice", "bob"):
                await tools["http_session_set"](
                    auth={"kind": "bearer", "token": identity}, cookies={"sid": identity}
                )
                await tools["http_request"](url)
            await tools["http_session_clear"]()
            await tools["http_request"](url)
            await collector.checkpoint()

            assert [request.headers.get("authorization") for request in requests] == [
                None,
                "Bearer alice",
                "Bearer bob",
                None,
            ]
            assert [request.headers.get("cookie") for request in requests] == [
                None,
                "sid=alice",
                "sid=bob",
                None,
            ]
            tags = [request.headers["x-request-id"] for request in requests]
            assert len(set(tags)) == 4
            assert [tag.rsplit("-", 1)[1] for tag in tags] == [
                "h000001",
                "h000002",
                "h000003",
                "h000004",
            ]
            observations = [row for payload, _ in client.calls for row in payload["observations"]]
            assert len(observations) == 4
            for row in observations:
                assert set(row) == {"url", "method", "outcome", "status", "at"}
                assert row["url"] == url and row["status"] == 200
            assert "alice" not in json.dumps(observations)
            assert "bob" not in json.dumps(observations)
        finally:
            await tools["http_request"].close()
            await collector.close()

    asyncio.run(scenario())


def test_failure_is_logged_without_changing_read_or_completion(caplog):
    async def scenario():
        client = Client()
        client.failure = ArtifactAPIError(
            grpc.StatusCode.FAILED_PRECONDITION, "artifact_history_protected", False
        )
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive_source("http", [observation()])
        await collector.checkpoint()
        await collector.checkpoint()
        await collector.close()
        assert len(client.calls) == 1 and collector._disabled
        assert "artifact_history_protected" in [getattr(r, "code", "") for r in caplog.records]
        assert not hasattr(collector, "status")

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["list_sitemap_entries", "read_sitemap_entry"])
@pytest.mark.parametrize(
    "read_error",
    [None, (403, "sitemap_access_denied", False), (503, "storage_transaction_conflict", True)],
)
def test_fresh_read_keeps_its_allowance_after_automatic_flush_exhaustion(operation, read_error):
    from fakes.clock import advance_clock, run_with_advanceable_clock
    from fakes.sitemap import identity
    from test_sitemap_tools import tools_for

    async def scenario():
        client = Client()
        client.block = True
        collector = SitemapCollector(client, clock=asyncio.get_running_loop().time, wait=no_timer)
        await collector.receive_source("http", [observation()])
        transport = SitemapTransport()
        transport.access_error = read_error
        tools, _ = await tools_for(transport, [operation])
        tool = tools[operation]
        tool._session._before_read = collector.checkpoint
        arguments = {} if operation == "list_sitemap_entries" else {"entry_id": identity("group-0")}
        reading = asyncio.create_task(tool(**arguments))
        await client.entered.wait()
        advance_clock(16)
        result = await reading
        if read_error is None:
            assert result["status"] == "completed" and result["entries"]
        else:
            assert result["status"] == "failed"
            assert result["error"] == {"code": read_error[1], "retryable": read_error[2]}
        assert any("/artifacts/sitemap/map?" in path for _, path, _, _ in transport.requests)
        assert collector._disabled and collector._publisher is None
        await tool(refresh=True, **arguments)
        assert len(client.calls) == 1
        await collector.close()
        await tool.close()

    run_with_advanceable_clock(scenario())


def test_cancelled_fresh_read_joins_checkpoint_before_propagating():
    from test_sitemap_tools import tools_for

    async def scenario():
        client = Client()
        client.block = True
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive_source("http", [observation()])
        transport = SitemapTransport()
        tools, _ = await tools_for(transport, ["list_sitemap_entries"])
        tool = tools["list_sitemap_entries"]
        tool._session._before_read = collector.checkpoint
        reading = asyncio.create_task(tool())
        await client.entered.wait()
        reading.cancel()
        await asyncio.sleep(0)
        assert not reading.done()
        client.resume.set()
        with pytest.raises(asyncio.CancelledError):
            await reading
        assert transport.requests == [] and collector._publisher is None
        await collector.close()
        await tool.close()

    asyncio.run(scenario())


def test_unknown_remote_failure_text_is_never_logged(caplog):
    async def scenario():
        client = Client()
        client.failure = ArtifactAPIError(
            grpc.StatusCode.UNAVAILABLE, "private-canary/remote-text", False
        )
        collector = SitemapCollector(client, wait=no_timer)
        await collector.receive_source("http", [observation()])
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
            client.failure = ArtifactAPIError(grpc.StatusCode.UNAVAILABLE, code, True)
            collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
            await collector.receive_source("http", [observation()])
            await collector.checkpoint()
            await collector.checkpoint()
            await collector.close()
            assert len(client.calls) == count and len({key for _, key in client.calls}) == 1
        client = Client()
        client.failure = ArtifactAPIError(
            grpc.StatusCode.UNAVAILABLE, "artifact_history_maintenance_pending", True
        )
        client.failure.retry_after_seconds = 2
        client.failure_once = True
        collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
        await collector.receive_source("http", [observation()])
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
                    await collector.receive_source("http", [observation(999)])
                else:
                    error = ArtifactAPIError(
                        grpc.StatusCode.UNAVAILABLE, "artifact_history_maintenance_pending", True
                    )
                    error.retry_after_seconds = 2
                    raise error

        async def advance(delay):
            delays.append(delay)
            now[0] += delay

        client = AdvancingClient()
        collector = SitemapCollector(client, clock=lambda: now[0], sleep=advance, wait=no_timer)
        await collector.receive_source("http", [observation()])
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
        await collector.receive_source("http", [observation()])
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
        state.resources.before_completion = checkpoint
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
            async def receive(self, observations):
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
        state.resources.http_observation_sink = Sink()
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
            resources=state.resources,
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
            async def receive(self, observations):
                entered.set()
                await finish.wait()

        task = asyncio.create_task(notify_http_observations(Sink(), [{}]))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("proxy", [False, True])
@pytest.mark.parametrize("invalid_status", [99, 600, 700])
def test_malformed_target_status_does_not_poison_observation_batch(tmp_path, proxy, invalid_status):
    from private_schemas import private_validator

    from contractor_runtime.adapters import AdapterHandles
    from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
    from contractor_runtime.toolsets.http.tools import HTTPToolError

    async def scenario():
        accepted, target_calls = [], []
        validator = private_validator("project-sitemap.schema.json#/$defs/SitemapCollectionRequest")

        class StrictTransport(SitemapTransport):
            async def request(self, method, path, *, headers, body, max_response_bytes):
                payload = json.loads(body)
                validator.validate(payload)
                accepted.extend(payload["input"]["observations"])
                return await super().request(
                    method, path, headers=headers, body=body, max_response_bytes=max_response_bytes
                )

        async def handler(request):
            target_calls.append(request.url.path)
            status = invalid_status if request.url.path == "/invalid" else 200
            return httpx.Response(status, request=request)

        collector = SitemapCollector(
            SitemapClient(ArtifactClient("allocation", StrictTransport())), wait=no_timer
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        state = WorkerState()
        state.resources.http_observation_sink = _SourceObservationSink(collector, "http")
        tools = await HTTPToolsetFactory(lambda: client).create_selected(
            selected=["http_request"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example", requestTimeoutSeconds=10
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            resources=state.resources,
            adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(client))
            if proxy
            else AdapterHandles(),
        )
        try:
            await tools["http_request"]("https://target.example/before", method="POST")
            with pytest.raises(HTTPToolError, match="http_response_invalid"):
                await tools["http_request"]("https://target.example/invalid", method="POST")
            await tools["http_request"]("https://target.example/after", method="POST")
            await collector.checkpoint()
            assert [(row["outcome"], row["status"]) for row in accepted] == [
                ("response", 200),
                ("transport-error", 0),
                ("response", 200),
            ]
            assert not collector._disabled
            await tools["http_request"]("https://target.example/later", method="POST")
            await collector.checkpoint()
            assert len(accepted) == 4 and accepted[-1]["status"] == 200
            assert target_calls == ["/before", "/invalid", "/after", "/later"]
        finally:
            await tools["http_request"].close()
            await collector.close()
            await client.aclose()

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
            async def receive(self, rows):
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
        state.resources.http_observation_sink = Sink()
        tools = await factory.create_selected(
            selected=["http_request"],
            allocation_id="allocation",
            run_id="run",
            namespace="research",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example", requestTimeoutSeconds=10
            ),
            workspace=AllocationWorkspace(tmp_path, tmp_path / "allocation"),
            resources=state.resources,
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
