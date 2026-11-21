from __future__ import annotations

import asyncio
import base64
import json
import threading
from copy import deepcopy

import httpx
import pytest
from test_caido_read_tools import (
    CAIDO_TOKEN,
    FakeArtifactClient,
    close_tools,
    create_tools,
    local_request,
    representative_responses,
    request_summary,
)

import contractor_runtime.toolsets.caido.tools as caido_tools


class RecordingSink:
    def __init__(self):
        self.rows = []
        self.gaps = []

    async def receive(self, observations, *, gaps=()):
        self.rows.extend(deepcopy(observations))
        self.gaps.extend(deepcopy(gaps))


def test_selected_reads_emit_only_metadata_before_output_redaction(tmp_path):
    async def scenario():
        raw_request = b"GET / HTTP/1.1\r\nHost: target.example\r\nX-Secret: raw-only\r\n\r\n"
        responses = representative_responses(raw_request, b"HTTP/1.1 200 OK\r\n\r\nbody-only")
        selected = request_summary()
        selected.update(path=f"/session/{CAIDO_TOKEN}", query="password=opaque&token=value")
        missing_response = request_summary()
        missing_response.update(id="selected-no-response", response=None, createdAt=1788220800123)
        malformed_metadata = request_summary()
        malformed_metadata.update(id="selected-invalid", host="", createdAt="invalid-provider-time")
        malformed_metadata["response"]["statusCode"] = 700
        responses["RequestsByOffset"]["requestsByOffset"] = {
            "count": {"value": 500},
            "nodes": [selected, malformed_metadata, missing_response],
        }
        responses["RequestDetail"]["request"].update(
            path=selected["path"],
            query=selected["query"],
            createdAt="2026-09-01T03:00:00.000+03:00",
        )
        calls = []

        async def handler(request):
            payload = json.loads(request.content)
            calls.append(payload)
            return httpx.Response(200, json={"data": responses[payload["operationName"]]})

        sink = RecordingSink()
        artifacts = FakeArtifactClient()
        tools, _state, handle = await create_tools(
            tmp_path, handler, artifacts, observation_sink=sink
        )
        try:
            history = await tools["caido_history"](filter='req.host.eq:"target.example"', limit=3)
            assert history["count"] == 500 and len(history["requests"]) == 3
            assert CAIDO_TOKEN not in history["requests"][0]["path"]
            assert len(sink.rows) == 2
            assert sink.rows[0] == {
                "url": f"https://target.example/session/{CAIDO_TOKEN}?password=opaque&token=value",
                "method": "GET",
                "outcome": "response",
                "status": 200,
                "at": "2026-09-01T00:00:00Z",
            }
            assert sink.rows[1]["outcome"] == "attempt" and sink.rows[1]["status"] == 0
            assert sink.rows[1]["at"] == "2026-09-01T00:00:00.123Z"
            # Invalid facts become a bounded omission locally, not an invented
            # attempt or a malformed Server row. The following valid row survives.
            assert sink.gaps == [{"reason": "projection", "count": 1}]
            detail = await tools["caido_request_detail"](local_request(tools, "request-1"))
            assert CAIDO_TOKEN not in detail["path"]
            assert detail["created_at"] == "2026-09-01T03:00:00.000+03:00"
            assert sink.rows[-1] == sink.rows[0]
            assert len(sink.rows) == 3
            await tools["caido_sitemap"](scope_id="scope-1")
            await tools["caido_automate_results"]("session-1")
            await tools["caido_workflow_findings"]()
            assert len(sink.rows) == 3
            assert [call["operationName"] for call in calls] == [
                "RequestsByOffset",
                "RequestDetail",
                "SitemapRoot",
                "AutomateSession",
                "AutomateEntryRequests",
                "FindingsByOffset",
            ]
            for row in sink.rows:
                assert set(row) == {"url", "method", "outcome", "status", "at"}
            assert "raw-only" not in json.dumps(sink.rows)
            assert "body-only" not in json.dumps(sink.rows)
            assert artifacts.payloads == {}
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_all_invalid_selected_metadata_still_notifies_bounded_gaps(tmp_path):
    async def scenario():
        bad_status = request_summary()
        bad_status["response"]["statusCode"] = 700
        bad_method = request_summary()
        bad_method["method"] = "get"
        bad_host = request_summary()
        bad_host["host"] = "invalid@target.example"
        calls = []

        async def handler(request):
            calls.append(json.loads(request.content)["operationName"])
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {
                            "count": {"value": 3},
                            "nodes": [bad_status, bad_method, bad_host],
                        }
                    }
                },
            )

        sink = RecordingSink()
        tools, _, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=sink,
        )
        try:
            result = await tools["caido_history"]()
            assert result["requests"][0]["status_code"] == 700
            assert result["requests"][1]["method"] == "get"
            assert result["requests"][2]["host"] == "invalid@target.example"
            assert sink.rows == [] and sink.gaps == [{"reason": "projection", "count": 3}]
            assert calls == ["RequestsByOffset"]
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_optional_preprocessing_failure_preserves_selected_tool_result(tmp_path, monkeypatch):
    def broken(_observations):
        raise RuntimeError("optional metadata preprocessing failed")

    monkeypatch.setattr(caido_tools, "normalize_http_observations", broken)

    async def scenario():
        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {"count": {"value": 1}, "nodes": [request_summary()]}
                    }
                },
            )

        sink = RecordingSink()
        tools, _, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=sink,
        )
        try:
            result = await tools["caido_history"]()
            assert result["requests"][0]["status_code"] == 200
            assert sink.rows == [] and sink.gaps == []
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_preprocessing_cancellation_joins_thread_and_subscriber_before_cleanup(
    tmp_path, monkeypatch
):
    async def scenario():
        entered, joined = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        normalize = caido_tools.normalize_http_observations

        def held(observations):
            loop.call_soon_threadsafe(entered.set)
            release.wait()
            result = normalize(observations)
            loop.call_soon_threadsafe(joined.set)
            return result

        monkeypatch.setattr(caido_tools, "normalize_http_observations", held)

        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {"count": {"value": 1}, "nodes": [request_summary()]}
                    }
                },
            )

        sink = RecordingSink()
        tools, _, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=sink,
        )
        try:
            reading = asyncio.create_task(tools["caido_history"]())
            await entered.wait()
            reading.cancel()
            closing = asyncio.create_task(close_tools(tools))
            await asyncio.sleep(0)
            assert not reading.done() and not closing.done() and not joined.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await reading
            await closing
            assert joined.is_set() and len(sink.rows) == 1
        finally:
            release.set()
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_completed_replay_uses_provider_time_and_actual_connection_without_extra_calls(tmp_path):
    async def scenario():
        calls = []
        replay_raw = ""

        async def handler(request):
            nonlocal replay_raw
            payload = json.loads(request.content)
            calls.append(payload)
            operation = payload["operationName"]
            if operation == "CreateReplaySession":
                data = {
                    "createReplaySession": {
                        "session": {
                            "id": "session",
                            "name": "replay",
                            "activeEntry": {"id": "entry"},
                        }
                    }
                }
            elif operation == "StartReplayTask":
                replay_raw = payload["variables"]["input"]["raw"]
                data = {
                    "startReplayTask": {
                        "error": None,
                        "task": {
                            "id": "task",
                            "replayEntry": {"id": "entry"},
                        },
                    }
                }
            else:
                assert operation == "ReplayEntry"
                assert "createdAt" in payload["query"]
                data = {
                    "replayEntry": {
                        "id": "entry",
                        "raw": replay_raw,
                        "error": None,
                        "request": {
                            "id": "replayed",
                            "method": "POST",
                            "host": "request-host.example",
                            "path": "/session/opaque",
                            "query": "token=value",
                            "createdAt": 1788220800123,
                            "response": {
                                "statusCode": 201,
                                "length": 2,
                                "roundtripTime": 7,
                                "raw": base64.b64encode(b"HTTP/1.1 201 Created\r\n\r\nok").decode(),
                            },
                        },
                    }
                }
            return httpx.Response(200, json={"data": data})

        sink = RecordingSink()
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
            observation_sink=sink,
        )
        try:
            arguments = {
                "raw_request": (
                    "POST /session/opaque?token=value HTTP/1.1\r\nHost: target.example\r\n\r\n"
                ),
                "host": "target.example",
                "port": 8443,
                "is_tls": True,
            }
            result = await tools["caido_replay"](**arguments, wait=True)
            assert result["status"] == "completed"
            assert "created_at" not in result and "_created_at" not in result
            assert sink.rows == [
                {
                    "url": "https://target.example:8443/session/opaque?token=value",
                    "method": "POST",
                    "outcome": "response",
                    "status": 201,
                    "at": "2026-09-01T00:00:00.123Z",
                }
            ]
            assert [call["operationName"] for call in calls] == [
                "CreateReplaySession",
                "StartReplayTask",
                "ReplayEntry",
            ]
            started = await tools["caido_replay"](**arguments, wait=False)
            assert started["status"] == "started" and len(sink.rows) == 1
            assert len(calls) == 5
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_selected_read_ignores_subscriber_failure(tmp_path, failure):
    async def scenario():
        class FailingSink:
            async def receive(self, _observations):
                raise failure("subscriber failure")

        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {
                            "count": {"value": 1},
                            "nodes": [request_summary()],
                        }
                    }
                },
            )

        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=FailingSink(),
        )
        try:
            result = await tools["caido_history"]()
            assert result["requests"][0]["status_code"] == 200
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_read_cancellation_joins_subscriber_before_session_cleanup(tmp_path):
    async def scenario():
        entered, release, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class BlockingSink:
            async def receive(self, _observations):
                entered.set()
                await release.wait()
                joined.set()

        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {
                            "count": {"value": 1},
                            "nodes": [request_summary()],
                        }
                    }
                },
            )

        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=BlockingSink(),
        )
        tool = tools["caido_history"]
        read = asyncio.create_task(tool())
        try:
            await entered.wait()
            read.cancel()
            close = asyncio.create_task(tool.close())
            await asyncio.sleep(0)
            assert not read.done() and not close.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await read
            await close
            assert joined.is_set()
            assert tool._session._observation_sink is None
        finally:
            release.set()
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_history_emits_one_selected_bounded_page_for_shared_buffer_admission(tmp_path):
    async def scenario():
        nodes = []
        for index in range(100):
            row = request_summary()
            row.update(id=f"selected-{index}", path=f"/{index}/" + "x" * 7000)
            nodes.append(row)
        calls = []

        async def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "data": {
                        "requestsByOffset": {
                            "count": {"value": 10000},
                            "nodes": nodes,
                        }
                    }
                },
            )

        sink = RecordingSink()
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_history"},
            observation_sink=sink,
        )
        try:
            result = await tools["caido_history"](limit=100, offset=50)
            assert result["count"] == 10000 and len(result["requests"]) == 100
            assert len(sink.rows) == 100 and len(json.dumps(sink.rows)) > 256 << 10
            assert len(calls) == 1 and calls[0]["variables"]["offset"] == 50
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())
