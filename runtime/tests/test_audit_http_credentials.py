"""Model-facing HTTP tool contract over the real controlled httpcore path."""

from __future__ import annotations

import asyncio
import base64
import inspect

import pytest
from fakes.worker_state import WorkerState
from test_engagement_identity import owner_for, wire_response

from contractor_runtime.toolsets.http.errors import HTTPToolError
from contractor_runtime.toolsets.http.tools import HTTPRequestTool, HTTPToolsetFactory


async def tools_for(owner):
    state = WorkerState()
    tools = await HTTPToolsetFactory().create_selected(
        selected=sorted(HTTPToolsetFactory.exported_tools),
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="worker",
        runtime_settings=owner._settings,
        workspace=None,
        resources=state.resources,
        controlled_http=owner,
    )
    return (tools, state)


def test_credential_id_is_a_separate_optional_tool_parameter():
    field = inspect.signature(HTTPRequestTool.__call__).parameters["credential_id"]
    assert field.default is None


def test_explicit_catalog_read_selection_and_body_readers():

    async def scenario():
        owner, _, network = owner_for()
        tools, state = await tools_for(owner)
        catalog = await tools["http_credentials"]()
        assert "private-cookie" not in str(catalog)
        assert not network.streams and (not network.streams)
        raw = await tools["http_credential_read"]("credential-account-a")
        assert raw["values"] == {"session": "private-cookie"}
        first = await tools["http_request"](
            "http://target.example/arbitrary", credential_id="credential-account-a"
        )
        second = await tools["http_request"]("http://target.example/another")
        assert first["status"] == second["status"] == 200
        assert b"Cookie: session=private-cookie" in network.streams[0].written
        assert b"private-cookie" not in network.streams[1].written
        body = await tools["http_read_body"](request_id=first["request_id"])
        assert base64.b64decode(body["data_b64"]) == b"ok"
        assert "private-cookie" not in str(state.metrics)
        assert len(network.streams) == 2
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_foreign_credential_fails_without_action_and_without_retry():

    async def scenario():
        owner, _, network = owner_for()
        tools, _state = await tools_for(owner)
        for invoke in (
            lambda: tools["http_credential_read"]("foreign"),
            lambda: tools["http_request"]("http://target.example/", credential_id="foreign"),
        ):
            with pytest.raises(HTTPToolError) as error:
                await invoke()
            assert error.value.code == "http_credential_unavailable" and (not error.value.retryable)
        assert not network.streams and (not network.streams)
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_failed_write_is_nonretryable_and_never_sent_twice():

    async def scenario():
        owner, _, network = owner_for(fail_write=True)
        owner._assignment.routes[0].methods.append("POST")
        tools, _state = await tools_for(owner)
        with pytest.raises(HTTPToolError) as error:
            await tools["http_request"](
                "http://target.example/write", method="POST", body_type="text", body="model body"
            )
        assert error.value.code == "http_request_failed" and (not error.value.retryable)
        assert len(network.streams) == len(network.streams) == 1
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_delete_cookie_update_and_clear_cannot_reapply_credential():

    async def scenario():
        owner, _, network = owner_for(
            responses=[wire_response(headers=[("Set-Cookie", "learned=x; Path=/")])]
        )
        tools, _state = await tools_for(owner)
        await tools["http_request"]("http://target.example/", credential_id="credential-account-a")
        await tools["http_session_clear"]()
        with pytest.raises(HTTPToolError):
            await tools["http_request"](
                "http://target.example/", credential_id="credential-account-a"
            )
        await tools["http_request"]("http://target.example/")
        assert len(network.streams) == 2 and b"Cookie:" not in network.streams[-1].written
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_explicit_methods_and_bodies_reach_the_wire(method):

    async def scenario():
        owner, _, network = owner_for()
        route = owner._assignment.routes[0]
        route.methods = [method]
        tools, _state = await tools_for(owner)
        payload = "model body" if method in {"POST", "PUT", "PATCH"} else None
        result = await tools["http_request"](
            "http://target.example/chosen?private=secret",
            method=method,
            body=payload,
            body_type="text" if payload else "none",
        )
        assert result["status"] == 200 and result["method"] == method
        wire = bytes(network.streams[0].written)
        assert wire.startswith(method.encode() + b" /chosen?private=secret HTTP/1.1")
        if payload:
            assert wire.endswith(payload.encode())
        other, _, _ = owner_for()
        await other.close()
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_cross_origin_redirect_drops_manual_auth_and_reports_final_method():
    import time

    from fakes.engagement import preflight_spec
    from fakes.engagement_engine import transport as HTTPTransport
    from test_engagement_target import Network, policy_config

    async def scenario():
        spec = preflight_spec()
        assignment = spec.engagement.http
        route = assignment.routes[0]
        route.methods = ["GET", "POST"]
        assignment.routes.append(route.model_copy(update={"origin": "http://other.example"}))
        assignment.identities[0].origins.append("http://other.example")
        network = Network(
            [],
            [
                wire_response(
                    303,
                    [
                        ("Location", "http://other.example/final"),
                        ("Set-Cookie", "learned=private; Path=/"),
                    ],
                ),
                wire_response(),
            ],
        )
        owner = HTTPTransport(
            spec,
            resolver=policy_config(extra={"other.example": ("10.20.0.8",)}).resolver,
            backend=network,
        )
        owner.configure(authority_deadline=lambda: time.monotonic() + 60)
        tools, _state = await tools_for(owner)
        result = await tools["http_request"](
            "http://target.example/start",
            method="POST",
            body_type="text",
            body="write",
            headers={"Authorization": "Bearer manual-secret", "Cookie": "manual=secret"},
        )
        assert result["method"] == "GET" and result["redirects"] == 1 and (result["retries"] == 0)
        assert [bytes(p.written).split(b" ", 1)[0].decode() for p in network.streams] == [
            "POST",
            "GET",
        ]
        assert b"manual-secret" in network.streams[0].written
        second = bytes(network.streams[1].written)
        assert second.startswith(b"GET /final HTTP/1.1")
        assert (
            b"Authorization:" not in second
            and b"Cookie:" not in second
            and (not second.endswith(b"write"))
        )
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_stop_joins_tool_processing_and_erases_response_buffer(monkeypatch):
    import contractor_runtime.toolsets.http.controlled as controlled

    async def scenario():
        owner, _, _ = owner_for()
        tools, _state = await tools_for(owner)
        await tools["http_request"]("http://target.example/retained")
        assert tools["http_request"]._session._captures.contains("http", "1")
        entered = asyncio.Event()
        finished = asyncio.Event()
        preview = controlled._encode_body

        def blocked_preview(*args):
            return preview(*args)

        original = controlled.to_thread_until_done

        async def held(function, *args, **kwargs):
            if function is blocked_preview:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()
            return await original(function, *args, **kwargs)

        monkeypatch.setattr(controlled, "_encode_body", blocked_preview)
        monkeypatch.setattr(controlled, "to_thread_until_done", held)
        task = asyncio.create_task(tools["http_request"]("http://target.example/in-flight"))
        await entered.wait()
        await owner.close()
        assert task.done() and task.cancelled() and finished.is_set()
        assert tools["http_request"]._session._closed
        assert not tools["http_request"]._session._captures.contains("http", "1")
        assert not tools["http_request"]._session._history
        with pytest.raises(HTTPToolError):
            await tools["http_read_body"](request_id=1)

    asyncio.run(scenario())


def test_controlled_headers_merge_without_case_duplicates_or_losing_explicit_content_type():
    async def scenario():
        owner, _, network = owner_for()
        tools, _ = await tools_for(owner)
        owner._assignment.routes[0].methods.append("POST")
        await tools["http_session_set"](headers={"X-Mode": "old"})
        await tools["http_session_set"](headers={"x-mode": "new"})
        assert (await tools["http_session_get"]())["default_headers"] == {"x-mode": "new"}
        await tools["http_request"](
            "http://target.example/",
            method="POST",
            body_type="json",
            body={"ok": True},
            headers={"X-MODE": "request", "content-type": "application/custom"},
        )
        wire = bytes(network.streams[0].written).lower()
        assert wire.count(b"x-mode:") == 1 and b"x-mode: request" in wire
        assert wire.count(b"content-type:") == 1 and b"content-type: application/custom" in wire
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("bound", ["count", "bytes"])
def test_controlled_header_updates_and_requests_enforce_aggregate_bounds(bound):
    async def scenario():
        owner, _, network = owner_for()
        tools, _ = await tools_for(owner)
        width, value = (40, "x") if bound == "count" else (4, "x" * 7500)
        first = {f"X-First-{i}": value for i in range(width)}
        second = {f"X-Second-{i}": value for i in range(width)}
        await tools["http_session_set"](headers=first)
        with pytest.raises(HTTPToolError) as update:
            await tools["http_session_set"](headers=second)
        assert update.value.code == "http_request_invalid"
        assert (await tools["http_session_get"]())["default_headers"] == first
        with pytest.raises(HTTPToolError) as request:
            await tools["http_request"]("http://target.example/", headers=second)
        assert request.value.code == "http_request_invalid"
        assert not network.streams
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("content_type", ["text/plain", "application/octet-stream"])
def test_controlled_truncated_prefix_remains_explicit_in_reads_history_and_private_capture(
    content_type,
):
    from test_engagement_target import Network, owner

    from contractor_runtime.contracts.engagement import HTTPLimits

    async def scenario():
        network = Network(
            [],
            [
                (
                    f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
                    "Content-Length: 8\r\n\r\nsecretxx"
                ).encode()
            ],
        )
        transport = owner(network, limits=HTTPLimits(responseBodyBytes=3))
        tools, state = await tools_for(transport)
        result = await tools["http_request"]("http://target.example/start")
        assert result["body_truncated"]
        for _ in range(2):
            page = await tools["http_read_body"](result["request_id"])
            assert page["body_truncated"] and page["total"] == 3 and page["eof"]
        assert (await tools["http_history"]())[0]["body_truncated"]
        assert not state.resources.http_captures.contains("http", str(result["request_id"]))
        capture = await tools["http_request"]._session._captures.get(
            "http", str(result["request_id"])
        )
        assert not capture["attempts"][-1]["response"]["complete"]
        assert capture["attempts"][-1]["error"] == "response_too_large"
        await tools["http_request"]._session.close()

    asyncio.run(scenario())
