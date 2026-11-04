"""Model-facing HTTP tool contract over the real controlled httpcore path."""

from __future__ import annotations

import asyncio
import base64
import inspect

import pytest
from fakes.engagement_engine import reply
from fakes.worker_state import WorkerState
from test_engagement_identity import owner_for

from contractor_runtime.adapters.bridge import BridgeFailure
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
        owner, _, results = owner_for()
        tools, state = await tools_for(owner)
        catalog = await tools["http_credentials"]()
        assert "private-cookie" not in str(catalog)
        assert not results.attempts
        raw = await tools["http_credential_read"]("credential-account-a")
        assert raw["values"] == {"session": "private-cookie"}
        first = await tools["http_request"](
            "http://target.example/arbitrary", credential_id="credential-account-a"
        )
        second = await tools["http_request"]("http://target.example/another")
        assert first["status"] == second["status"] == 200
        assert (b"Cookie", b"session=private-cookie") in results.attempts[0].request.headers
        assert not any(b"private-cookie" in v for _, v in results.attempts[1].request.headers)
        body = await tools["http_read_body"](request_id=first["request_id"])
        assert base64.b64decode(body["data_b64"]) == b"ok"
        assert "private-cookie" not in str(state.metrics)
        assert len(results.attempts) == 2
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_foreign_credential_fails_without_action_and_without_retry():

    async def scenario():
        owner, _, results = owner_for()
        tools, _state = await tools_for(owner)
        for invoke in (
            lambda: tools["http_credential_read"]("foreign"),
            lambda: tools["http_request"]("http://target.example/", credential_id="foreign"),
        ):
            with pytest.raises(HTTPToolError) as error:
                await invoke()
            assert error.value.code == "http_credential_unavailable" and (not error.value.retryable)
        assert not results.attempts
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_failed_write_is_nonretryable_and_never_sent_twice():

    async def scenario():
        owner, _, results = owner_for(responses=[BridgeFailure("transport_failed")])
        owner._assignment.routes[0].methods.append("POST")
        tools, _state = await tools_for(owner)
        with pytest.raises(HTTPToolError) as error:
            await tools["http_request"](
                "http://target.example/write", method="POST", body_type="text", body="model body"
            )
        assert error.value.code == "http_request_failed" and (not error.value.retryable)
        assert len(results.attempts) == 1
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_delete_cookie_update_and_clear_cannot_reapply_credential():

    async def scenario():
        owner, _, results = owner_for(
            responses=[reply(headers=[("Set-Cookie", "learned=x; Path=/")])]
        )
        tools, _state = await tools_for(owner)
        await tools["http_request"]("http://target.example/", credential_id="credential-account-a")
        await tools["http_session_clear"]()
        with pytest.raises(HTTPToolError):
            await tools["http_request"](
                "http://target.example/", credential_id="credential-account-a"
            )
        await tools["http_request"]("http://target.example/")
        assert len(results.attempts) == 2 and not any(
            k.lower() == b"cookie" for k, _ in results.attempts[-1].request.headers
        )
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_explicit_methods_and_bodies_reach_the_engine(method):

    async def scenario():
        owner, _, results = owner_for()
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
        attempt = results.attempts[0]
        assert attempt.request.method == method
        assert attempt.url == "http://target.example/chosen?private=secret"
        assert attempt.request.body == (payload.encode() if payload else b"")
        other, _, _ = owner_for()
        await other.close()
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


def test_cross_origin_redirect_drops_manual_auth_and_reports_final_method():
    import time

    from fakes.engagement import preflight_spec
    from fakes.engagement_engine import transport as HTTPTransport
    from test_engagement_target import Results, policy_config

    async def scenario():
        spec = preflight_spec()
        assignment = spec.engagement.http
        route = assignment.routes[0]
        route.methods = ["GET", "POST"]
        assignment.routes.append(route.model_copy(update={"origin": "http://other.example"}))
        assignment.identities[0].origins.append("http://other.example")
        results = Results(
            [
                reply(
                    303,
                    [
                        ("Location", "http://other.example/final"),
                        ("Set-Cookie", "learned=private; Path=/"),
                    ],
                ),
                reply(),
            ],
        )
        owner = HTTPTransport(
            spec,
            resolver=policy_config(extra={"other.example": ("10.20.0.8",)}).resolver,
            engine=results,
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
        assert [p.request.method for p in results.attempts] == ["POST", "GET"]
        assert (b"Authorization", b"Bearer manual-secret") in results.attempts[0].request.headers
        second = results.attempts[1]
        assert second.url == "http://other.example/final" and second.request.body == b""
        assert not any(
            k.lower() in {b"authorization", b"cookie"} for k, _ in second.request.headers
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
        owner, _, results = owner_for()
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
        headers = [(k.lower(), v) for k, v in results.attempts[0].request.headers]
        assert [v for k, v in headers if k == b"x-mode"] == [b"request"]
        assert [v for k, v in headers if k == b"content-type"] == [b"application/custom"]
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("bound", ["count", "bytes"])
def test_controlled_header_updates_and_requests_enforce_aggregate_bounds(bound):
    async def scenario():
        owner, _, results = owner_for()
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
        assert not results.attempts
        await tools["http_request"]._session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("content_type", ["text/plain", "application/octet-stream"])
def test_controlled_truncated_prefix_remains_explicit_in_reads_history_and_private_capture(
    content_type,
):
    from test_engagement_target import Results, owner

    from contractor_runtime.contracts.engagement import HTTPLimits

    async def scenario():
        results = Results(
            [reply(headers=[("Content-Type", content_type)], body=b"sec", truncated=True)]
        )
        transport = owner(results, limits=HTTPLimits(responseBodyBytes=3))
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
