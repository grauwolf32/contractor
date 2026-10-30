"""Model-facing HTTP tool contract over the real controlled httpcore path."""

from __future__ import annotations

import asyncio
import base64
import inspect

import pytest
from test_pentest_identity import owner_for, wire_response

from contractor_runtime.allocation import WorkerState
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
        state=state,
        controlled_http=owner,
    )
    return tools, state


def test_credential_id_is_a_separate_optional_tool_parameter():
    field = inspect.signature(HTTPRequestTool.__call__).parameters["credential_id"]
    assert field.default is None


def test_explicit_catalog_read_selection_and_body_readers():
    async def scenario():
        owner, journal, network = owner_for()
        tools, state = await tools_for(owner)
        catalog = await tools["http_credentials"]()
        assert "private-cookie" not in str(catalog)
        assert not journal.prepares and not network.streams
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
        assert len(journal.prepares) == 2
        await state.http_session.close()

    asyncio.run(scenario())


def test_foreign_credential_fails_without_action_and_without_retry():
    async def scenario():
        owner, journal, network = owner_for()
        tools, state = await tools_for(owner)
        for invoke in (
            lambda: tools["http_credential_read"]("foreign"),
            lambda: tools["http_request"]("http://target.example/", credential_id="foreign"),
        ):
            with pytest.raises(HTTPToolError) as error:
                await invoke()
            assert error.value.code == "http_credential_unavailable" and not error.value.retryable
        assert not journal.prepares and not network.streams
        await state.http_session.close()

    asyncio.run(scenario())


def test_unknown_write_is_nonretryable_and_never_sent_twice():
    async def scenario():
        owner, journal, network = owner_for(fail_write=True)
        owner._assignment.routes[0].methods.append("POST")
        owner._assignment.routes[0].action_classes.append("probe")
        tools, state = await tools_for(owner)
        with pytest.raises(HTTPToolError) as error:
            await tools["http_request"](
                "http://target.example/write", method="POST", body_type="text", body="model body"
            )
        assert error.value.code == "http_request_unknown_outcome" and not error.value.retryable
        assert len(journal.prepares) == len(network.streams) == 1
        assert journal.prepares[0].requestBodyBytes == 10
        assert journal.outcomes[0].kind == "unknown_outcome"
        await state.http_session.close()

    asyncio.run(scenario())


def test_delete_cookie_update_and_clear_cannot_reapply_credential():
    async def scenario():
        owner, journal, network = owner_for(
            responses=[wire_response(headers=[("Set-Cookie", "learned=x; Path=/")])]
        )
        tools, state = await tools_for(owner)
        await tools["http_request"]("http://target.example/", credential_id="credential-account-a")
        await tools["http_session_clear"]()
        with pytest.raises(HTTPToolError):
            await tools["http_request"](
                "http://target.example/", credential_id="credential-account-a"
            )
        await tools["http_request"]("http://target.example/")
        assert len(journal.prepares) == 2 and b"Cookie:" not in network.streams[-1].written
        await state.http_session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_explicit_methods_bodies_and_keyed_request_identity(method):
    from contractor_runtime.contracts.pentest_action import action_class_for_method

    async def scenario():
        owner, journal, network = owner_for()
        route = owner._assignment.routes[0]
        route.methods = [method]
        route.action_classes = [action_class_for_method(method)]
        tools, state = await tools_for(owner)
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
        prepared = journal.prepares[0]
        assert prepared.method == method and prepared.requestBodyBytes == (10 if payload else 0)
        other, _, _ = owner_for()
        assert prepared.targetDigest == owner._target_digest(prepared.url)
        assert prepared.targetDigest != other._target_digest(prepared.url)
        assert "private=secret" not in prepared.requestFingerprint
        await other.close()
        await state.http_session.close()

    asyncio.run(scenario())


def test_cross_origin_redirect_drops_manual_auth_and_reports_final_method():
    import time

    from fakes.pentest import preflight_spec
    from test_pentest_target import Journal, Network, policy_config

    from contractor_runtime.adapters.pentest_target import PentestHTTPTransport

    async def scenario():
        spec = preflight_spec()
        assignment = spec.pentestEngagement.http
        route = assignment.routes[0]
        route.methods = ["GET", "POST"]
        route.action_classes = ["read", "probe"]
        assignment.routes.append(route.model_copy(update={"origin": "http://other.example"}))
        assignment.identities[0].origins.append("http://other.example")
        journal = Journal([])
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
        owner = PentestHTTPTransport(
            spec,
            journal,
            target_policy=policy_config(extra={"other.example": ("10.20.0.8",)}),
            backend=network,
        )
        owner.configure(authority_deadline=lambda: time.monotonic() + 60)
        tools, state = await tools_for(owner)
        result = await tools["http_request"](
            "http://target.example/start",
            method="POST",
            body_type="text",
            body="write",
            headers={"Authorization": "Bearer manual-secret", "Cookie": "manual=secret"},
        )
        assert result["method"] == "GET" and result["redirects"] == 1 and result["retries"] == 0
        assert [p.method for p in journal.prepares] == ["POST", "GET"]
        assert b"manual-secret" in network.streams[0].written
        second = bytes(network.streams[1].written)
        assert second.startswith(b"GET /final HTTP/1.1")
        assert (
            b"Authorization:" not in second
            and b"Cookie:" not in second
            and not second.endswith(b"write")
        )
        await state.http_session.close()

    asyncio.run(scenario())


def test_stop_joins_tool_processing_and_erases_response_buffer(monkeypatch):
    import contractor_runtime.toolsets.http.controlled as controlled

    async def scenario():
        owner, _, _ = owner_for()
        tools, state = await tools_for(owner)
        await tools["http_request"]("http://target.example/retained")
        assert state.http_session._captures.contains("http", "1")
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
        assert state.http_session._closed
        assert not state.http_session._captures.contains("http", "1")
        assert not state.http_session._history
        with pytest.raises(HTTPToolError):
            await tools["http_read_body"](request_id=1)

    asyncio.run(scenario())
