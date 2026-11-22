"""Controlled HTTP outcomes across native sockets, tool adaptation and ADK."""

import asyncio
import json
import ssl
import time

import pytest
from fakes.engagement_socket import SocketTarget, preflight, resolver_for, socket_spec, tls_contexts
from fakes.model import scripted_model, text_result, tool_call
from test_adk_runtime import create_runtime, stage_request
from test_audit_http_credentials import tools_for

from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.target_policy import TargetUnresolved
from contractor_runtime.toolsets.http.errors import HTTPToolError
from contractor_runtime.worker.instrumentation import _safe_tool_response


class Clock:
    def __init__(self):
        self.now = 100.0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay


def test_native_rps_allows_more_than_forty_requests_and_denial_exposes_scope():
    async def scenario():
        async with SocketTarget() as target:
            clock = Clock()
            async with preflight(
                target,
                limits=HTTPLimits(requestsPerSecond=2),
                monotonic=clock,
                sleep=clock.sleep,
            ) as (owner, _):
                owner.configure(authority_deadline=lambda: clock() + 60)
                owner._assignment.routes[0].path_prefixes = ["/allowed"]
                tools, _state = await tools_for(owner)
                description = tools["http_request"].__doc__
                assert target.origin in description and '"allowed_origins"' in description
                assert '"methods"' in description and '"exclusions"' in description
                assert '"protected_destinations"' not in description
                for _ in range(45):
                    result = await tools["http_request"](target.origin + "/allowed")
                    assert result["status"] == 200
                assert len(target.requests) == 45
                assert clock.waits == [0.5] * 44
                for url in ["http://other.example/", target.origin + "/forbidden"]:
                    with pytest.raises(HTTPToolError) as raised:
                        await tools["http_request"](url)
                    response = _safe_tool_response("http_request", raised.value)["error"]
                    assert response["code"] == "http_target_denied"
                    assert response["scope"]["allowed_origins"] == [target.origin]
                    assert len(target.requests) == 45

    asyncio.run(scenario())


def test_native_redirect_dns_failure_retains_prior_post_send_state():
    async def scenario():
        async def respond(_target, _reader, writer, _message):
            writer.write(
                b"HTTP/1.1 303 See Other\r\nLocation: http://unresolved.example/final\r\n"
                b"Content-Length: 0\r\n\r\n"
            )
            await writer.drain()

        async with SocketTarget(respond) as target:
            spec = socket_spec(target)
            assignment = spec.engagement.http
            assignment.routes.append(
                assignment.routes[0].model_copy(update={"origin": "http://unresolved.example"})
            )
            assignment.identities[0].origins.append("http://unresolved.example")
            config = resolver_for(target)
            original = config.resolver

            async def resolve(host, port):
                if host == "unresolved.example":
                    raise TargetUnresolved
                return await original(host, port)

            async with preflight(target, spec=spec, config=type(config)(resolver=resolve)) as (
                owner,
                _,
            ):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                tools, _state = await tools_for(owner)
                with pytest.raises(HTTPToolError) as raised:
                    await tools["http_request"](
                        target.origin + "/write", method="POST", body_type="text", body="change"
                    )
                error = _safe_tool_response("http_request", raised.value)["error"]
                assert error["code"] == "http_dns_failed"
                assert error["request_may_have_been_sent"] is True
                assert "Do not repeat" in error["message"]
                assert len(target.requests) == 1
                assert target.requests[0].startswith(b"POST /write ")

    asyncio.run(scenario())


@pytest.mark.parametrize("queue", ["expires", "consumes_budget"])
def test_controlled_session_queue_uses_original_request_deadline(queue):
    async def scenario():
        release = asyncio.Event()
        second_received = asyncio.Event()

        async def respond(_target, _reader, writer, message):
            if message.startswith(b"GET /first "):
                await release.wait()
            else:
                second_received.set()
                await asyncio.sleep(0.6)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

        async with SocketTarget(respond) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, _state = await tools_for(owner)
            first = asyncio.create_task(tools["http_request"](target.origin + "/first", timeout=5))
            async with asyncio.timeout(5):
                await target.request_received.wait()
            second = asyncio.create_task(
                tools["http_request"](target.origin + "/second", timeout=1)
            )
            try:
                if queue == "consumes_budget":
                    await asyncio.sleep(0.6)
                    release.set()
                async with asyncio.timeout(2):
                    with pytest.raises(HTTPToolError) as raised:
                        await second
                error = _safe_tool_response("http_request", raised.value)["error"]
                assert error["code"] == "http_timeout"
                assert error["request_may_have_been_sent"] == (queue == "consumes_budget")
            finally:
                release.set()
                await first
                if not second.done():
                    second.cancel()
                await asyncio.gather(second, return_exceptions=True)
            assert len(target.requests) == (2 if queue == "consumes_budget" else 1)
            assert second_received.is_set() == (queue == "consumes_budget")
            assert not owner._tasks

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["bridge", "lease"])
def test_parallel_tool_cancellation_preserves_native_fatal_worker_failure(
    tmp_path, monkeypatch, failure
):
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, state = await tools_for(owner)
            await owner._initialize()
            original = owner._engine.handle_hook
            pending = asyncio.Event()
            joined = asyncio.Event()

            async def pending_tool() -> dict:
                """Perform a pending local operation."""
                pending.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    joined.set()
                return {}

            async def broken(pool, hook):
                if hook.name == "requestheaders":
                    await pending.wait()
                    if failure == "bridge":
                        raise RuntimeError("private-parallel-bridge-canary")
                    owner.configure(authority_deadline=lambda: 0.0)
                return await original(pool, hook)

            monkeypatch.setattr(owner._engine, "handle_hook", broken)
            response = tool_call(
                "http_request", {"url": target.origin + "/allowed"}, call_id="http-1"
            )
            response.content.parts.extend(
                tool_call("pending_tool", {}, call_id="pending-1").content.parts
            )
            model = scripted_model([response, text_result("Must not be requested")])
            worker = await create_runtime(
                tmp_path,
                state,
                {"http_request": tools["http_request"], "pending_tool": pending_tool},
                model,
            )
            try:
                completion = await worker.invoke(stage_request())
                assert completion.failure is not None
                assert completion.failure.code == (
                    "http_bridge_failed" if failure == "bridge" else "http_allocation_closed"
                )
                assert not completion.failure.retryable
                assert joined.is_set()
                assert not owner._tasks
                assert "private-parallel-bridge-canary" not in completion.model_dump_json()
                refused = await worker.invoke(
                    stage_request().model_copy(update={"subtask_id": "1"})
                )
                assert refused.failure is not None and refused.failure.code == "worker_draining"
                assert len(model.requests) == 1
            finally:
                from datetime import UTC, datetime, timedelta

                await worker.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["dns", "tls", "connection", "timeout"])
def test_native_network_failure_has_concrete_model_reason_and_send_state(tmp_path, failure):
    async def scenario():
        clock = Clock()

        async def respond(_target, _reader, writer, _message):
            if failure == "timeout":
                clock.now += 31
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

        server_tls = tls_contexts(tmp_path)[0] if failure == "tls" else None
        async with SocketTarget(respond, ssl_context=server_tls) as target:
            config = resolver_for(target)
            if failure == "dns":
                original = config.resolver

                async def resolver(host, port):
                    if host == "target.example":
                        raise TargetUnresolved
                    return await original(host, port)

                config = type(config)(resolver=resolver)
            if failure == "connection":
                target.server.close()
                await target.server.wait_closed()
            async with preflight(
                target,
                config=config,
                monotonic=clock,
                ssl_context=ssl.create_default_context() if server_tls else None,
            ) as (owner, _):
                owner.configure(authority_deadline=lambda: clock() + 60)
                tools, state = await tools_for(owner)
                with pytest.raises(HTTPToolError) as raised:
                    await tools["http_request"](target.origin + "/allowed")
                error = _safe_tool_response("http_request", raised.value)["error"]
                expected = {
                    "dns": "http_dns_failed",
                    "tls": "http_tls_failed",
                    "connection": "http_connection_failed",
                    "timeout": "http_timeout",
                }[failure]
                assert error["code"] == expected
                assert error["request_may_have_been_sent"] == (failure == "timeout")
                assert not error["retryable"]
                assert ("TIMED OUT" if failure == "timeout" else failure.upper()) in error[
                    "message"
                ].upper()
                assert len(target.requests) == (1 if failure == "timeout" else 0)
                report = state.metrics.build_report(report_id="network", duration_ms=1)
                assert expected in report.model_dump_json()
                assert target.origin not in report.model_dump_json()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["bridge", "lease"])
def test_native_fatal_failure_stops_actual_worker_before_another_model_call(
    tmp_path,
    monkeypatch,
    failure,
):
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, state = await tools_for(owner)
            if failure == "bridge":
                await owner._initialize()
                original = owner._engine.handle_hook

                async def broken(pool, hook):
                    if hook.name == "requestheaders":
                        raise RuntimeError("private-bridge-exception-canary")
                    return await original(pool, hook)

                monkeypatch.setattr(owner._engine, "handle_hook", broken)
            else:
                owner.configure(authority_deadline=lambda: 0.0)
            model = scripted_model(
                [
                    tool_call("http_request", {"url": target.origin + "/allowed"}, call_id="h1"),
                    text_result("Must not be requested"),
                ]
            )
            worker = await create_runtime(
                tmp_path, state, {"http_request": tools["http_request"]}, model
            )
            try:
                completion = await worker.invoke(stage_request())
                assert completion.failure is not None
                assert completion.failure.code == (
                    "http_bridge_failed" if failure == "bridge" else "http_allocation_closed"
                )
                assert not completion.failure.retryable
                assert len(model.requests) == 1
                assert "private-bridge-exception-canary" not in completion.model_dump_json()
                assert not owner._tasks
            finally:
                from datetime import UTC, datetime, timedelta

                await worker.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_native_cancelled_request_is_joined_without_model_continuation(tmp_path):
    async def scenario():
        release = asyncio.Event()

        async def stall(_target, _reader, _writer, _message):
            await release.wait()

        async with SocketTarget(stall) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, state = await tools_for(owner)
            model = scripted_model(
                [
                    tool_call("http_request", {"url": target.origin + "/allowed"}, call_id="h1"),
                    text_result("Must not be requested"),
                ]
            )
            worker = await create_runtime(
                tmp_path, state, {"http_request": tools["http_request"]}, model
            )
            invocation = asyncio.create_task(worker.invoke(stage_request()))
            try:
                async with asyncio.timeout(10):
                    await target.request_received.wait()
                invocation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await invocation
                assert len(model.requests) == 1
                assert len(target.requests) == 1
                assert not owner._tasks
            finally:
                release.set()
                from datetime import UTC, datetime, timedelta

                await worker.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_actual_adk_model_receives_scope_and_safe_network_reason(tmp_path):
    from test_adk_runtime import CapturingScriptedLlm, RecordingInstrumentation

    async def scenario():
        async with SocketTarget() as target:
            config = resolver_for(target)
            original = config.resolver

            async def unresolved(host, port):
                if host == "target.example":
                    raise TargetUnresolved
                return await original(host, port)

            async with preflight(target, config=type(config)(resolver=unresolved)) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                tools, state = await tools_for(owner)
                model = CapturingScriptedLlm(
                    model="deterministic-fake",
                    responses=[
                        tool_call(
                            "http_request", {"url": target.origin + "/allowed"}, call_id="h1"
                        ),
                        text_result("DNS failed; coverage is incomplete."),
                    ],
                )
                instrumentation = RecordingInstrumentation()
                worker = await create_runtime(
                    tmp_path,
                    state,
                    {"http_request": tools["http_request"]},
                    model,
                    instrumentation=instrumentation,
                )
                try:
                    await worker.invoke(stage_request())
                    declaration = model.payloads[0]["tools"][0]["function"]
                    assert target.origin in declaration["description"]
                    results = [
                        json.loads(message["content"])
                        for payload in model.payloads
                        for message in payload["messages"]
                        if message["role"] == "tool"
                    ]
                    error = next(result["error"] for result in results if "error" in result)
                    assert error["code"] == "http_dns_failed"
                    assert error["message"] == "DNS resolution failed."
                    assert error["request_may_have_been_sent"] is False
                    spans = [s for s in instrumentation.spans if s.name == "contractor.worker.tool"]
                    assert len(spans) == 1
                    assert spans[0].attributes["error.code"] == "http_dns_failed"
                    assert target.origin not in json.dumps(spans[0].attributes)
                    assert not target.requests
                finally:
                    from datetime import UTC, datetime, timedelta

                    await worker.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["cancel", "deadline", "lease"])
def test_pending_native_rps_wait_never_sends_after_termination(stop):
    from contractor_runtime.core.tool_errors import ToolExecutionError

    async def scenario():
        clock = Clock()
        entered = asyncio.Event()

        async def sleep(delay):
            entered.set()
            if stop == "cancel":
                await asyncio.Event().wait()
            clock.now += 31 if stop == "deadline" else delay

        async with (
            SocketTarget() as target,
            preflight(
                target,
                limits=HTTPLimits(requestsPerSecond=2),
                monotonic=clock,
                sleep=sleep,
            ) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: clock() + 60)
            tools, _state = await tools_for(owner)
            await tools["http_request"](target.origin + "/allowed")
            if stop == "lease":
                lease = clock.now + 0.25
                owner.configure(authority_deadline=lambda: lease)
            call = asyncio.create_task(tools["http_request"](target.origin + "/allowed"))
            async with asyncio.timeout(5):
                await entered.wait()
            if stop == "cancel":
                call.cancel()
            expected = {
                "cancel": asyncio.CancelledError,
                "deadline": HTTPToolError,
                "lease": ToolExecutionError,
            }[stop]
            with pytest.raises(expected) as raised:
                await call
            if stop == "deadline":
                assert raised.value.code == "http_timeout"
                assert not raised.value.model_details["request_may_have_been_sent"]
            assert len(target.requests) == 1
            assert not owner._tasks

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "configured",
    [
        {},
        {"requestBodyBytes": 3, "responseBodyBytes": 5},
        {
            "requestBodyBytes": 2**32 - 1,
            "responseBodyBytes": 2**32 - 1,
            "requestsPerSecond": 2,
            "concurrentRequests": 3,
        },
    ],
)
def test_description_uses_enforced_body_limits(configured):
    from contractor_runtime.shared_config import get_shared_config

    async def scenario():
        async with SocketTarget() as target:
            spec = socket_spec(target)
            spec.engagement.http.routes[0].methods = ["GET", "POST"]
            async with preflight(target, spec=spec, limits=HTTPLimits(**configured)) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                tools, _state = await tools_for(owner)
                description = tools["http_request"].__doc__
                payload = description.split("Allocation-local HTTP limits: ")[1]
                limits, _ = json.JSONDecoder().raw_decode(payload)
                assert limits["configured"] == configured
                shared = get_shared_config()
                assert limits["effectiveBodyLimits"] == {
                    "requestBodyBytes": min(
                        configured.get("requestBodyBytes", 2**32),
                        shared.http_request_body_max_bytes,
                    ),
                    "responseBodyBytes": min(
                        configured.get("responseBodyBytes", 2**32),
                        shared.http_response_body_max_bytes,
                    ),
                }
                assert "no total HTTP request quota" in description
                if configured.get("requestBodyBytes") == 3:
                    await tools["http_request"](
                        target.origin + "/allowed", method="POST", body_type="text", body="abc"
                    )
                    with pytest.raises(HTTPToolError):
                        await tools["http_request"](
                            target.origin + "/allowed", method="POST", body_type="text", body="abcd"
                        )
                    assert len(target.requests) == 1

    asyncio.run(scenario())
