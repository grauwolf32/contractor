"""Recovery retries one model call; it never reruns the surrounding invocation."""

import asyncio
import json

import httpx
import pytest

from contractor_runtime.llm.client import GatewayRequestError, new_gateway_client
from contractor_runtime.llm.recovery import RecoveryDecision


class Authority:
    def __init__(self, *, wait=False):
        self.events = []
        self.wait = wait
        self.observed = asyncio.Event()

    async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
        self.events.append((model, request_id, action, code, retry_after_seconds))
        self.observed.set()
        return RecoveryDecision(
            allowed=not self.wait,
            retry_after_seconds=60,
            request_timeout_seconds=1,
            requires_retry=self.wait,
        )


def test_model_unload_retries_only_identical_model_request():
    async def scenario():
        authority = Authority()
        requests = []

        async def gateway(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(400, json={"error": "Model is unloaded."})
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key="secret",
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            payload = {"model": "worker", "messages": [{"role": "tool", "content": "already done"}]}
            assert await handle.complete(payload) == {"ok": True}
        assert len(requests) == 2
        assert (
            requests[0].content
            == requests[1].content
            == json.dumps(payload, separators=(",", ":")).encode()
        )
        assert [event[2] for event in authority.events] == [
            "acquire",
            "failed",
            "acquire",
            "succeeded",
        ]
        assert authority.events[1][3] == "model_unavailable"
        assert authority.events[0][1] == authority.events[1][1]
        assert authority.events[2][1] == authority.events[3][1] != authority.events[0][1]
        assert "secret" not in repr(authority.events)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,body",
    [
        (400, {"error": "invalid messages"}),
        (401, {"error": "bad credential"}),
        (429, {"error": {"code": "insufficient_quota"}}),
    ],
)
def test_permanent_error_does_not_block_route_or_retry(status, body):
    async def scenario():
        authority = Authority()
        requests = []

        async def gateway(request):
            requests.append(request)
            return httpx.Response(status, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            with pytest.raises(GatewayRequestError) as caught:
                await handle.complete({"model": "worker"})
        assert not caught.value.retryable
        assert len(requests) == 1
        assert [event[2] for event in authority.events] == ["acquire", "finished"]

    asyncio.run(scenario())


def test_cancellation_interrupts_manual_recovery_wait_without_model_call():
    async def scenario():
        authority = Authority(wait=True)

        async def gateway(_request):
            pytest.fail("waiting invocation sent a model request")

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            task = asyncio.create_task(handle.complete({"model": "worker"}))
            await asyncio.wait_for(authority.observed.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        assert [event[2] for event in authority.events] == ["acquire"]

    asyncio.run(scenario())


def test_cancelled_probe_releases_its_grant_before_propagating():
    async def scenario():
        authority = Authority()
        sent = asyncio.Event()

        async def gateway(_request):
            sent.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            task = asyncio.create_task(handle.complete({"model": "worker"}))
            await asyncio.wait_for(sent.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        assert [event[2] for event in authority.events] == ["acquire", "finished"]
        assert authority.events[0][1] == authority.events[1][1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,terminal",
    [
        (200, ("succeeded", None, 0)),
        (503, ("failed", "gateway_unavailable", 0)),
        (400, ("finished", None, 0)),
    ],
)
def test_cancellation_while_reporting_a_granted_outcome_redelivers_it(status, terminal):
    class BlockingOutcome(Authority):
        def __init__(self):
            super().__init__()
            self.blocked = asyncio.Event()

        async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
            decision = await super().update(model, request_id, action, code, retry_after_seconds)
            if action != "acquire" and not self.blocked.is_set():
                self.blocked.set()
                await asyncio.Event().wait()
            return decision

    async def scenario():
        authority = BlockingOutcome()

        async def gateway(_request):
            body = {"ok": True} if status == 200 else {"error": "unavailable"}
            return httpx.Response(status, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                http_client=http,
                recovery=authority,
            )
            task = asyncio.create_task(handle.complete({"model": "worker"}))
            await asyncio.wait_for(authority.blocked.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        request_ids = {event[1] for event in authority.events}
        assert len(request_ids) == 1
        assert [event[2:] for event in authority.events] == [
            ("acquire", None, 0),
            terminal,
            terminal,
        ]

    asyncio.run(scenario())


def test_cancellation_before_a_grant_sends_no_release():
    class BlockingAcquire(Authority):
        async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
            await super().update(model, request_id, action, code, retry_after_seconds)
            await asyncio.Event().wait()

    async def scenario():
        authority = BlockingAcquire()
        handle = new_gateway_client(
            base_url="https://gateway.test/v1",
            api_key=None,
            timeout_seconds=1,
            recovery=authority,
        )
        task = asyncio.create_task(handle.complete({"model": "worker"}))
        await asyncio.wait_for(authority.observed.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await handle.close()
        assert [event[2] for event in authority.events] == ["acquire"]

    asyncio.run(scenario())


def test_unexpected_probe_failure_releases_grant_and_tolerates_lost_authority(monkeypatch):
    from contractor_runtime.artifacts import ArtifactTransportError
    from contractor_runtime.llm import client as client_module

    class Defect(Exception):
        pass

    async def defective_transport(self, payload, max_retries, timeout_seconds):
        raise Defect

    class UnreachableOnRelease(Authority):
        async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
            decision = await super().update(model, request_id, action, code, retry_after_seconds)
            if action == "finished":
                raise ArtifactTransportError("connection refused")
            return decision

    class HangingOnRelease(Authority):
        async def update(self, model, request_id, action, code=None, retry_after_seconds=0):
            decision = await super().update(model, request_id, action, code, retry_after_seconds)
            if action == "finished":
                await asyncio.Event().wait()
            return decision

    monkeypatch.setattr(
        client_module.GatewayClientHandle, "_complete_transport", defective_transport
    )
    monkeypatch.setattr(client_module, "RECOVERY_RELEASE_TIMEOUT_SECONDS", 0.01)

    async def scenario():
        for authority in (Authority(), UnreachableOnRelease(), HangingOnRelease()):
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key=None,
                timeout_seconds=1,
                recovery=authority,
            )
            with pytest.raises(Defect):
                await asyncio.wait_for(handle.complete({"model": "worker"}), 1)
            await handle.close()
            assert [event[2] for event in authority.events] == ["acquire", "finished"]

    asyncio.run(scenario())


# --- GatewayRecoveryClient authority loop -----------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedTransport:
    """Yield transport errors, status codes or bodies in order; last entry repeats."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def request(self, method, path, *, headers, body, max_response_bytes):
        from contractor_runtime.artifacts import ArtifactHTTPResponse, ArtifactTransportError

        self.calls += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if step == "transport":
            raise ArtifactTransportError("connection refused")
        if isinstance(step, int):
            return ArtifactHTTPResponse(step, {}, b"")
        return ArtifactHTTPResponse(200, {}, step)


DECISION = json.dumps(
    {"allowed": True, "retryAfterSeconds": 0, "requestTimeoutSeconds": 1, "requiresRetry": False}
).encode()


def recovery_client(script, clock, *, on_retry=None):
    from contractor_runtime.llm.recovery import GatewayRecoveryClient

    transport = ScriptedTransport(script)
    return (
        GatewayRecoveryClient(
            "alloc-1",
            transport,
            on_retry=on_retry,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            jitter=lambda: 0.5,
        ),
        transport,
    )


def test_authority_5xx_is_bounded_by_its_own_deadline_but_transport_loss_is_not():
    from contractor_runtime.llm.recovery import (
        AUTHORITY_ERROR_DEADLINE_SECONDS,
        AUTHORITY_RECONNECT_CEILING_SECONDS,
        RecoveryStoppedError,
    )

    async def scenario():
        clock = FakeClock()
        client, transport = recovery_client([503], clock)
        started = clock.now
        with pytest.raises(RecoveryStoppedError, match="deadline"):
            await client.update("worker", "req-1", "acquire")
        elapsed = clock.now - started
        assert AUTHORITY_ERROR_DEADLINE_SECONDS <= elapsed < AUTHORITY_ERROR_DEADLINE_SECONDS + 10
        assert transport.calls > 20
        # Delays double from 0.5s and never exceed the reconnect ceiling.
        assert clock.sleeps[:4] == [0.5, 1.0, 2.0, 4.0]
        assert max(clock.sleeps) == AUTHORITY_RECONNECT_CEILING_SECONDS

        # Transport loss alone waits indefinitely (the lease bounds it); a
        # decision after a long outage is still honoured.
        clock = FakeClock()
        script = ["transport"] * 200 + [DECISION]
        client, transport = recovery_client(script, clock)
        decision = await client.update("worker", "req-2", "acquire")
        assert decision.allowed and transport.calls == 201
        assert clock.now - 1000.0 > AUTHORITY_ERROR_DEADLINE_SECONDS

        # A transport gap resets the 5xx deadline: the bound is on continuous
        # authority errors, not on the whole call.
        clock = FakeClock()
        script = [503] * 20 + ["transport"] + [503] * 20 + [DECISION]
        client, transport = recovery_client(script, clock)
        assert (await client.update("worker", "req-3", "acquire")).allowed

    asyncio.run(scenario())


def test_refused_authority_connection_is_retried_as_transport_loss():
    import socket
    import ssl

    from contractor_runtime.artifacts import MTLSArtifactTransport
    from contractor_runtime.llm.recovery import GatewayRecoveryClient

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    class Stop(Exception):
        pass

    async def stop_after_first_retry(seconds):
        raise Stop

    async def scenario():
        causes = []
        client = GatewayRecoveryClient(
            "alloc-1",
            MTLSArtifactTransport(
                f"https://127.0.0.1:{port}/private/v1",
                ssl.create_default_context(),
                timeout_seconds=3,
                runtime_instance_id="runtime-1",
            ),
            on_retry=causes.append,
            sleep=stop_after_first_retry,
        )
        with pytest.raises(Stop):
            await client.update("worker", "req-1", "acquire")
        assert causes == ["transport"]

    asyncio.run(scenario())


def test_authority_4xx_and_invalid_decisions_stop_without_retry():
    from contractor_runtime.llm.recovery import RecoveryStoppedError

    async def scenario():
        for script, match in [
            ([404], "no longer available"),
            ([b"not json"], "invalid decision"),
            ([b'{"allowed": "yes"}'], "invalid decision"),
            (
                [
                    b'{"allowed": true, "retryAfterSeconds": -1,'
                    b' "requestTimeoutSeconds": 1, "requiresRetry": false}'
                ],
                "invalid decision",
            ),
        ]:
            clock = FakeClock()
            client, transport = recovery_client(script, clock)
            with pytest.raises(RecoveryStoppedError, match=match):
                await client.update("worker", "req", "finished")
            assert transport.calls == 1 and clock.sleeps == []

    asyncio.run(scenario())


def test_authority_retries_are_logged_rate_limited_and_counted(caplog):
    from contractor_runtime.llm.recovery import AUTHORITY_RETRY_LOG_INTERVAL_SECONDS

    async def scenario():
        clock = FakeClock()
        causes = []
        script = ["transport"] * 3 + [502] * 2 + [DECISION]
        client, _ = recovery_client(script, clock, on_retry=causes.append)
        with caplog.at_level("WARNING", logger="contractor_runtime.llm.recovery"):
            await client.update("worker", "req", "acquire")
        assert causes == ["transport", "transport", "transport", "http_502", "http_502"]
        records = [
            record for record in caplog.records if "recovery authority" in record.getMessage()
        ]
        # Five retries within the log interval produce exactly one warning.
        assert sum(clock.sleeps) < AUTHORITY_RETRY_LOG_INTERVAL_SECONDS
        assert len(records) == 1
        assert "action=acquire cause=transport attempts=1" in records[0].getMessage()
        assert "req" not in records[0].getMessage()

        clock = FakeClock()
        script = [503] * 15 + [DECISION]
        client, _ = recovery_client(script, clock)
        caplog.clear()
        with caplog.at_level("WARNING", logger="contractor_runtime.llm.recovery"):
            await client.update("worker", "req", "acquire")
        records = [
            record for record in caplog.records if "recovery authority" in record.getMessage()
        ]
        assert sum(clock.sleeps) > AUTHORITY_RETRY_LOG_INTERVAL_SECONDS
        assert 2 <= len(records) <= 3
        assert "cause=http_503" in records[-1].getMessage()

    asyncio.run(scenario())


def test_lost_authority_surfaces_as_typed_recovery_failure():
    from contractor_runtime.telemetry.metrics import MetricsState

    async def scenario():
        clock = FakeClock()
        metrics = MetricsState()
        client, _ = recovery_client([502], clock, on_retry=metrics.record_recovery_authority_retry)
        from contractor_runtime.llm.recovery import AUTHORITY_ERROR_DEADLINE_SECONDS

        async def gateway(_request):
            pytest.fail("a model request was sent without authority")

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://gateway.test/v1",
                api_key="secret",
                timeout_seconds=1,
                http_client=http,
                recovery=client,
            )
            with pytest.raises(GatewayRequestError) as caught:
                await handle.complete({"model": "worker"})
        error = caught.value
        assert error.provider_error_type == "RecoveryAuthorityUnavailable"
        assert error.retryable is False
        assert error.failure.code == "recovery_authority_unavailable"
        assert "secret" not in repr(vars(error))
        metrics.record_model_error(error)
        assert metrics.counters["llm_recovery_authority_retries"] >= 20
        assert (
            metrics.counters["llm_recovery_authority_retries.http_502"]
            == metrics.counters["llm_recovery_authority_retries"]
        )
        assert (
            "RecoveryAuthorityUnavailable; recovery_authority_unavailable"
            in metrics.errors[-1].message
        )
        assert clock.now - 1000.0 >= AUTHORITY_ERROR_DEADLINE_SECONDS

    asyncio.run(scenario())


def test_bounded_backoff_matches_control_client_shape():
    from contractor_runtime.backoff import bounded_backoff

    assert [bounded_backoff(n, 5.0, lambda: 0.5) for n in (1, 2, 3, 4, 5, 20)] == [
        0.5,
        1.0,
        2.0,
        4.0,
        5.0,
        5.0,
    ]
    assert bounded_backoff(0, 5.0, lambda: 0.5) == 0.5
    assert bounded_backoff(1, 5.0, lambda: 0.0) == 0.375
    assert bounded_backoff(1, 5.0, lambda: 0.999) < 0.625
    assert bounded_backoff(3, 1.0, lambda: 0.999) == 1.0
    with pytest.raises(ValueError):
        bounded_backoff(1, 0, lambda: 0.5)
