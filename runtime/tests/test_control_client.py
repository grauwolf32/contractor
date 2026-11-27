from __future__ import annotations

import asyncio
import logging
import subprocess
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import grpc
import pytest

from contractor_runtime._grpc_client import ControlRPCError
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.control_client import (
    ControlClient,
    ControlClientError,
    RegistrationRejectedError,
)
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState


def test_registration_commits_idle_and_uses_server_timing(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-register", capabilities=runtime_capabilities)
        transport = FakeTransport(
            [
                {
                    "apiVersion": "contractor/v1alpha1",
                    "heartbeatIntervalSeconds": 12,
                    "confirmedLeaseSeconds": 72,
                }
            ]
        )
        client = ControlClient(make_settings(), state, transport)
        response = await client.register()
        assert response.heartbeat_interval_seconds == 12
        assert client.timing.heartbeat_interval_seconds == 12
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert transport.requests[0][0] == "register"
        assert transport.requests[0][1]["instanceId"] == "runtime-register"
        assert transport.requests[0][1]["softwareVersion"] == "0.1.0"
        assert transport.requests[0][1]["apiVersion"] == "contractor/v1alpha1"
        assert "initialLabels" not in transport.requests[0][1]
        assert "workspaceCapabilities" not in transport.requests[0][1]
        assert "runtimeAgentId" not in transport.requests[0][1]

    asyncio.run(scenario())


def test_response_loss_never_advances_echoed_ack(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-heartbeat", capabilities=runtime_capabilities)
        transport = FakeTransport(
            [
                registration_response(),
                heartbeat_response(1),
                TimeoutError("response containing private-token was lost"),
                heartbeat_response(3),
            ]
        )
        client = ControlClient(make_settings(), state, transport)
        await client.register()
        await client.heartbeat_once()
        assert client.sequence == 1
        assert client.echoed_ack == 1
        with pytest.raises(TimeoutError):
            await client.heartbeat_once()
        assert client.sequence == 2
        assert client.echoed_ack == 1
        await client.heartbeat_once()
        assert client.sequence == 3
        assert client.echoed_ack == 3
        heartbeat_requests = [
            payload for path, payload in transport.requests if "heartbeat" in path
        ]
        assert [request["heartbeatSeq"] for request in heartbeat_requests] == [1, 2, 3]
        assert [request["echoedAckSeq"] for request in heartbeat_requests] == [0, 1, 1]

    asyncio.run(scenario())


def test_wrong_ack_is_rejected_without_changing_echo(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-wrong-ack", capabilities=runtime_capabilities)
        transport = FakeTransport([registration_response(), heartbeat_response(9)])
        client = ControlClient(make_settings(), state, transport)
        await client.register()
        with pytest.raises(ControlClientError):
            await client.heartbeat_once()
        assert client.sequence == 1
        assert client.echoed_ack == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        b'{"apiVersion":"contractor/v1alpha1","ackSeq":9,"ackSeq":1,"action":"continue"}',
        b'{"apiVersion":"contractor/v1alpha1","ackSeq":1,"action":"private-token"}',
        b'{"apiVersion":"contractor/v1alpha1","ackSeq":NaN,"action":"continue"}',
        b'{"ackSeq":1,"action":"continue"}',
    ],
)
def test_invalid_heartbeat_never_renews_lease_or_exposes_response(response: bytes) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-invalid-heartbeat")
        await state.mark_registered()
        watchdog = LeaseWatchdog(noop_expiry)
        await watchdog.arm(60)
        deadline = watchdog.confirmed_deadline
        client = ControlClient(make_settings(), state, FakeTransport([response]), watchdog=watchdog)

        with pytest.raises(ControlClientError) as error:
            await client.heartbeat_once()

        assert client.echoed_ack == 0
        assert watchdog.last_ack == 0
        assert watchdog.confirmed_deadline == deadline
        assert "private-token" not in "".join(traceback.format_exception(error.value))

    asyncio.run(scenario())


def test_lease_deadline_starts_when_request_is_sent_not_answered(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    class SlowTransport(FakeTransport):
        async def exchange(
            self, path: str, payload: Mapping[str, Any]
        ) -> Mapping[str, Any] | bytes:
            nonlocal now
            now += 5.0
            return await super().exchange(path, payload)

    now = 1000.0

    def clock() -> float:
        return now

    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-lease-latency", capabilities=runtime_capabilities)
        watchdog = LeaseWatchdog(noop_expiry, monotonic=clock)
        transport = SlowTransport([registration_response(), heartbeat_response(1)])
        client = ControlClient(make_settings(), state, transport, watchdog=watchdog)

        await client.register()
        assert watchdog.confirmed_deadline == 1000.0 + 60
        await client.heartbeat_once()
        assert watchdog.confirmed_deadline == 1005.0 + 60

    asyncio.run(scenario())


def test_concurrent_heartbeats_are_serialized() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        requests: list[Mapping[str, Any]] = []

        class BlockingTransport:
            async def heartbeat(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
                requests.append(payload)
                started.set()
                await release.wait()
                return heartbeat_response(payload["heartbeatSeq"])

        state = RuntimeState(instance_id="runtime-concurrent-heartbeats")
        await state.mark_registered()
        client = ControlClient(make_settings(), state, BlockingTransport())
        first = asyncio.create_task(client.heartbeat_once())
        await started.wait()
        second = asyncio.create_task(client.heartbeat_once())
        await asyncio.sleep(0)
        release.set()
        responses = await asyncio.gather(first, second)

        assert [response.ack_seq for response in responses] == [1, 2]
        assert [request["echoedAckSeq"] for request in requests] == [0, 1]
        assert client.echoed_ack == 2

    asyncio.run(scenario())


def test_valid_reconciliation_actions_are_applied_after_ack(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-actions", capabilities=runtime_capabilities)
        handler = RecordingReconciliation()
        watchdog = LeaseWatchdog(noop_expiry)
        transport = FakeTransport(
            [
                registration_response(),
                heartbeat_response(1, action="drain", allocation_id="allocation-1"),
                heartbeat_response(2, action="release", allocation_id="allocation-1"),
            ]
        )
        client = ControlClient(
            make_settings(),
            state,
            transport,
            watchdog=watchdog,
            reconciliation=handler,
        )
        await client.register()
        await state.reserve_allocation("allocation-1")
        await state.commit_allocation("allocation-1")
        await client.heartbeat_once()
        await client.heartbeat_once()
        await client.wait_for_reconciliation()

        assert handler.drains == [("allocation-1", 10.0)]
        assert handler.releases == ["allocation-1"]
        assert watchdog.last_ack == 2

    asyncio.run(scenario())


def test_reregister_action_starts_a_new_confirmed_lease_generation(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-reregister", capabilities=runtime_capabilities)
        transport = FakeTransport(
            [
                registration_response(),
                heartbeat_response(1, action="reregister"),
                registration_response(),
            ]
        )
        watchdog = LeaseWatchdog(noop_expiry)
        client = ControlClient(make_settings(), state, transport, watchdog=watchdog)

        await client.register()
        await client.heartbeat_once()

        assert [path for path, _ in transport.requests].count("register") == 2
        assert not watchdog.expired
        assert watchdog.last_ack == 0

    asyncio.run(scenario())


def test_failed_reregistrations_do_not_extend_the_local_lease(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        now = 1000.0
        state = RuntimeState(instance_id="runtime-lost-server", capabilities=runtime_capabilities)
        watchdog = LeaseWatchdog(state.fence_control_lease, monotonic=lambda: now)
        responses: list[Mapping[str, Any] | BaseException] = [registration_response()]
        statuses = [
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.FAILED_PRECONDITION,
            grpc.StatusCode.INVALID_ARGUMENT,
        ] * 2
        for sequence, status in enumerate(statuses, start=1):
            responses.extend(
                [
                    heartbeat_response(sequence, action="reregister"),
                    ControlRPCError(status, "rejected", False, "request_test"),
                ]
            )
        transport = FakeTransport(responses)
        client = ControlClient(make_settings(), state, transport, watchdog=watchdog)
        await client.register()
        deadline = watchdog.confirmed_deadline
        assert deadline == 1060.0

        for sequence, status in enumerate(statuses, start=1):
            now += 10.0
            with pytest.raises(ControlRPCError) as caught:
                await client.heartbeat_once()
            assert caught.value.status_code == status
            assert client.echoed_ack == sequence
            assert watchdog.last_ack == 0
            assert watchdog.confirmed_deadline == (deadline if now < deadline else None)

        assert await watchdog.expire_if_due()
        assert watchdog.expired
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


def test_local_lease_loss_stays_fenced_until_release_registration_succeeds(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        now = 0.0
        state = RuntimeState(
            instance_id="runtime-local-lease-loss", capabilities=runtime_capabilities
        )
        watchdog = LeaseWatchdog(state.fence_control_lease, monotonic=lambda: now)
        transport = FakeTransport(
            [
                registration_response(),
                heartbeat_response(1, action="reregister"),
                TimeoutError("registration unavailable"),
                heartbeat_response(2, action="reregister"),
                registration_response(),
            ]
        )

        class StateReconciliation:
            async def reconcile_drain(self, _allocation_id: str, _grace: float) -> None:
                raise AssertionError("idle slot cannot drain")

            async def confirm_release(
                self, allocation_id: str | None, *, keep_fenced: bool = False
            ) -> None:
                await state.confirm_release(allocation_id, keep_fenced=keep_fenced)

        client = ControlClient(
            make_settings(),
            state,
            transport,
            watchdog=watchdog,
            reconciliation=StateReconciliation(),
        )
        await client.register()
        now = 60.0
        assert await watchdog.expire_if_due()
        assert (await state.snapshot()).process_state is ProcessState.FENCED

        with pytest.raises(TimeoutError):
            await client.heartbeat_once()
        await client.wait_for_reconciliation()
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert watchdog.expired and watchdog.confirmed_deadline is None

        await client.heartbeat_once()
        await client.wait_for_reconciliation()
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert not watchdog.expired and watchdog.confirmed_deadline == 120.0
        assert [path for path, _ in transport.requests].count("register") == 3

        now = 121.0
        assert await watchdog.expire_if_due()
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


def test_registration_retry_backoff_has_fixed_ceiling_before_server_timing() -> None:
    low = ControlClient(make_settings(), RuntimeState(), FakeTransport([]), jitter=lambda: 0.0)
    high = ControlClient(make_settings(), RuntimeState(), FakeTransport([]), jitter=lambda: 1.0)
    assert low._backoff(1) == 0.375
    assert high._backoff(100) == 5.0


def test_retry_logging_does_not_include_exception_message(
    caplog: pytest.LogCaptureFixture,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-redaction", capabilities=runtime_capabilities)
        stop = asyncio.Event()

        async def stop_after_retry(_: float) -> None:
            stop.set()

        transport = FakeTransport([RuntimeError("recognizable-private-token")])
        client = ControlClient(make_settings(), state, transport, sleep=stop_after_retry)
        assert await client.register_until_stopped(stop) is False

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert "recognizable-private-token" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (grpc.StatusCode.INVALID_ARGUMENT, "invalid_request"),
        (grpc.StatusCode.UNAVAILABLE, "internal_error"),
        (grpc.StatusCode.PERMISSION_DENIED, "permission_denied"),
        (grpc.StatusCode.INTERNAL, "invalid_response"),
    ],
)
def test_permanent_registration_rejection_stops_without_retrying(
    status: grpc.StatusCode,
    code: str,
    caplog: pytest.LogCaptureFixture,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        state = RuntimeState(instance_id="runtime-rejected", capabilities=runtime_capabilities)
        transport = FakeTransport(
            [ControlRPCError(status, code, False, "request_1"), registration_response()]
        )
        client = ControlClient(make_settings(), state, transport, sleep=record_sleep)
        with pytest.raises(RegistrationRejectedError) as rejected:
            await client.register_until_stopped(asyncio.Event())
        assert code in str(rejected.value)
        assert "server-message-must-not-escape" not in str(rejected.value)
        assert rejected.value.status_code == status
        assert len(transport.requests) == 1
        assert delays == []
        assert (await state.snapshot()).process_state is ProcessState.STARTING

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert "server-message-must-not-escape" not in caplog.text


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (grpc.StatusCode.UNAVAILABLE, "internal_error"),
        (grpc.StatusCode.FAILED_PRECONDITION, "conflict"),
        (grpc.StatusCode.INVALID_ARGUMENT, "invalid_request"),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, "transport_unavailable"),
    ],
)
def test_retryable_registration_rejection_backs_off_and_registers(
    status: grpc.StatusCode,
    code: str,
    caplog: pytest.LogCaptureFixture,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        state = RuntimeState(instance_id="runtime-retried", capabilities=runtime_capabilities)
        transport = FakeTransport(
            [ControlRPCError(status, code, True, "request_1"), registration_response()]
        )
        client = ControlClient(
            make_settings(), state, transport, sleep=record_sleep, jitter=lambda: 0.0
        )
        assert await client.register_until_stopped(asyncio.Event()) is True
        assert len(transport.requests) == 2
        assert delays == [0.375]
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert code in caplog.text
    assert "server-message-must-not-escape" not in caplog.text


def test_inflight_heartbeat_cancels_cleanly() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-cancel")
        waiting = WaitingTransport()
        client = ControlClient(make_settings(), state, waiting)
        await state.mark_registered()
        task = asyncio.create_task(client.run_heartbeats(asyncio.Event()))
        await waiting.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("loop_name", ["register_until_stopped", "run_heartbeats"])
@pytest.mark.parametrize("cancel", [False, True])
def test_control_loop_interrupts_sleep_and_cleans_up_tasks(
    loop_name: str, cancel: bool, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        stop = asyncio.Event()
        sleeping = asyncio.Event()
        sleep_cancelled = asyncio.Event()

        async def blocked_sleep(_: float) -> None:
            sleeping.set()
            try:
                await asyncio.Event().wait()
            finally:
                sleep_cancelled.set()

        state = RuntimeState(instance_id="runtime-stop-sleep", capabilities=runtime_capabilities)
        await state.mark_registered()
        transport = FakeTransport([TimeoutError()])
        client = ControlClient(make_settings(), state, transport, sleep=blocked_sleep)
        existing_tasks = asyncio.all_tasks()
        task = asyncio.create_task(getattr(client, loop_name)(stop))
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            stop.set()
            await asyncio.wait_for(task, timeout=1)
        assert sleep_cancelled.is_set()
        assert len(transport.requests) == 1
        assert asyncio.all_tasks() == existing_tasks

    asyncio.run(scenario())


@pytest.mark.parametrize("loop_name", ["register_until_stopped", "run_heartbeats"])
def test_control_loop_skips_backoff_when_request_sets_stop(
    loop_name: str, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        stop = asyncio.Event()
        delays: list[float] = []

        class StoppingTransport:
            async def register(self, payload):
                return await self.exchange("register", payload)

            async def heartbeat(self, payload):
                return await self.exchange("heartbeat", payload)

            async def exchange(self, _: str, __: Mapping[str, Any]) -> bytes:
                stop.set()
                raise TimeoutError()

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        state = RuntimeState(instance_id="runtime-stop-request", capabilities=runtime_capabilities)
        await state.mark_registered()
        client = ControlClient(make_settings(), state, StoppingTransport(), sleep=record_sleep)
        await getattr(client, loop_name)(stop)
        assert delays == []

    asyncio.run(scenario())


class FakeTransport:
    async def register(self, payload):
        return await self.exchange("register", payload)

    async def heartbeat(self, payload):
        return await self.exchange("heartbeat", payload)

    def __init__(self, responses: list[Mapping[str, Any] | bytes | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, Mapping[str, Any]]] = []

    async def exchange(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | bytes:
        self.requests.append((path, payload))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class WaitingTransport:
    async def register(self, payload):
        return await self.exchange("register", payload)

    async def heartbeat(self, payload):
        return await self.exchange("heartbeat", payload)

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def exchange(self, _: str, __: Mapping[str, Any]) -> Mapping[str, Any]:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def registration_response() -> Mapping[str, Any]:
    return {
        "apiVersion": "contractor/v1alpha1",
        "heartbeatIntervalSeconds": 10,
        "confirmedLeaseSeconds": 60,
    }


def heartbeat_response(
    sequence: int, *, action: str = "continue", allocation_id: str | None = None
) -> Mapping[str, Any]:
    return {
        "apiVersion": "contractor/v1alpha1",
        "ackSeq": sequence,
        "action": "continue" if allocation_id is not None else action,
        "allocationActions": [{"allocationId": allocation_id, "action": action}]
        if allocation_id is not None
        else [],
    }


async def noop_expiry() -> None:
    return


class RecordingReconciliation:
    def __init__(self) -> None:
        self.drains: list[tuple[str, float]] = []
        self.releases: list[str | None] = []

    async def reconcile_drain(self, allocation_id: str, grace: float) -> None:
        self.drains.append((allocation_id, grace))

    async def confirm_release(
        self, allocation_id: str | None, *, keep_fenced: bool = False
    ) -> None:
        self.releases.append(allocation_id)


def make_settings() -> Settings:
    placeholder = Path("/tmp/contractor-test-placeholder")
    return Settings(
        control_plane_url="https://localhost:8443",
        advertised_control_url="https://localhost:9443",
        advertised_a2a_url="https://localhost:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
    )


def generate_pki(root: Path) -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    for arguments in (
        ("init-ca", "--root", str(root)),
        ("issue-control-plane", "--root", str(root)),
        ("issue-runtime", "--root", str(root), "--name", "agent-1"),
    ):
        subprocess.run(
            ["go", "run", "./cmd/contractor-pki", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    return {
        "ca": root / "ca.crt",
        "control_plane_certificate": root / "control-plane.crt",
        "control_plane_key": root / "control-plane.key",
        "agent_certificate": root / "agents" / "agent-1.crt",
        "agent_key": root / "agents" / "agent-1.key",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("runtimeAgentId", "a" * 64), ("labels", ["private-token"]), ("labelRevision", 1)],
)
def test_registration_rejects_principal_echo_before_state_or_lease_mutation(
    runtime_capabilities: CapabilitySnapshot, field: str, value: object
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-unused-echo", capabilities=runtime_capabilities)
        watchdog = LeaseWatchdog(noop_expiry)
        await watchdog.arm(60)
        deadline = watchdog.confirmed_deadline
        response = dict(registration_response())
        response[field] = value
        client = ControlClient(make_settings(), state, FakeTransport([response]), watchdog=watchdog)

        with pytest.raises(ControlClientError) as error:
            await client.register()

        assert (await state.snapshot()).process_state is ProcessState.STARTING
        assert watchdog.confirmed_deadline == deadline
        assert "private-token" not in "".join(traceback.format_exception(error.value))

    asyncio.run(scenario())
