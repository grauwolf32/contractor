from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from contractor_runtime import _https
from contractor_runtime.artifacts import ArtifactHTTPResponse, ArtifactTransportError
from contractor_runtime.contracts import API_VERSION
from contractor_runtime.contracts.planner import PlannerOperationResult
from contractor_runtime.planner import server_host
from contractor_runtime.planner.host import PlannerStoppedError
from contractor_runtime.planner.server_host import ServerPlannerHost


@pytest.mark.parametrize("disconnect", [False, True])
def test_planner_operation_uses_stage_deadline_and_replays_identical_request(
    monkeypatch, disconnect
):
    async def scenario():
        now = datetime(2026, 11, 26, tzinfo=UTC)
        clock = Mock(wraps=datetime)
        clock.now.return_value = now
        monkeypatch.setattr(server_host, "datetime", clock)
        reply = PlannerOperationResult(
            apiVersion=API_VERSION, result={"accepted": True}, stop=True, escalate=True
        )
        response = ArtifactHTTPResponse(200, {}, reply.model_dump_json(by_alias=True).encode())
        responses = (
            [ArtifactTransportError("response lost"), response] if disconnect else [response]
        )
        transport = Mock(request=AsyncMock(side_effect=responses))
        host = ServerPlannerHost("allocation-1", transport, deadline=now + timedelta(hours=1))
        assert await host.execute("execute_current_subtask", {"subtask_id": "0"}) == reply
        attempts = transport.request.call_args_list
        assert len(attempts) == (2 if disconnect else 1)
        assert all(call.kwargs["timeout_seconds"] == 3600 for call in attempts)
        if disconnect:
            assert attempts[0] == attempts[1]

    asyncio.run(scenario())


def test_planner_host_expires_before_sending_an_operation(monkeypatch):
    async def scenario():
        now = datetime(2026, 11, 26, tzinfo=UTC)
        clock = Mock(wraps=datetime)
        clock.now.return_value = now
        monkeypatch.setattr(server_host, "datetime", clock)
        transport = Mock(request=AsyncMock())
        host = ServerPlannerHost("allocation-1", transport, deadline=now)
        with pytest.raises(PlannerStoppedError) as error:
            await host.execute("before_model", {})
        assert error.value.code == "planner_deadline_exceeded"
        transport.request.assert_not_called()

    asyncio.run(scenario())


def test_private_transport_operation_deadline_does_not_change_other_requests():
    from test_artifacts import mtls_transport

    async def scenario():
        connections = Mock(exchange=AsyncMock(return_value=_https.HTTPResponse(200, {}, b"{}")))
        transport = mtls_transport(connections)
        await transport.request(
            "POST",
            "/planner-operations",
            headers={},
            body=b"{}",
            max_response_bytes=2,
            timeout_seconds=3600,
        )
        await transport.request("GET", "/artifacts", headers={}, body=b"", max_response_bytes=2)
        assert [call.kwargs["timeout_seconds"] for call in connections.exchange.call_args_list] == [
            3600,
            1,
        ]

    asyncio.run(scenario())


def test_planner_host_rejects_oversized_operation_without_transport_or_retry():
    async def scenario():
        transport = Mock(request=AsyncMock())
        host = ServerPlannerHost(
            "allocation-1", transport, deadline=datetime.now(UTC) + timedelta(minutes=1)
        )
        with pytest.raises(PlannerStoppedError) as error:
            await host.execute("add_subtask", {"instructions": "x" * (2 << 20)})
        assert error.value.code == "planner_operation_invalid"
        assert not error.value.retryable
        transport.request.assert_not_called()

    asyncio.run(scenario())
