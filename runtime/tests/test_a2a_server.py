from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskState,
)
from a2a.utils.constants import TransportProtocol
from fakes.model import scripted_model, text_result
from fakes.spec import allocation_spec
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value

from contractor_runtime.a2a_server import MAX_A2A_REQUEST_BYTES
from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.server import create_app
from contractor_runtime.state import RuntimeState

SECRET = "recognizable-a2a-test-token"


def model_result(result: str) -> object:
    return text_result(result)


def test_a2a_sdk_round_trip_and_stale_allocation_rejection(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        model = scripted_model([model_result("done")])
        state, service = await allocation_service(tmp_path, model, runtime_capabilities)
        spec = allocation_spec(secret=SECRET)
        prepared = await service.prepare(spec)
        assert service._context is not None and service._context.worker is not None
        worker = service._context.worker
        card_dict = prepared.worker_handle.agent_card
        card = ParseDict(card_dict, AgentCard())
        assert len(card.skills) == 1
        assert card.supported_interfaces[0].tenant == spec.allocation_id
        assert card.supported_interfaces[0].protocol_version == "1.0"
        assert SECRET not in str(card_dict)

        application = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="https://runtime.example",
        ) as http_client:
            card_response = await http_client.get(
                f"/private/v1/allocations/{spec.allocation_id}/a2a/.well-known/agent-card.json"
            )
            assert card_response.status_code == 200
            assert card_response.json()["supportedInterfaces"][0]["tenant"] == spec.allocation_id
            sessions = await worker._session_service.list_sessions(  # type: ignore[attr-defined]
                app_name=worker._app_name,
                user_id=worker._user_id,  # type: ignore[attr-defined]
            )
            assert sessions.sessions == []

            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(card)
            result = await send(client, data_request(spec.allocation_id))
            assert "failure" not in result
            assert result["result"]["result"] == "done"
            assert result["result"]["subtaskId"] == "0"
            assert result["result"]["summarized"] is False
            assert result["stateRevision"] > 0
            assert len(model.requests) == 2

            invalid = await send(client, text_request(spec.allocation_id))
            assert invalid["failure"]["code"] == "invalid_stage_content"
            assert len(model.requests) == 2

            wrong_media = await send(
                client,
                data_request(
                    spec.allocation_id,
                    message_id="wrong-media",
                    media_type="application/json",
                ),
            )
            assert wrong_media["failure"]["code"] == "invalid_stage_content"
            assert len(model.requests) == 2

            wrong_version = await send(
                client,
                data_request(
                    spec.allocation_id,
                    message_id="wrong-version",
                    api_version="contractor.dev/v999",
                ),
            )
            assert wrong_version["failure"]["code"] == "invalid_stage_content"
            assert len(model.requests) == 2

            oversized = await http_client.post(
                f"/private/v1/allocations/{spec.allocation_id}/a2a",
                content=b"x" * (MAX_A2A_REQUEST_BYTES + 1),
            )
            assert oversized.status_code == 413
            assert oversized.json()["code"] == "request_too_large"
            assert len(model.requests) == 2
            sessions = await worker._session_service.list_sessions(  # type: ignore[attr-defined]
                app_name=worker._app_name,
                user_id=worker._user_id,  # type: ignore[attr-defined]
            )
            assert sessions.sessions == []

            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
            with pytest.raises(Exception, match="409"):
                await send(client, data_request(spec.allocation_id))
            assert len(model.requests) == 2

            await service.release(
                ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
            )
            with pytest.raises(Exception, match="409"):
                await send(client, data_request(spec.allocation_id))
            assert len(model.requests) == 2
            await client.close()

    asyncio.run(scenario())


def test_concurrent_a2a_message_receives_worker_busy(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        model = scripted_model([model_result("first")], block=True)
        state, service = await allocation_service(tmp_path, model, runtime_capabilities)
        spec = allocation_spec(secret=SECRET)
        prepared = await service.prepare(spec)
        card = ParseDict(prepared.worker_handle.agent_card, AgentCard())
        assert service._context is not None and service._context.worker is not None
        worker = service._context.worker
        application = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="https://runtime.example",
        ) as http_client:
            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(card)
            first = asyncio.create_task(send(client, data_request(spec.allocation_id)))
            await asyncio.wait_for(model.started.wait(), timeout=1)
            sessions = await worker._session_service.list_sessions(  # type: ignore[attr-defined]
                app_name=worker._app_name,
                user_id=worker._user_id,  # type: ignore[attr-defined]
            )
            assert len(sessions.sessions) == 1
            busy = await send(client, data_request(spec.allocation_id, message_id="second"))
            assert busy["failure"]["code"] == "worker_busy"
            assert len(model.requests) == 1
            sessions = await worker._session_service.list_sessions(  # type: ignore[attr-defined]
                app_name=worker._app_name,
                user_id=worker._user_id,  # type: ignore[attr-defined]
            )
            assert len(sessions.sessions) == 1
            model.release()
            completed = await first
            assert completed["result"]["result"] == "first"
            sessions = await worker._session_service.list_sessions(  # type: ignore[attr-defined]
                app_name=worker._app_name,
                user_id=worker._user_id,  # type: ignore[attr-defined]
            )
            assert sessions.sessions == []
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalization-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=2),
                )
            )
            await client.close()

    asyncio.run(scenario())


def test_return_immediately_exposes_working_task_while_worker_continues(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        model = scripted_model([model_result("done")], block=True)
        state, service = await allocation_service(tmp_path, model, runtime_capabilities)
        spec = allocation_spec(secret=SECRET)
        prepared = await service.prepare(spec)
        card = ParseDict(prepared.worker_handle.agent_card, AgentCard())
        application = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application),
            base_url="https://runtime.example",
        ) as http_client:
            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(card)
            request = data_request(spec.allocation_id)
            request.configuration.CopyFrom(SendMessageConfiguration(return_immediately=True))
            responses = []
            async for response in client.send_message(request):
                responses.append(response)
            assert len(responses) == 1 and responses[0].HasField("task")
            task = responses[0].task
            assert task.status.state == TaskState.TASK_STATE_WORKING
            # RETURN_IMMEDIATELY guarantees an observable working Task, not
            # that every asynchronous before-run callback has already reached
            # the model provider in the same event-loop turn.
            await asyncio.wait_for(model.started.wait(), timeout=1)

            model.release()
            for _ in range(100):
                task = await client.get_task(GetTaskRequest(tenant=spec.allocation_id, id=task.id))
                if task.status.state == TaskState.TASK_STATE_COMPLETED:
                    break
                await asyncio.sleep(0.01)
            assert task.status.state == TaskState.TASK_STATE_COMPLETED
            result = result_from_message(task.status.message)
            assert result["result"]["result"] == "done"
            await client.close()

    asyncio.run(scenario())


async def allocation_service(
    tmp_path: Path, model: object, capabilities: CapabilitySnapshot
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-a2a-test")
    await state.mark_registered()
    factories = built_in_factories(
        tmp_path,
        model_factory=lambda _: model,  # type: ignore[arg-type,return-value]
    )
    service = AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
    )
    return state, service


async def send(client: object, request: SendMessageRequest) -> dict[str, object]:
    responses = []
    async for response in client.send_message(request):  # type: ignore[attr-defined]
        responses.append(response)
    assert len(responses) == 1
    response = responses[0]
    if response.HasField("message"):
        message = response.message
    else:
        assert response.HasField("task") and response.task.status.HasField("message")
        message = response.task.status.message
    return result_from_message(message)


def result_from_message(message: Message) -> dict[str, object]:
    assert len(message.parts) == 1 and message.parts[0].HasField("data")
    assert message.parts[0].media_type == "application/vnd.contractor.worker-completion+json"
    value = MessageToDict(message.parts[0].data)
    assert isinstance(value, dict)
    return value


def data_request(
    allocation_id: str,
    *,
    message_id: str = "message-1",
    api_version: str = API_VERSION,
    media_type: str = "application/vnd.contractor.stage-content+json",
) -> SendMessageRequest:
    payload = {
        "apiVersion": api_version,
        "subtaskId": "0",
        "objective": "Produce a result",
        "instructions": "Follow the template instructions.",
        "parameters": {"mode": "test"},
        "artifacts": {},
    }
    return SendMessageRequest(
        tenant=allocation_id,
        message=Message(
            role=Role.ROLE_USER,
            message_id=message_id,
            parts=[Part(data=ParseDict(payload, Value()), media_type=media_type)],
        ),
    )


def text_request(allocation_id: str) -> SendMessageRequest:
    return SendMessageRequest(
        tenant=allocation_id,
        message=Message(
            role=Role.ROLE_USER,
            message_id="invalid-text",
            parts=[Part(text="not a StageContentRequest")],
        ),
    )
