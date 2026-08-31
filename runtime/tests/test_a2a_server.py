from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, Role, SendMessageRequest
from a2a.utils.constants import TransportProtocol
from fakes.model import json_result, scripted_model
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


def test_a2a_sdk_round_trip_and_stale_allocation_rejection(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        model = scripted_model([json_result(success_payload("done"))])
        state, service = await allocation_service(tmp_path, model, runtime_capabilities)
        spec = allocation_spec(secret=SECRET)
        prepared = await service.prepare(spec)
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

            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC],
                )
            ).create(card)
            result = await send(client, data_request(spec.allocation_id))
            assert result["outcome"] == "succeeded"
            assert result["summary"] == "done"
            assert len(model.requests) == 1

            invalid = await send(client, text_request(spec.allocation_id))
            assert invalid["error"]["code"] == "invalid_stage_content"
            assert len(model.requests) == 1

            wrong_version = await send(
                client,
                data_request(
                    spec.allocation_id,
                    message_id="wrong-version",
                    api_version="contractor.dev/v999",
                ),
            )
            assert wrong_version["error"]["code"] == "invalid_stage_content"
            assert len(model.requests) == 1

            oversized = await http_client.post(
                f"/private/v1/allocations/{spec.allocation_id}/a2a",
                content=b"x" * (MAX_A2A_REQUEST_BYTES + 1),
            )
            assert oversized.status_code == 413
            assert oversized.json()["code"] == "request_too_large"
            assert len(model.requests) == 1

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
            assert len(model.requests) == 1

            await service.release(
                ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
            )
            with pytest.raises(Exception, match="409"):
                await send(client, data_request(spec.allocation_id))
            assert len(model.requests) == 1
            await client.close()

    asyncio.run(scenario())


def test_concurrent_a2a_message_receives_worker_busy(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot
) -> None:
    async def scenario() -> None:
        model = scripted_model([json_result(success_payload("first"))], block=True)
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
            first = asyncio.create_task(send(client, data_request(spec.allocation_id)))
            await asyncio.wait_for(model.started.wait(), timeout=1)
            busy = await send(client, data_request(spec.allocation_id, message_id="second"))
            assert busy["error"]["code"] == "worker_busy"
            assert len(model.requests) == 1
            model.release()
            completed = await first
            assert completed["summary"] == "first"
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
    assert responses[0].HasField("message")
    message = responses[0].message
    assert len(message.parts) == 1 and message.parts[0].HasField("data")
    value = MessageToDict(message.parts[0].data)
    assert isinstance(value, dict)
    return value


def data_request(
    allocation_id: str,
    *,
    message_id: str = "message-1",
    api_version: str = API_VERSION,
) -> SendMessageRequest:
    payload = {
        "apiVersion": api_version,
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
            parts=[Part(data=ParseDict(payload, Value()))],
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


def success_payload(summary: str) -> dict[str, object]:
    return {
        "apiVersion": API_VERSION,
        "outcome": "succeeded",
        "summary": summary,
        "artifacts": {},
    }
