from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.security_findings import SecurityFindingsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


def test_finding_uses_runtime_identity_and_exact_evidence() -> None:
    async def scenario() -> None:
        client = FakeFindingClient()
        state = WorkerState()
        factory = SecurityFindingsToolsetFactory(lambda _allocation, _settings: client)  # type: ignore[arg-type]
        tools = await factory.create_selected(
            selected=["finding"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="worker",
            runtime_settings=_settings(),
            workspace=_workspace(),
            state=state,
        )
        result = await tools["finding"](
            client_key="candidate-1",
            title="Missing ownership guard",
            description="The selected path reaches storage without a visible ownership check.",
            subject={"kind": "openapi-operation", "key": "op-1"},
            evidence_refs=[
                {"namespace": "worker", "name": "trace", "revision": "rev-2"},
                {"namespace": "worker", "name": "source", "revision": "rev-1"},
            ],
            tool_context=FakeToolContext("worker-invocation-1"),  # type: ignore[arg-type]
            proposed_checks=[{"objective": "Trace the guard", "method": "static-trace"}],
            severity_suggestion="medium",
        )
        assert result == {"proposal_id": "proposal-1", "receipt_id": "receipt-1"}
        request = client.requests[0]
        assert set(request) == {
            "apiVersion",
            "invocationId",
            "submissionId",
            "proposal",
            "evidenceRefs",
        }
        assert "runId" not in repr(request) and "allocation" not in repr(request)
        assert request["invocationId"] == "worker-invocation-1"
        assert request["evidenceRefs"] == [
            {"namespace": "worker", "name": "source", "revision": "rev-1"},
            {"namespace": "worker", "name": "trace", "revision": "rev-2"},
        ]
        proposal = request["proposal"]
        assert isinstance(proposal, dict)
        assert "hypothesis" not in proposal
        assert proposal["evidence_ids"] == ["evidence-1", "evidence-2"]
        assert state.metrics.counters["tool_calls"] == 1
        assert "ownership check" not in repr(state.metrics.tool_calls[0])

    asyncio.run(scenario())


def test_finding_rejects_non_exact_and_duplicate_evidence() -> None:
    async def scenario() -> None:
        client = FakeFindingClient()
        state = WorkerState()
        tools = await SecurityFindingsToolsetFactory(
            lambda _allocation, _settings: client  # type: ignore[arg-type]
        ).create_selected(
            selected=["finding"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="worker",
            runtime_settings=_settings(),
            workspace=_workspace(),
            state=state,
        )
        with pytest.raises(ValueError, match="exact ArtifactRef"):
            await tools["finding"](
                "candidate-1",
                "Candidate",
                "Description",
                {"kind": "code", "key": "handler"},
                [{"namespace": "worker", "name": "trace"}],
                FakeToolContext("worker-invocation-1"),  # type: ignore[arg-type]
            )
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


def test_finding_transport_loss_retries_byte_identical_submission() -> None:
    async def scenario() -> None:
        transport = LossyFindingTransport()
        client = ArtifactClient("allocation-1", transport)  # type: ignore[arg-type]
        response = await client.submit_finding_proposal(
            {
                "apiVersion": "contractor/v1alpha1",
                "invocationId": "worker-invocation-1",
                "submissionId": "finding-" + "a" * 64,
                "proposal": {"client_key": "candidate-1"},
                "evidenceRefs": [],
            }
        )
        assert response["receiptId"] == "receipt-1"
        assert len(transport.bodies) == 2
        assert transport.bodies[0] == transport.bodies[1]

    asyncio.run(scenario())


class FakeToolContext:
    def __init__(self, invocation_id: str) -> None:
        self.invocation_id = invocation_id


class FakeFindingClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def submit_finding_proposal(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return {
            "apiVersion": "contractor/v1alpha1",
            "proposalId": "proposal-1",
            "receiptId": "receipt-1",
            "proposal": {
                "ref": {
                    "namespace": "finding-proposals",
                    "name": "proposal-1",
                    "revision": "rev-1",
                }
            },
            "replayed": False,
        }


class LossyFindingTransport:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        del headers, max_response_bytes
        assert method == "POST"
        assert path == "/allocations/allocation-1/finding-proposals"
        self.bodies.append(body)
        if len(self.bodies) == 1:
            raise ArtifactTransportError("response lost")
        payload = json.dumps(
            {
                "apiVersion": "contractor/v1alpha1",
                "proposalId": "proposal-1",
                "receiptId": "receipt-1",
                "proposal": {
                    "ref": {
                        "namespace": "finding-proposals",
                        "name": "proposal-1",
                        "revision": "revision-1",
                    },
                    "digest": "sha256:" + "a" * 64,
                    "mediaType": "application/json",
                    "sizeBytes": 2,
                },
                "replayed": True,
            },
            separators=(",", ":"),
        ).encode()
        return ArtifactHTTPResponse(
            200,
            {"content-type": "application/json", "content-length": str(len(payload))},
            payload,
        )


def _settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken="secret",
        artifactApiUrl="https://cp.example/private/v1",
        requestTimeoutSeconds=5,
    )


def _workspace() -> AllocationWorkspace:
    return AllocationWorkspace(
        root=Path("/tmp/contractor-tests"), path=Path("/tmp/contractor-tests/a")
    )
