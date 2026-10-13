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
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.finding_proposals import FindingProposalClient
from contractor_runtime.toolsets.common.cwe import (
    CWEReferenceError,
    check_cwe_reference,
    cwe_reference,
)
from contractor_runtime.toolsets.common.finding_locations import ExactEvidenceRef
from contractor_runtime.toolsets.security_findings.facades import GeneralFindingsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


def test_cwe_reference_uses_the_bundled_classification() -> None:
    for cwe in ("CWE-89", "CWE-639", "CWE-532"):
        assert cwe_reference(cwe) == [{"scheme": "CWE", "version": "4.20", "requirement_id": cwe}]
    with pytest.raises(ToolInputError, match="pinned CWE catalog"):
        cwe_reference("CWE-999999")


@pytest.mark.parametrize(
    "cwe",
    [
        "CWE-71",
        "CWE-92",
        "CWE-132",
        "CWE-216",
        "CWE-217",
        "CWE-218",
        "CWE-225",
        "CWE-247",
        "CWE-249",
        "CWE-292",
        "CWE-365",
        "CWE-373",
        "CWE-423",
        "CWE-443",
        "CWE-458",
        "CWE-516",
        "CWE-533",
        "CWE-534",
        "CWE-542",
        "CWE-545",
        "CWE-592",
        "CWE-596",
        "CWE-769",
        "CWE-1187",
        "CWE-1324",
    ],
)
def test_cwe_reference_rejects_deprecated_v420_weaknesses(cwe: str) -> None:
    with pytest.raises(ToolInputError, match="pinned CWE catalog"):
        cwe_reference(cwe)


def test_cwe_standard_reference_check_matches_the_pinned_catalog() -> None:
    check_cwe_reference({"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-639"})
    # Other schemes are Audit standards, checked against assigned tasks instead.
    check_cwe_reference({"scheme": "OWASP-ASVS", "version": "4.9", "requirement_id": "CWE-534"})
    for reference, reason in (
        (
            {"scheme": "CWE", "version": "4.9", "requirement_id": "CWE-79"},
            "Use version 4.20 of the pinned CWE catalog, or pass the weakness ID as cwe.",
        ),
        (
            {"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-534"},
            "Use a non-deprecated weakness ID from the pinned CWE catalog, "
            "or omit the CWE reference.",
        ),
        (
            {"scheme": "CWE", "version": "4.20", "requirement_id": "CWE-999999"},
            "Use a non-deprecated weakness ID from the pinned CWE catalog, "
            "or omit the CWE reference.",
        ),
    ):
        with pytest.raises(CWEReferenceError) as error:
            check_cwe_reference(reference)
        assert error.value.reason == reason
        assert str(error.value) == f"standard_refs: {reason}"
        assert isinstance(error.value, ToolInputError) and not error.value.retryable


def test_finding_uses_runtime_identity_and_exact_evidence() -> None:
    async def scenario() -> None:
        client = FakeFindingClient()
        state = WorkerState()
        factory = GeneralFindingsToolsetFactory(
            lambda _allocation, _settings: ArtifactClient("allocation-1", client)
        )
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
            title="Missing ownership guard",
            description="The selected path reaches storage without a visible ownership check.",
            evidence_refs=[
                ExactEvidenceRef(namespace="worker", name="trace", revision="rev-2"),
                ExactEvidenceRef(namespace="worker", name="source", revision="rev-1"),
            ],
            tool_context=FakeToolContext("worker-invocation-1"),
        )
        assert result["proposal_id"] == "proposal-1" and result["receipt_id"] == "receipt-1"
        assert result["client_key"].startswith("call-")
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
        tools = await GeneralFindingsToolsetFactory(
            lambda _allocation, _settings: ArtifactClient("allocation-1", client)
        ).create_selected(
            selected=["finding"],
            allocation_id="allocation-1",
            run_id="run-1",
            namespace="worker",
            runtime_settings=_settings(),
            workspace=_workspace(),
            state=state,
        )
        with pytest.raises(ValueError):
            ExactEvidenceRef(namespace="worker", name="trace")
        ref = ExactEvidenceRef(namespace="worker", name="trace", revision="r1")
        with pytest.raises(ValueError, match="duplicate"):
            await tools["finding"](
                title="Candidate",
                description="Description",
                evidence_refs=[ref, ref],
                tool_context=FakeToolContext("worker-invocation-1"),
            )
        assert client.requests == []
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


def test_finding_transport_loss_retries_byte_identical_submission() -> None:
    async def scenario() -> None:
        transport = LossyFindingTransport()
        client = FindingProposalClient("allocation-1", transport)  # type: ignore[arg-type]
        response = await client.submit(
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
        self.function_call_id = "call-test"


class FakeFindingClient:
    """Fakes the finding submissions of one allocation.

    A test that builds a FindingPublisher uses it as the FindingProposalClient
    (``submit``); a test that builds a finding toolset uses it as the private
    API transport of the allocation's ArtifactClient (``request``). Both record
    the submitted requests.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def submit(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return json.loads(_receipt_response(201, replayed=False, request=request).body)

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
        assert path.startswith("/allocations/") and path.endswith("/finding-proposals")
        self.requests.append(json.loads(body))
        return _receipt_response(201, replayed=False, request=self.requests[-1])


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
        return _receipt_response(200, replayed=True, request=json.loads(body))


def _receipt_response(
    status: int, *, replayed: bool, request: dict | None = None
) -> ArtifactHTTPResponse:
    request = request or {}
    refs = list(request.get("evidenceRefs", []))
    if "httpEvidence" in request:
        from contractor_runtime.toolsets.common.http_capture import CapturedHTTPExchange

        CapturedHTTPExchange.model_validate(request["httpEvidence"])
        refs.append(
            {
                "namespace": "finding-proposals",
                "name": "http.exchange.proposal-1",
                "revision": "http-rev-1",
            }
        )
    evidence = [
        {
            "ref": ref,
            "digest": "sha256:" + "a" * 64,
            "mediaType": "application/vnd.contractor.http-exchange+json"
            if ref["name"].startswith("http.exchange.")
            else "text/plain",
            "sizeBytes": 2,
        }
        for ref in refs
    ]
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
            "replayed": replayed,
            "evidence": evidence,
        },
        separators=(",", ":"),
    ).encode()
    return ArtifactHTTPResponse(
        status,
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
