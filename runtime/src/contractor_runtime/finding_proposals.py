"""Allocation-bound client for finding proposal submission (private API)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from contractor_runtime import threads
from contractor_runtime._bulk_rpc import prepare_finding_submission
from contractor_runtime._grpc_metadata import REQUEST_ID_PATTERN
from contractor_runtime.artifacts import (
    ArtifactTransport,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    DIGEST_PATTERN,
    ArtifactRef,
    validate_media_type,
)
from contractor_runtime.contracts.artifacts import (
    FINDING_PROPOSAL_NAMESPACE,
    HTTP_EXCHANGE_NAMESPACE,
    http_exchange_artifact_name,
)
from contractor_runtime.contracts.base import RESOURCE_ID_PATTERN
from contractor_runtime.shared_config import artifact_payload_max_bytes

# findingintake.MaxRequestBytes / auditdomain.MaximumDocumentBytes.
MAX_FINDING_PROPOSAL_BYTES = 8 * 1024 * 1024


class FindingProposalClient:
    """Submits the finding proposals of one active allocation."""

    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        if (
            not isinstance(allocation_id, str)
            or RESOURCE_ID_PATTERN.fullmatch(allocation_id) is None
        ):
            raise ValueError("invalid allocation identity")
        self._allocation_id = allocation_id
        self._transport = transport

    async def submit(self, request: Mapping[str, object]) -> dict[str, object]:
        """Submit one stable allocation-bound finding request.

        A transport loss is retried once with byte-identical content. The
        Server receipt is the idempotency authority; this client never invents
        a second submission identity after an ambiguous commit.
        """

        capture = request.get("httpEvidence")
        capture_identity = (
            {key: capture.get(key) for key in ("source", "request_id")}
            if isinstance(capture, Mapping)
            else None
        )
        submission = await threads.to_thread_until_done(
            prepare_finding_submission,
            request,
            artifact_payload_max_bytes(),
            MAX_FINDING_PROPOSAL_BYTES,
            name="finding-submission-encode",
        )
        request = {
            key: item
            for key, item in submission.header.items()
            if key not in {"proposalSize", "derivedAnchorsSize", "httpEvidenceSize"}
        }
        if capture_identity is not None:
            request["httpEvidence"] = capture_identity
        value = None
        for attempt in range(2):
            try:
                value = await self._transport.submit_finding(self._allocation_id, submission)
                break
            except ArtifactTransportError:
                if attempt != 0:
                    raise
        expected = {
            "apiVersion",
            "proposalId",
            "receiptId",
            "proposal",
            "evidence",
            "replayed",
            "findingId",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ArtifactTransportError("finding proposal API returned an invalid receipt")
        published = value["findingId"]
        if not isinstance(published, str) or REQUEST_ID_PATTERN.fullmatch(published) is None:
            raise ArtifactTransportError("finding proposal API returned an invalid Finding ID")
        if value.get("apiVersion") != "contractor/v1alpha1":
            raise ArtifactTransportError("finding proposal API returned an invalid version")
        for field_name in ("proposalId", "receiptId"):
            candidate = value.get(field_name)
            if not isinstance(candidate, str) or REQUEST_ID_PATTERN.fullmatch(candidate) is None:
                raise ArtifactTransportError("finding proposal API returned an invalid identity")
        if not isinstance(value.get("replayed"), bool):
            raise ArtifactTransportError("finding proposal API returned an invalid replay marker")
        proposal = value.get("proposal")
        if not isinstance(proposal, dict) or set(proposal) != {
            "ref",
            "digest",
            "mediaType",
            "sizeBytes",
        }:
            raise ArtifactTransportError("finding proposal API returned invalid artifact metadata")
        try:
            ref = ArtifactRef.model_validate(proposal.get("ref"))
            ref.require_exact()
        except (ValidationError, ValueError, AttributeError) as error:
            raise ArtifactTransportError(
                "finding proposal API returned invalid artifact metadata"
            ) from error
        digest = proposal.get("digest")
        size = proposal.get("sizeBytes")
        if (
            ref.namespace != FINDING_PROPOSAL_NAMESPACE
            or not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
            or proposal.get("mediaType") != "application/json"
            or type(size) is not int
            or not 0 <= size <= MAX_FINDING_PROPOSAL_BYTES
        ):
            raise ArtifactTransportError("finding proposal API returned invalid artifact metadata")
        evidence = value.get("evidence")
        expected_count = len(request.get("evidenceRefs", [])) + ("httpEvidence" in request)
        if not isinstance(evidence, list) or len(evidence) != expected_count or len(evidence) > 256:
            raise ArtifactTransportError("finding proposal API returned invalid evidence")
        seen = set()
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {
                "ref",
                "digest",
                "mediaType",
                "sizeBytes",
            }:
                raise ArtifactTransportError("finding proposal API returned invalid evidence")
            try:
                exact = ArtifactRef.model_validate(item["ref"]).require_exact()
                validate_media_type(item["mediaType"])
            except (ValueError, AttributeError) as error:
                raise ArtifactTransportError(
                    "finding proposal API returned invalid evidence ref"
                ) from error
            key = (exact.namespace, exact.name, exact.revision)
            if (
                key in seen
                or not isinstance(item["digest"], str)
                or DIGEST_PATTERN.fullmatch(item["digest"]) is None
            ):
                raise ArtifactTransportError(
                    "finding proposal API returned invalid evidence identity"
                )
            seen.add(key)
            if (
                type(item["sizeBytes"]) is not int
                or not 0 <= item["sizeBytes"] <= artifact_payload_max_bytes()
            ):
                raise ArtifactTransportError("finding proposal API returned invalid evidence size")
        for item in request.get("evidenceRefs", []):
            if (item["namespace"], item["name"], item["revision"]) not in seen:
                raise ArtifactTransportError("finding proposal API omitted requested evidence")
        if "httpEvidence" in request:
            selected = evidence[-1]
            if (
                selected["ref"]["namespace"] != HTTP_EXCHANGE_NAMESPACE
                or selected["ref"]["name"]
                != http_exchange_artifact_name(
                    request["httpEvidenceInvocationId"],
                    request["httpEvidence"]["source"],
                    request["httpEvidence"]["request_id"],
                )
                or selected["mediaType"] != "application/vnd.contractor.http-exchange+json"
            ):
                raise ArtifactTransportError("finding proposal API returned invalid HTTP evidence")
        return value
