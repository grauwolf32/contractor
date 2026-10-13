"""Allocation-bound client for finding proposal submission (private API)."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import ValidationError

from contractor_runtime import _https, threads
from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_JSON_BYTES,
    ArtifactHTTPResponse,
    ArtifactTransport,
    ArtifactTransportError,
    allocation_path,
    raise_for_status,
    require_json,
)
from contractor_runtime.contracts import DIGEST_PATTERN, ArtifactRef, validate_media_type

# findingintake.MaxRequestBytes / auditdomain.MaximumDocumentBytes.
MAX_FINDING_SUBMISSION_BYTES = MAX_ARTIFACT_BYTES
MAX_FINDING_PROPOSAL_BYTES = 8 * 1024 * 1024


class FindingProposalClient:
    """Submits the finding proposals of one active allocation."""

    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        self._path = allocation_path(allocation_id, "finding-proposals")
        self._transport = transport

    async def submit(self, request: Mapping[str, object]) -> dict[str, object]:
        """Submit one stable allocation-bound finding request.

        A transport loss is retried once with byte-identical content. The
        Server receipt is the idempotency authority; this client never invents
        a second submission identity after an ambiguous commit.
        """

        body = await threads.to_thread_until_done(
            lambda: json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            name="finding-submission-encode",
        )
        if len(body) > MAX_FINDING_SUBMISSION_BYTES:
            raise ValueError("finding proposal request exceeds its size limit")
        response: ArtifactHTTPResponse | None = None
        for attempt in range(2):
            try:
                response = await self._transport.request(
                    "POST",
                    self._path,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
                )
                break
            except ArtifactTransportError:
                if attempt != 0:
                    raise
        if not (response is not None):
            raise RuntimeError("Expected response is not None")
        raise_for_status(response)
        if response.status_code not in {200, 201}:
            raise ArtifactTransportError("finding proposal API returned an invalid status")
        require_json(response.headers)
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactTransportError("finding proposal API returned invalid JSON") from error
        expected = {"apiVersion", "proposalId", "receiptId", "proposal", "evidence", "replayed"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ArtifactTransportError("finding proposal API returned an invalid receipt")
        if value.get("apiVersion") != "contractor/v1alpha1":
            raise ArtifactTransportError("finding proposal API returned an invalid version")
        for field_name in ("proposalId", "receiptId"):
            candidate = value.get(field_name)
            if (
                not isinstance(candidate, str)
                or _https.REQUEST_ID_PATTERN.fullmatch(candidate) is None
            ):
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
            ref.namespace != "finding-proposals"
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
                or not 0 <= item["sizeBytes"] <= MAX_ARTIFACT_BYTES
            ):
                raise ArtifactTransportError("finding proposal API returned invalid evidence size")
        for item in request.get("evidenceRefs", []):
            if (item["namespace"], item["name"], item["revision"]) not in seen:
                raise ArtifactTransportError("finding proposal API omitted requested evidence")
        if "httpEvidence" in request:
            selected = evidence[-1]
            if (
                selected["ref"]["namespace"] != "finding-proposals"
                or selected["ref"]["name"] != "http.exchange." + value["proposalId"]
                or selected["mediaType"] != "application/vnd.contractor.http-exchange+json"
            ):
                raise ArtifactTransportError("finding proposal API returned invalid HTTP evidence")
        return value
