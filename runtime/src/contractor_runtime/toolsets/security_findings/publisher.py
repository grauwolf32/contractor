"""One submission path shared by the general, code and HTTP finding facades."""

from __future__ import annotations

import hashlib
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics
from contractor_runtime.finding_proposals import FindingProposalClient
from contractor_runtime.toolsets.common.cwe import (
    check_cwe_reference,
    cwe_reference,
)
from contractor_runtime.toolsets.common.finding_locations import (
    StandardReference,
    normalize_locations,
)
from contractor_runtime.toolsets.common.http_capture import RecentHTTPCaptures
from contractor_runtime.toolsets.security_findings.anchors import prepare_anchors
from contractor_runtime.toolsets.security_findings.submission import build_submission


def call_client_key(context: ToolContext) -> str:
    call_id = getattr(context, "function_call_id", None)
    if not isinstance(call_id, str) or not call_id:
        raise ToolInputError("Runtime tool-call identity is unavailable; finding was not submitted")
    # The provider call ID is retained on the ADK event. Hash the complete value;
    # retries of that event reuse it, while distinct calls remain distinct.
    return "call-" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()


class FindingPublisher:
    def __init__(
        self,
        client: FindingProposalClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        state: Any,
        artifacts: ArtifactClient | None = None,
    ):
        self.client = client
        self.metrics = metrics
        self.secrets = secrets
        self.state = state
        self.artifacts = artifacts

    async def submit(
        self,
        *,
        title: str,
        description: str,
        locations: list,
        context: ToolContext,
        cwe: str | None,
        evidence_refs: list[ArtifactRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
        request_id: int | None = None,
        caido_request_id: str | None = None,
    ) -> dict[str, Any]:
        with RecordedToolCall(
            self.metrics, "finding", {}, secrets=self.secrets, bound_error=_bounded_finding_error
        ) as call:
            normalized = normalize_locations(locations)
            # ADK leaves the whole list as raw values if any model item fails
            # conversion. The submission validator supplies bounded repair text.
            refs = [
                ref.model_dump(exclude_none=True) if isinstance(ref, ArtifactRef) else ref
                for ref in evidence_refs or []
            ]
            standards = [
                ref.model_dump() if isinstance(ref, StandardReference) else ref
                for ref in standard_refs or []
            ]
            for reference in cwe_reference(cwe):
                if reference not in standards:
                    standards.append(reference)
            exchange = await self._http_evidence(
                request_id, caido_request_id, context.invocation_id
            )
            request = build_submission(
                invocation_id=context.invocation_id,
                client_key=call_client_key(context),
                title=title,
                description=description,
                evidence_refs=refs,
                standard_refs=standards,
            )
            proposal = request["proposal"]
            if normalized:
                proposal["locations"] = normalized
            if exchange is not None:
                request["httpEvidence"] = exchange
            audit_binding = getattr(self.state, "audit_completion_binding", None)
            audit_collector = audit_binding.current() if audit_binding is not None else None
            if audit_collector is not None:
                audit_collector.check_proposal_refs(
                    context.invocation_id, proposal["standard_refs"]
                )
            # Audit import refuses CWE references outside the pinned catalog, so
            # reject them before intake. An Audit preflight above reports the
            # same problem as its own field error.
            for reference in proposal["standard_refs"]:
                check_cwe_reference(reference)
            if self.artifacts is not None:
                request["derived_anchors"] = await prepare_anchors(self.artifacts, request)
            receipt = await self.client.submit(request)
            if audit_collector is not None:
                await audit_collector.register_proposal(
                    context.invocation_id, proposal["client_key"], proposal["standard_refs"]
                )
            result = {
                "proposal_id": str(receipt["proposalId"]),
                "receipt_id": str(receipt["receiptId"]),
                "client_key": proposal["client_key"],
            }
            if exchange is not None:
                result["evidence_refs"] = [item["ref"] for item in receipt["evidence"]]
            call.succeed(result, arguments={"location_count": len(normalized)})
            return result

    async def _http_evidence(
        self, request_id: int | None, caido_request_id: str | None, invocation_id: str
    ) -> dict[str, Any] | None:
        if request_id is None and caido_request_id is None:
            return None
        if request_id is not None and caido_request_id is not None:
            raise ToolInputError("select request_id or caido_request_id, not both")
        if request_id is not None and (type(request_id) is not int or request_id <= 0):
            raise ToolInputError("request_id must be an integer returned by http_request")
        if caido_request_id is not None and (
            not isinstance(caido_request_id, str) or not caido_request_id
        ):
            raise ToolInputError("caido_request_id must be an ID from inspected Caido traffic")
        captures = getattr(self.state, "http_captures", None)
        if not isinstance(captures, RecentHTTPCaptures):
            raise ToolInputError(
                "captured HTTP evidence requires this worker's HTTP or Caido session"
            )
        try:
            return await captures.get(
                "http" if request_id is not None else "caido",
                str(request_id) if request_id is not None else caido_request_id,
                invocation_id=invocation_id,
            )
        except ValueError:
            raise ToolInputError(
                "HTTP evidence is unavailable in this invocation; "
                "inspect a recent request or omit the ID"
            ) from None

    def close(self) -> None:
        self.secrets = ()


def _bounded_finding_error(error: Exception) -> Exception:
    # Pydantic diagnostics include input values; never log captured
    # Authorization/Cookie/body data when a validation check fails.
    if isinstance(error, ToolInputError) or not isinstance(
        error, ValidationError | UnicodeError | ValueError
    ):
        return error
    if isinstance(error, ValidationError):
        # Field paths and error codes are useful for repair. Never include
        # Pydantic's input or context, which may contain request secrets.
        problem = error.errors(include_input=False, include_context=False)[0]
        field = ".".join(str(part) for part in problem["loc"]) or "finding"
        return ToolInputError(f"invalid {field}: {problem['type']}; check the tool schema")
    return ToolInputError("invalid finding; check coordinates, classification and evidence")
