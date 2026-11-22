"""One submission path shared by the general, code and HTTP finding facades."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics
from contractor_runtime.finding_proposals import FindingProposalClient
from contractor_runtime.toolsets.common.argument_diagnostics import NormalizationNotices
from contractor_runtime.toolsets.common.call_identity import call_client_key
from contractor_runtime.toolsets.common.check_plans import ProposedCheck
from contractor_runtime.toolsets.common.cwe import (
    check_cwe_reference,
    cwe_reference,
)
from contractor_runtime.toolsets.common.finding_locations import (
    StandardReference,
    normalize_locations,
)
from contractor_runtime.toolsets.common.http_capture import RecentHTTPCaptures
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.security_findings.anchors import (
    ExactSourceCache,
    prepare_anchors,
)
from contractor_runtime.toolsets.security_findings.memories import (
    MAX_MEMORY_FACT_CALLS,
    memory_evidence,
)
from contractor_runtime.toolsets.security_findings.submission import (
    build_submission,
    normalize_evidence_refs,
    validate_informational_payload,
)


class FindingPublisher:
    def __init__(
        self,
        client: FindingProposalClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        resources: ToolResources,
        artifacts: ArtifactClient | None = None,
        namespace: str = "facts",
        source_preparer=None,
    ):
        self.client = client
        self.metrics = metrics
        self.secrets = secrets
        self.resources = resources
        self.artifacts = artifacts
        self.namespace = namespace
        self._fact_invocation: str | None = None
        self._fact_calls: set[str] = set()
        # The publisher lives as long as the allocation's tools.
        self.sources = ExactSourceCache()
        self.source_preparer = source_preparer

    def _resolve_evidence_refs(self, refs):
        if refs is not None and (not isinstance(refs, list) or len(refs) > 256):
            raise ToolInputError("evidence_refs must be an array of at most 256 selected materials")
        return [
            self.resources.selected_versions.resolve(ref).model_dump(exclude_none=True)
            for ref in refs or []
        ]

    async def prepare_source_selection(self, kwargs):
        selected = kwargs.pop("source_selection", None)
        if selected is not None:
            if self.resources.proposal_collector is not None:
                self.resources.proposal_collector().check_finding_item(
                    kwargs["context"].invocation_id,
                    kwargs.get("item_key"),
                    incidental=kwargs.get("incidental", False),
                )
            if self.source_preparer is None:
                raise RuntimeError("Finding source preparation is unavailable")
            context = kwargs["context"]
            notices = NormalizationNotices()
            refs, request_id = await self.source_preparer.prepare_for_finding(
                selected,
                context.invocation_id,
                call_client_key(context),
                max_sources=4 if kwargs.get("_tool_name") == "fact" else 256,
                additional_sources=int(kwargs.get("memory_name") is not None),
                notices=notices,
            )
            normalization = {}
            notices.attach(normalization)
            kwargs["_source_normalization"] = normalization.get("normalization")
            kwargs["evidence_refs"] = [ref.model_dump(exclude_none=True) for ref in refs]
            kwargs["request_id"] = request_id
        else:
            kwargs["evidence_refs"] = self._resolve_evidence_refs(kwargs.get("evidence_refs"))
        return kwargs

    async def submit(self, **kwargs):
        with RecordedToolCall(
            self.metrics,
            kwargs.get("_tool_name", "finding"),
            {},
            secrets=self.secrets,
            bound_error=_bounded_finding_error,
        ) as call:
            kwargs = await self.prepare_source_selection(kwargs)
            return await self._submit(**kwargs, _call=call)

    async def _submit(
        self,
        *,
        title: str,
        description: str,
        locations: list,
        context: ToolContext,
        cwe: str | None,
        evidence_refs: list[ArtifactRef] | None = None,
        standard_refs: list[StandardReference] | None = None,
        proposed_checks: list[ProposedCheck] | None = None,
        request_id: int | None = None,
        memory_name: str | None = None,
        severity_suggestion: str | None = None,
        item_key: str | None = None,
        incidental: bool = False,
        _call: RecordedToolCall | None = None,
        _tool_name: str = "finding",
        _source_normalization=None,
    ) -> dict[str, Any]:
        recording = (
            nullcontext(_call)
            if _call is not None
            else RecordedToolCall(
                self.metrics,
                _tool_name,
                {},
                secrets=self.secrets,
                bound_error=_bounded_finding_error,
            )
        )
        with recording as call:
            if _tool_name == "fact" and "memories" in self.resources.invocation_artifacts:
                if self._fact_invocation != context.invocation_id:
                    self._fact_invocation = context.invocation_id
                    self._fact_calls.clear()
                key = call_client_key(context)
                if key not in self._fact_calls and len(self._fact_calls) >= MAX_MEMORY_FACT_CALLS:
                    raise ToolInputError(
                        "Fact extraction permits at most eight calls per invocation"
                    )
                self._fact_calls.add(key)
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
            checks = (
                [
                    check.model_dump() if isinstance(check, ProposedCheck) else check
                    for check in proposed_checks
                ]
                if isinstance(proposed_checks, list)
                else proposed_checks
            )
            for reference in cwe_reference(cwe):
                if reference not in standards:
                    standards.append(reference)
            exchange = await self._http_evidence(request_id, context.invocation_id)
            request = build_submission(
                invocation_id=context.invocation_id,
                client_key=call_client_key(context),
                title=title,
                description=description,
                evidence_refs=refs,
                standard_refs=standards,
                proposed_checks=checks,
                severity_suggestion=severity_suggestion,
                item_key=item_key,
                incidental=incidental,
            )
            proposal = request["proposal"]
            # Account for the selected memory in the authored size/source bounds
            # before materializing it. Its exact ref is filled after all preflights.
            if memory_name is not None:
                proposal["evidence_ids"].append(f"evidence-{len(refs) + 1}")
            if normalized:
                proposal["locations"] = normalized
            if exchange is not None:
                request["httpEvidenceInvocationId"], request["httpEvidence"] = exchange
            validate_informational_payload(
                proposal, len(refs) + (exchange is not None) + (memory_name is not None)
            )
            audit_binding = self.resources.proposal_collector
            audit_collector = audit_binding() if audit_binding is not None else None
            if audit_collector is not None:
                audit_collector.check_finding_item(
                    context.invocation_id, item_key, incidental=incidental
                )
                audit_collector.check_proposal_refs(
                    context.invocation_id, proposal["standard_refs"]
                )
            # Audit import refuses CWE references outside the pinned catalog, so
            # reject them before intake. An Audit preflight above reports the
            # same problem as its own field error.
            for reference in proposal["standard_refs"]:
                check_cwe_reference(reference)
            if memory_name is not None:
                if self.artifacts is None:
                    raise RuntimeError("Memory evidence requires an Artifact client")
                snapshot = self.resources.invocation_artifacts.get("memories")
                if snapshot is None:
                    raise ToolInputError(
                        "memory_name requires the assigned exact memories snapshot"
                    )
                ref = await memory_evidence(self.artifacts, snapshot, memory_name, self.namespace)
                request["evidenceRefs"] = normalize_evidence_refs(
                    [*request["evidenceRefs"], ref.model_dump(exclude_none=True)]
                )
            if self.artifacts is not None and proposal["severity_suggestion"] != "informational":
                request["derived_anchors"] = await prepare_anchors(
                    self.artifacts, request, self.sources
                )
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
            if "findingId" in receipt:
                result["finding_id"] = receipt["findingId"]
            if exchange is not None or memory_name is not None:
                result["evidence_refs"] = [item["ref"] for item in receipt["evidence"]]
            call.succeed(result, arguments={"location_count": len(normalized)})
            if _source_normalization:
                result["normalization"] = _source_normalization
            return result

    async def _http_evidence(
        self, request_id: int | None, invocation_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        if request_id is None:
            return None
        captures = self.resources.http_captures
        if not isinstance(captures, RecentHTTPCaptures):
            raise ToolInputError("captured HTTP evidence requires this worker's HTTP session")
        try:
            return await captures.read_exchange(request_id, invocation_id)
        except ValueError:
            raise ToolInputError(
                "request_id is unavailable in this allocation; select a retained request"
            ) from None

    def close(self) -> None:
        self.secrets = ()
        self.sources.close()
        self._fact_calls.clear()
        self._fact_invocation = None


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
