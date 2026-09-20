"""One submission path shared by the general, code and HTTP finding facades."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.toolsets.security_findings.classification import cwe_reference
from contractor_runtime.toolsets.security_findings.http_evidence import HTTPExchange
from contractor_runtime.toolsets.security_findings.locations import (
    StandardReference,
    normalize_locations,
)
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
        self, client: ArtifactClient, metrics: ToolMetrics, secrets: tuple[str, ...], state: Any
    ):
        self.client = client
        self.metrics = metrics
        self.secrets = secrets
        self.state = state

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
    ) -> dict[str, str]:
        started = time.perf_counter_ns()
        try:
            normalized = normalize_locations(locations)
            refs = [ref.model_dump(exclude_none=True) for ref in evidence_refs or []]
            standards = [ref.model_dump() for ref in standard_refs or []]
            for reference in cwe_reference(cwe):
                if reference not in standards:
                    standards.append(reference)
            exchange, body_ref = await self._http_evidence(request_id, context.invocation_id)
            if body_ref is not None:
                encoded_ref = body_ref.model_dump(exclude_none=True)
                if encoded_ref not in refs:
                    refs.append(encoded_ref)
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
                if body_ref is not None:
                    index = request["evidenceRefs"].index(body_ref.model_dump(exclude_none=True))
                    exchange["response_body_evidence_id"] = proposal["evidence_ids"][index]
                HTTPExchange.model_validate(exchange)
                proposal["http_exchange"] = exchange
            receipt = await self.client.submit_finding_proposal(request)
            result = {
                "proposal_id": str(receipt["proposalId"]),
                "receipt_id": str(receipt["receiptId"]),
                "client_key": proposal["client_key"],
            }
            self.metrics.record_tool_call(
                "finding",
                arguments={"location_count": len(normalized)},
                result=result,
                secrets=self.secrets,
                duration_ms=_elapsed_ms(started),
            )
            return result
        except (ValidationError, UnicodeError, ValueError) as error:
            # Pydantic diagnostics include input values; never log captured
            # Authorization/Cookie/body data when a validation check fails.
            if isinstance(error, ToolInputError):
                bounded = error
            elif isinstance(error, ValidationError):
                # Field paths and error codes are useful for repair. Never include
                # Pydantic's input or context, which may contain request secrets.
                problem = error.errors(include_input=False, include_context=False)[0]
                field = ".".join(str(part) for part in problem["loc"]) or "finding"
                bounded = ToolInputError(
                    f"invalid {field}: {problem['type']}; check the tool schema"
                )
            else:
                bounded = ToolInputError(
                    "invalid finding; check coordinates, classification and evidence"
                )
            self._record_failure(bounded, started)
            raise bounded from None
        except Exception as error:
            self._record_failure(error, started)
            raise

    async def _http_evidence(
        self, request_id: int | None, invocation_id: str
    ) -> tuple[dict[str, Any] | None, ArtifactRef | None]:
        if request_id is None:
            return None, None
        session = getattr(self.state, "http_session", None)
        if session is None:
            raise ToolInputError(
                "request_id requires this worker's HTTP session; omit it to report a URL"
            )
        try:
            return await session.finding_exchange(request_id, invocation_id)
        except ValueError:
            raise ToolInputError(
                "request_id is unavailable in this invocation; use a recent ID or omit it"
            ) from None

    def _record_failure(self, error: Exception, started: int) -> None:
        self.metrics.record_tool_call(
            "finding",
            arguments={},
            error=error,
            secrets=self.secrets,
            duration_ms=_elapsed_ms(started),
        )

    def close(self) -> None:
        self.secrets = ()


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
