"""Incremental selected evidence; success means the Server accepted retention."""

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.audit_evidence import AuditEvidenceClient, evidence_snapshot_ref
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.base import is_blank
from contractor_runtime.toolsets.audit_results.arguments import array_argument
from contractor_runtime.toolsets.common.argument_diagnostics import NormalizationNotices
from contractor_runtime.toolsets.common.call_identity import call_client_key
from contractor_runtime.toolsets.common.sources import (
    SourceArgumentError,
    SourcePreparer,
    SourceSelector,
)
from contractor_runtime.toolsets.common.tool_base import ArtifactTool


class EvidenceTool(ArtifactTool):
    name = "evidence"
    description = """Save selected sources and interpretation for one assigned check during work.

    Call read_audit_task for the assigned key and materials. Select actual sources:
    artifact uses kind=artifact and namespace/name of a trusted input or prior
    successful content selection; Runtime supplies its exact revision. HTTP uses
    kind=http and this invocation's integer request_id; no target request is replayed.
    Code uses kind=code, file, start_line and end_line from the opened pinned archive;
    workspace=true captures current workspace coordinates and verifies their source mapping.
    Explain what the sources establish, conditions and uncertainty. Reading a source
    alone does not record check evidence. Reuse exact sources for a different claim
    with an independent interpretation; repeated bytes are not corroboration.
    Inspect persistence=accepted before considering this evidence durable. This does
    not publish a problem Finding or settle an item. Publish evidenced problems,
    including incidental ones, through finding before final result submission.

    Args:
        item_key: Key of the assigned check, including in a singleton.
        sources: Selected code, http or artifact objects; native array or serialized once.
        interpretation: What this selection establishes for the assigned check.
        tested_conditions: Actual conditions examined, native or serialized array.
        limitations: Remaining uncertainty, native or serialized array.

    Returns:
        The accepted evidence receipt and persistence state.
    """

    def __init__(self, collector, client, resources, run_id, namespace, timeout, workspace=None):
        super().__init__(client, resources.metrics, ())
        self.collector = collector
        self.resources = resources
        self.sources = SourcePreparer(client, resources, run_id, namespace, timeout, workspace)

    async def __call__(
        self,
        item_key: str,
        sources: list[SourceSelector] | str,
        interpretation: str,
        tool_context: ToolContext,
        tested_conditions: list[str] | str | None = None,
        limitations: list[str] | str | None = None,
    ):
        with self._recorded({}) as call:
            collector = self.collector()
            notices = NormalizationNotices()
            conditions = array_argument(
                "tested_conditions",
                [] if tested_conditions is None else tested_conditions,
                512,
                notices,
            )
            gaps = array_argument(
                "limitations", [] if limitations is None else limitations, 512, notices
            )

            for field, values, limit in (
                ("interpretation", [interpretation], 65536),
                ("tested_conditions", conditions, 512),
                ("limitations", gaps, 512),
            ):
                try:
                    valid = all(
                        isinstance(value, str)
                        and not is_blank(value)
                        and "\x00" not in value
                        and len(value.encode()) <= limit
                        for value in values
                    ) and len(set(values)) == len(values)
                except (UnicodeError, TypeError):
                    valid = False
                if not valid:
                    raise SourceArgumentError(field, "Use distinct, nonempty bounded text.")

            async def save():
                refs = await self.sources.prepare(
                    sources,
                    tool_context.invocation_id,
                    call_client_key(tool_context),
                    notices=notices,
                )
                for ref in refs:
                    self.resources.selected_versions.admit_write(evidence_snapshot_ref(ref))
                return await AuditEvidenceClient(
                    self._client, run_id=self.sources.run_id, audit_id=collector.audit_id
                ).submit(
                    {
                        "invocationId": tool_context.invocation_id,
                        "submissionId": call_client_key(tool_context),
                        "itemKey": item_key,
                        "sources": [ref.model_dump(exclude_none=True) for ref in refs],
                        "interpretation": interpretation,
                        "testedConditions": conditions,
                        "limitations": gaps,
                    }
                )

            receipt = await collector.record_evidence(tool_context.invocation_id, item_key, save)
            if receipt["material"]["scope_id"] != collector.audit_id:
                raise RuntimeError("Evidence receipt belongs to another Audit")
            for selected in receipt["selectedSources"]:
                self.resources.selected_versions.observe(
                    ArtifactRef.model_validate(selected["ref"])
                )
            result = {
                "sourceSelectors": [
                    {
                        "kind": "artifact",
                        "namespace": selected["ref"]["namespace"],
                        "name": selected["ref"]["name"],
                    }
                    for selected in receipt["selectedSources"]
                ],
                "status": "recorded",
                "persistence": "accepted",
                **receipt,
            }
            notices.attach(result)
            call.succeed({"itemKey": item_key, "persistence": "accepted"})
            return result
