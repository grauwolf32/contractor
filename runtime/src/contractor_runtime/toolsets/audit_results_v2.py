"""Unregistered audit-results@2 tools bound to one trusted invocation collector.

The common completion boundary will own construction, failure cleanup and
activation. These tools have no artifact client and cannot publish results.
"""

import time
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.audit_completion_contracts import (
    ASSESSMENTS,
    IDENTIFIER,
    MAX_EVIDENCE,
    MAX_PROPOSAL_KEYS,
    MAX_SUMMARY_BYTES,
    MAX_VALUES,
    AuditEvidence,
    NormalizedAuditItem,
)
from contractor_runtime.audit_result_collector import AuditCollectionError, InvocationAuditCollector
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.audit_results import BatchResultArgument, EvidenceArgument
from contractor_runtime.toolsets.run_artifacts import ToolMetrics


def _text(value, field, key):
    try:
        valid = (
            isinstance(value, str) and value.strip() and len(value.encode()) <= MAX_SUMMARY_BYTES
        )
    except UnicodeError:
        valid = False
    if not valid:
        raise AuditCollectionError(
            field, "Provide nonempty UTF-8 text of at most 16 KiB.", item_key=key
        )
    return value


def _values(value, field, key, maximum=MAX_VALUES):
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(entry, str) or not IDENTIFIER.fullmatch(entry) for entry in value)
        or len(set(value)) != len(value)
    ):
        raise AuditCollectionError(
            field,
            f"Provide a list of at most {maximum} unique bounded identifiers.",
            item_key=key,
        )
    return tuple(sorted(value))


def _normalize(key, value):
    allowed = {"assessment", "summary", "completed", "gaps", "evidence", "proposal_keys"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise AuditCollectionError("result", "Use only documented result fields.", item_key=key)
    assessment = value.get("assessment")
    if not isinstance(assessment, str) or assessment not in ASSESSMENTS:
        raise AuditCollectionError(
            "assessment",
            "Use satisfied, violated, supported, refuted, blocked, inconclusive or not-tested.",
            item_key=key,
        )
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE:
        raise AuditCollectionError(
            "evidence", "Provide at most 256 evidence records.", item_key=key
        )
    normalized_evidence = []
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {"kind", "summary"}:
            raise AuditCollectionError(
                "evidence",
                "Each evidence record requires exactly kind and summary.",
                item_key=key,
            )
        kind = entry["kind"]
        if not isinstance(kind, str) or not IDENTIFIER.fullmatch(kind):
            raise AuditCollectionError("evidence.kind", "Use a bounded identifier.", item_key=key)
        normalized_evidence.append(
            AuditEvidence(kind, _text(entry["summary"], "evidence.summary", key))
        )
    return NormalizedAuditItem(
        key,
        assessment,
        _text(value.get("summary"), "summary", key),
        _values(value.get("completed"), "completed", key),
        _values(value.get("gaps"), "gaps", key),
        tuple(normalized_evidence),
        _values(value.get("proposal_keys", []), "proposal_keys", key, MAX_PROPOSAL_KEYS),
    )


class ReadAuditTaskTool:
    name = "read_audit_task"

    def __init__(
        self,
        collector: InvocationAuditCollector,
        task_ref: ArtifactRef,
        metrics: ToolMetrics,
    ):
        if not task_ref.revision:
            raise ValueError("audit-results@2 requires the validated exact task ref")
        self._collector = collector
        self._task_ref = task_ref.model_copy(deep=True)
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = "Read the pinned Audit tasks, requested coverage and evidence contracts."

    async def __call__(self, tool_context: ToolContext) -> dict[str, Any]:
        started = time.perf_counter_ns()
        failure = None
        try:
            result = await self._collector.read_tasks(tool_context.invocation_id)
            result["taskArtifact"] = self._task_ref.model_dump(by_alias=True)
        except AuditCollectionError as error:
            failure = error
            result = {"status": "error", "error": error.as_dict()}
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            result={"status": result.get("status", "read"), "batchSize": result.get("batchSize")},
            secrets=(),
            error=failure,
            duration_ms=(time.perf_counter_ns() - started) // 1_000_000,
        )
        return result


class SubmitCheckResultTool:
    name = "submit_check_result"

    def __init__(self, collector: InvocationAuditCollector, metrics: ToolMetrics):
        self._collector = collector
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = """Record Audit results locally; success does not publish or accept an Audit.

        Read the task first. For one item provide assessment, summary, completed and
        gaps, plus evidence and proposal_keys as needed. Supply item_key for multiple
        tasks. Corrections require expected_revision from the recorded receipt.
        Identical retries preserve revisions. Alternatively supply only results,
        a complete array in task order. Every conclusive checklist result needs
        the requested evidence kinds; completed coverage alone is insufficient.
        Errors preserve previously recorded results. Receipts give all revisions,
        acceptedCount, totalCount, missingItemKeys and complete.
        """

    async def __call__(
        self,
        tool_context: ToolContext,
        item_key: str | None = None,
        assessment: str | None = None,
        summary: str | None = None,
        completed: list[str] | None = None,
        gaps: list[str] | None = None,
        evidence: list[EvidenceArgument] | None = None,
        proposal_keys: list[str] | None = None,
        expected_revision: int | None = None,
        results: list[BatchResultArgument] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        failure = None
        try:
            self._collector.check_invocation(tool_context.invocation_id)
            keys = self._collector.owner.item_keys
            if results is not None:
                if any(
                    value is not None
                    for value in (
                        item_key,
                        assessment,
                        summary,
                        completed,
                        gaps,
                        evidence,
                        proposal_keys,
                        expected_revision,
                    )
                ):
                    raise AuditCollectionError("results", "Do not mix batch and scalar arguments.")
                if not isinstance(results, list) or len(results) != len(keys):
                    raise AuditCollectionError(
                        "results", "Provide the complete batch in task order."
                    )
                receipt = await self._collector.record_batch(
                    tuple(_normalize(key, value) for key, value in zip(keys, results, strict=True))
                )
            else:
                if item_key is None and len(keys) == 1:
                    item_key = keys[0]
                if not isinstance(item_key, str) or item_key not in keys:
                    raise AuditCollectionError(
                        "item_key", "Choose an item key from read_audit_task."
                    )
                value = {
                    "assessment": assessment,
                    "summary": summary,
                    "completed": completed,
                    "gaps": gaps,
                    "evidence": [] if evidence is None else evidence,
                    "proposal_keys": [] if proposal_keys is None else proposal_keys,
                }
                receipt = await self._collector.record(
                    _normalize(item_key, value),
                    expected_revision=expected_revision,
                )
            result = {
                "status": receipt.status,
                "revisions": [
                    {"itemKey": key, "revision": revision} for key, revision in receipt.revisions
                ],
                "acceptedCount": receipt.accepted_count,
                "totalCount": receipt.total_count,
                "missingItemKeys": list(receipt.missing_item_keys),
                "complete": receipt.complete,
            }
        except AuditCollectionError as error:
            failure = error
            result = {"status": "error", "error": error.as_dict()}
        # Neither submitted text nor raw invalid arguments enter diagnostics.
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            result=result,
            secrets=(),
            error=failure,
            duration_ms=(time.perf_counter_ns() - started) // 1_000_000,
        )
        return result
