"""audit-results@2 tools bound to the trusted active invocation collector."""

import time
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.audit_results.arguments import (
    AuditArgumentError,
    BatchResultArgument,
    EvidenceArgument,
    array_argument,
    identifier_list,
)
from contractor_runtime.toolsets.audit_results.collector import (
    AuditCollectionError,
    InvocationAuditCollector,
)
from contractor_runtime.toolsets.audit_results.contracts import (
    ASSESSMENTS,
    IDENTIFIER,
    MAX_EVIDENCE,
    MAX_PROPOSAL_KEYS,
    MAX_SUMMARY_BYTES,
    MAX_VALUES,
    AuditEvidence,
    NormalizedAuditItem,
)
from contractor_runtime.toolsets.audit_results.packages import MAX_BATCH_ITEMS
from contractor_runtime.toolsets.common.metrics import ToolMetrics


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
    try:
        return tuple(identifier_list(field, value, maximum=maximum))
    except AuditArgumentError as error:
        error.details["itemKey"] = key
        raise


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
    evidence = array_argument("evidence", value.get("evidence", []), MAX_EVIDENCE)
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
        collector: InvocationAuditCollector | Callable[[], InvocationAuditCollector],
        task_ref: ArtifactRef,
        metrics: ToolMetrics,
    ):
        if not task_ref.revision:
            raise ValueError("audit-results@2 requires the validated exact task ref")
        self._collector_source = collector
        self._task_ref = task_ref.model_copy(deep=True)
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = """Read the pinned Audit tasks, requested coverage and evidence contracts.

        Returns batchSize and parallel, task-ordered arrays: tasks, taskPackageIds
        and requestedCoverage. Each submitted completed array must be a verified
        subset of its matching requestedCoverage array. For one task also returns
        task and taskPackageId. Read evidence contracts from each task.
        """

    @property
    def _collector(self):
        source = self._collector_source
        return source() if callable(source) else source

    async def close(self):
        binding = getattr(self, "completion_binding", None)
        if binding is not None:
            await binding.end()

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

    def __init__(
        self,
        collector: InvocationAuditCollector | Callable[[], InvocationAuditCollector],
        metrics: ToolMetrics,
    ):
        self._collector_source = collector
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = """Record Audit results locally; success does not publish or accept an Audit.

        Read the task first. Use either individual fields or results, never both.
        Send actual JSON arrays (including []), not strings containing JSON.
        Identifier lists are sorted and deduplicated by this tool, not by you.
        Every conclusive checklist result needs the requested evidence kinds;
        completed coverage alone is insufficient. An operation with assessment
        not-tested must have empty completed coverage. Identical retries preserve
        revisions; corrections require the revision from the previous receipt.

        Args:
            item_key: Exact key from read_audit_task. May be omitted for a single
                assigned task; required when submitting one item from multiple tasks.
            assessment: satisfied, violated, supported, refuted, blocked,
                inconclusive or not-tested, subject to the task's evidence contract.
            summary: Non-empty explanation of the outcome, at most 16 KiB of UTF-8.
            completed: Array of coverage identifiers actually verified. Use only
                the matching requestedCoverage from read_audit_task, not work steps.
                Required for individual submission; [] is allowed.
            gaps: Array of short identifiers for unresolved limitations; [] is
                allowed. These need not appear in requestedCoverage. Use letters,
                digits, '.', '_', ':', '-' only; start with a letter or digit;
                at most 160 characters each. Required for individual submission.
            evidence: Optional array of objects with kind and summary only.
                Follow the assigned task's requested evidence kinds and bounds.
            proposal_keys: Optional array of client keys of finding proposals.
            expected_revision: Revision from the receipt when correcting a recorded
                item. Omit for its first submission. Not used in batch mode.
            results: Alternative batch mode: one object per task in task order,
                each with assessment, summary, completed, gaps and optional evidence
                and proposal_keys. Omit all individual fields in this mode.

        Returns:
            A receipt with revisions, acceptedCount, totalCount, missingItemKeys
            and complete, or an error identifying the invalid field and repair.
            Errors preserve recorded results. Correct the indicated arguments
            before resubmitting; do not repeat the same invalid call unchanged.
        """

    @property
    def _collector(self):
        source = self._collector_source
        return source() if callable(source) else source

    async def close(self):
        binding = getattr(self, "completion_binding", None)
        if binding is not None:
            await binding.end()

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
            collector = self._collector
            collector.check_invocation(tool_context.invocation_id)
            keys = collector.owner.item_keys
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
                results = array_argument("results", results, MAX_BATCH_ITEMS)
                if len(results) != len(keys):
                    raise AuditCollectionError(
                        "results", "Provide the complete batch in task order."
                    )
                receipt = await collector.record_batch(
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
                receipt = await collector.record(
                    _normalize(item_key, value),
                    expected_revision=expected_revision,
                )
            binding = getattr(self, "completion_binding", None)
            if binding is not None:
                binding.record_progress(receipt.accepted_count)
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
        except (AuditCollectionError, AuditArgumentError) as error:
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


class AuditResultsToolsetFactory:
    ref = "audit-results@2"
    exported_tools = frozenset({"read_audit_task", "submit_check_result"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory):
        self._client_factory = client_factory

    async def probe(self):
        return self.exported_tools if self._client_factory is not None else frozenset()

    async def create_selected(
        self,
        *,
        selected,
        allocation_id,
        namespace,
        runtime_settings,
        state,
        completion_contract=None,
        **kwargs,
    ):
        from contractor_runtime.toolsets.audit_results.completion import PreparedAuditCompletion

        if (
            completion_contract is None
            or completion_contract.result_artifact.namespace != namespace
            or set(selected) != self.exported_tools
            or self._client_factory is None
        ):
            raise ValueError("audit-results@2 requires trusted completion preparation")
        binding = await PreparedAuditCompletion.prepare(
            contract=completion_contract,
            allocation_id=allocation_id,
            client=self._client_factory(allocation_id, runtime_settings),
            timeout=runtime_settings.request_timeout_seconds,
        )
        binding.diagnostics_sink = state.metrics.record_completion
        tools = {
            "read_audit_task": ReadAuditTaskTool(
                binding.current, completion_contract.task, state.metrics
            ),
            "submit_check_result": SubmitCheckResultTool(binding.current, state.metrics),
        }
        for tool in tools.values():
            tool.completion_binding = binding
        return tools
