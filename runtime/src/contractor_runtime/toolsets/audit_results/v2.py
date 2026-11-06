"""audit-results@2: bounded judgments recorded in the active invocation."""

import asyncio
import time
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.tool_metrics import ToolCallCancelled, ToolMetrics, elapsed_ms
from contractor_runtime.toolsets.audit_results.arguments import (
    AuditArgumentError,
    BatchResultArgument,
    array_argument,
)
from contractor_runtime.toolsets.audit_results.collector import (
    AuditCollectionError,
    InvocationAuditCollector,
)
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_EVIDENCE,
    MAX_ITEMS,
    MAX_VALUES,
    NormalizedAuditItem,
)
from contractor_runtime.toolsets.common.resources import ToolResources


def _normalize(key, value):
    allowed = {
        "rationale",
        "tested_conditions",
        "limitations",
        "evidence",
        "verdict",
        "http_request_ids",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise AuditCollectionError("result", "Use only declared conclusion fields.")
    try:
        return NormalizedAuditItem(
            item_key=key,
            rationale=value.get("rationale"),
            tested_conditions=tuple(
                array_argument("tested_conditions", value.get("tested_conditions"), MAX_VALUES)
            ),
            limitations=tuple(array_argument("limitations", value.get("limitations"), MAX_VALUES)),
            evidence=tuple(array_argument("evidence", value.get("evidence", []), MAX_EVIDENCE)),
            verdict=value.get("verdict"),
            http_request_ids=tuple(
                array_argument("http_request_ids", value.get("http_request_ids", []), MAX_EVIDENCE)
            ),
        )
    except (ValueError, TypeError, UnicodeError):
        raise AuditCollectionError(
            "result", "Use bounded nonempty text, distinct arrays and the declared verdicts."
        ) from None


class ReadAuditTaskTool:
    name = "read_audit_task"

    def __init__(self, collector, context_ref: ArtifactRef, metrics: ToolMetrics):
        if not context_ref.revision:
            raise ValueError("audit-results@2 requires an exact context")
        self._collector_source, self._context_ref, self._metrics = collector, context_ref, metrics
        self.__name__ = self.name
        self.__doc__ = """Read the assigned tasks, objectives and optional hypotheses.

        Returns tasks with item_key and evidence handles with exact materials.
        Use these handles when recording evidence. A task without hypothesis is
        exploration and must omit verdict. Each assigned task needs a result.
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
        started, failure = time.perf_counter_ns(), None
        try:
            result = await self._collector.read_tasks(tool_context.invocation_id)
            result["contextArtifact"] = self._context_ref.model_dump(by_alias=True)
        except AuditCollectionError as error:
            failure, result = error, {"status": "error", "error": error.as_dict()}
        except asyncio.CancelledError:
            _record_cancelled(self._metrics, self.name, started)
            raise
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            secrets=(),
            error=failure,
            result={"status": result.get("status", "read"), "batchSize": result.get("batchSize")},
            duration_ms=elapsed_ms(started),
        )
        return result


class SubmitCheckResultTool:
    name = "submit_check_result"

    def __init__(
        self,
        collector: InvocationAuditCollector | Callable[[], InvocationAuditCollector],
        metrics: ToolMetrics,
    ):
        self._collector_source, self._metrics = collector, metrics
        self.__name__ = self.name
        self.__doc__ = """Record bounded conclusions locally for the assigned tasks.

        Read the tasks first. Supply rationale, tested_conditions, limitations,
        and evidence handles as JSON arrays, including []. For a task with a
        hypothesis supply supported, refuted or inconclusive as verdict. A
        conclusive verdict needs tested conditions and evidence; inconclusive
        needs a limitation. For exploration omit verdict. Use item_key for a
        batch member, or results for the complete batch in assignment order.
        Select actual HTTP requests with http_request_ids, an array of IDs from
        http_request in this invocation. Only selected exchanges become evidence
        Artifacts; no Finding proposal or repeat request is needed. Evicted or
        foreign IDs fail explicitly. Existing evidence handles remain supported.
        Correct a recorded result with expected_revision from its receipt.
        Returns local revisions and completeness; publication follows completion.
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
        rationale: str | None = None,
        tested_conditions: list[str] | None = None,
        limitations: list[str] | None = None,
        evidence: list[str] | None = None,
        verdict: str | None = None,
        http_request_ids: list[int] | None = None,
        expected_revision: int | None = None,
        results: list[BatchResultArgument] | None = None,
    ) -> dict[str, Any]:
        started, failure = time.perf_counter_ns(), None
        try:
            collector = self._collector
            collector.check_invocation(tool_context.invocation_id)
            keys = collector.owner.item_keys
            if results is not None:
                if any(
                    value is not None
                    for value in (
                        item_key,
                        rationale,
                        tested_conditions,
                        limitations,
                        evidence,
                        verdict,
                        expected_revision,
                        http_request_ids,
                    )
                ):
                    raise AuditCollectionError("results", "Do not mix batch and scalar fields.")
                values = array_argument("results", results, MAX_ITEMS)
                if len(values) != len(keys):
                    raise AuditCollectionError("results", "Provide the complete assigned batch.")
                receipt = await collector.record_batch(
                    tuple(_normalize(key, value) for key, value in zip(keys, values, strict=True))
                )
            else:
                if item_key is None and len(keys) == 1:
                    item_key = keys[0]
                if not isinstance(item_key, str) or item_key not in keys:
                    raise AuditCollectionError("item_key", "Choose an assigned item key.")
                receipt = await collector.record(
                    _normalize(
                        item_key,
                        {
                            "rationale": rationale,
                            "tested_conditions": tested_conditions,
                            "limitations": limitations,
                            "evidence": [] if evidence is None else evidence,
                            "verdict": verdict,
                            "http_request_ids": []
                            if http_request_ids is None
                            else http_request_ids,
                        },
                    ),
                    expected_revision=expected_revision,
                )
            binding = getattr(self, "completion_binding", None)
            if binding is not None:
                binding.record_progress(receipt.accepted_count)
            result = {
                "status": receipt.status,
                "revisions": [{"itemKey": key, "revision": rev} for key, rev in receipt.revisions],
                "acceptedCount": receipt.accepted_count,
                "totalCount": receipt.total_count,
                "missingItemKeys": list(receipt.missing_item_keys),
                "complete": receipt.complete,
            }
        except (AuditCollectionError, AuditArgumentError) as error:
            failure, result = error, {"status": "error", "error": error.as_dict()}
        except asyncio.CancelledError:
            _record_cancelled(self._metrics, self.name, started)
            raise
        self._metrics.record_tool_call(
            self.name,
            arguments={},
            result=result,
            secrets=(),
            error=failure,
            duration_ms=elapsed_ms(started),
        )
        return result


def _record_cancelled(metrics, name, started):
    metrics.record_tool_call(
        name, arguments={}, error=ToolCallCancelled(), secrets=(), duration_ms=elapsed_ms(started)
    )


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
        resources: ToolResources,
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
            captures=resources.http_captures,
        )
        binding.diagnostics_sink = resources.metrics.record_completion
        resources.proposal_collector = binding.current
        tools = {
            "read_audit_task": ReadAuditTaskTool(
                binding.current, completion_contract.context, resources.metrics
            ),
            "submit_check_result": SubmitCheckResultTool(binding.current, resources.metrics),
        }
        for tool in tools.values():
            tool.completion_binding = binding
        return tools
