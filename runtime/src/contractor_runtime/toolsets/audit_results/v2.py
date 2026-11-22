"""audit-results@2: bounded judgments recorded in the active invocation."""

import asyncio
import time
from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.tool_metrics import ToolCallCancelled, ToolMetrics, elapsed_ms
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.audit_results.arguments import (
    AuditArgumentError,
    BatchResultArgument,
    admit_arguments,
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
    AuditVerdict,
    NormalizedAuditItem,
)
from contractor_runtime.toolsets.common.argument_diagnostics import NormalizationNotices
from contractor_runtime.toolsets.common.resources import ToolResources


def _normalize(key, value, notices, path=""):
    if not isinstance(key, str) or not key:
        raise AuditCollectionError("item_key", "Every conclusion requires its assigned item_key.")
    allowed = {
        "item_key",
        "rationale",
        "tested_conditions",
        "limitations",
        "evidence",
        "verdict",
        "http_request_ids",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise AuditArgumentError(
            path.rstrip(".") or "result",
            "Use only declared conclusion fields.",
            expected_type="object",
            actual_type="object",
        )
    try:
        return NormalizedAuditItem(
            item_key=key,
            rationale=value.get("rationale"),
            tested_conditions=tuple(
                array_argument(
                    path + "tested_conditions", value.get("tested_conditions"), MAX_VALUES, notices
                )
            ),
            limitations=tuple(
                array_argument(path + "limitations", value.get("limitations"), MAX_VALUES, notices)
            ),
            evidence=tuple(
                array_argument(path + "evidence", value.get("evidence", []), MAX_EVIDENCE, notices)
            ),
            verdict=value.get("verdict"),
            http_request_ids=tuple(
                array_argument(
                    path + "http_request_ids",
                    value.get("http_request_ids", []),
                    MAX_EVIDENCE,
                    notices,
                    int,
                )
            ),
        )
    except AuditArgumentError:
        raise
    except (ValueError, TypeError, UnicodeError):
        raise AuditCollectionError(
            "result", "Use bounded nonempty text, distinct arrays and the declared verdicts."
        ) from None


def _prepare_single(value):
    admit_arguments(value)
    notices = NormalizationNotices()
    return _normalize(value["item_key"], value, notices), notices


def _prepare_batch(items):
    admit_arguments({"items": items})
    notices = NormalizationNotices()
    values = tuple(
        _normalize(value.get("item_key"), value, notices, f"items[{index}].")
        for index, value in enumerate(array_argument("items", items, MAX_ITEMS, notices, dict))
    )
    return values, notices


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
        self.__doc__ = self.description

    description = """Record one complete conclusion for an assigned item.

    Read read_audit_task for item keys and evidence handles. For a hypothesis,
    verdict is supported, refuted or inconclusive. Supported/refuted need tested
    conditions and evidence. Inconclusive needs a limitation. Omit verdict for
    exploration. A later submission with new content replaces this item; identical content replays.

    Args:
        item_key: Assigned item key, including for a single-item assignment.
        rationale: Bounded explanation of the conclusion.
        tested_conditions: Conditions examined as a native or once-serialized JSON
            array, including [].
        limitations: Remaining limitations, as a native or once-serialized JSON array, including [].
        evidence: Selected evidence handles from this item's assignment; defaults to [].
        verdict: Hypothesis conclusion; omit for exploration.
        http_request_ids: Local IDs from http_request in this invocation; defaults
            to []. Runtime attaches actual exchanges without replaying requests.

    All array arguments also accept one serialized JSON array. Nested values
    retain their types; request IDs must be integers, never strings or booleans.

    Returns:
        Local revisions, missing item keys, completeness and any normalization notice.
        Runtime publishes the accepted conclusions after Worker completion.
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
        item_key: str,
        rationale: str,
        tested_conditions: list[str] | str,
        limitations: list[str] | str,
        evidence: list[str] | str | None = None,
        verdict: AuditVerdict | None = None,
        http_request_ids: list[int] | str | None = None,
    ) -> dict[str, Any]:
        base = self._collector.item_base(item_key)
        return await self._submit(
            tool_context,
            partial(
                _prepare_single,
                {
                    "item_key": item_key,
                    "rationale": rationale,
                    "tested_conditions": tested_conditions,
                    "limitations": limitations,
                    "evidence": [] if evidence is None else evidence,
                    "verdict": verdict,
                    "http_request_ids": [] if http_request_ids is None else http_request_ids,
                },
            ),
            lambda collector, item: collector.record(item, expected_revision=base),
        )

    async def _submit(self, tool_context, prepare, submit) -> dict[str, Any]:
        started, failure = time.perf_counter_ns(), None
        try:
            collector = self._collector
            collector.check_invocation(tool_context.invocation_id)
            values, notices = await to_thread_until_done(prepare, name="audit-tool-arguments")
            receipt = await submit(collector, values)
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
            notices.attach(result)
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


class SubmitCheckResultsTool(SubmitCheckResultTool):
    name = "submit_check_results"
    description = """Record the complete assigned batch, with an item_key on each conclusion.

    Items may arrive in any order. Each assigned key must occur exactly once.
    Validation and acceptance apply to the whole batch. Identical existing
    conclusions are replays. Correct a changed conclusion with submit_check_result
    alone. Field and evidence rules match the single-item tool.

    Args:
        items: Complete JSON array of conclusions, native or serialized once.
            Declared arrays inside each conclusion also accept one serialized array.
            Each requires item_key,
            rationale, tested_conditions and limitations. Optional evidence and
            http_request_ids default to []. Supply verdict only for a hypothesis.

    Returns:
        Local revisions in assignment order, missing keys, completeness and any
        normalization notice.
    """

    async def __call__(
        self, tool_context: ToolContext, items: list[BatchResultArgument] | str
    ) -> dict[str, Any]:
        return await self._submit(
            tool_context,
            partial(_prepare_batch, items),
            lambda collector, values: collector.record_batch(values),
        )


def _record_cancelled(metrics, name, started):
    metrics.record_tool_call(
        name, arguments={}, error=ToolCallCancelled(), secrets=(), duration_ms=elapsed_ms(started)
    )


class AuditResultsToolsetFactory:
    ref = "audit-results@2"
    exported_tools = frozenset({"read_audit_task", "submit_check_result", "submit_check_results"})
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
            "submit_check_results": SubmitCheckResultsTool(binding.current, resources.metrics),
        }
        for tool in tools.values():
            tool.completion_binding = binding
        return tools
