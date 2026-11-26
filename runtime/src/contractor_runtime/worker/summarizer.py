"""Bounded transcript projection and one-shot terminal Worker summarizer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
from collections import Counter, deque
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import PrivateAttr

from contractor_runtime.adapters.content import capture_span_content, model_request_content
from contractor_runtime.adapters.genai import usage_attributes
from contractor_runtime.adapters.instrumentation import (
    RuntimeInstrumentation,
    end_isolated_span,
    start_isolated_span,
)
from contractor_runtime.contracts import (
    ResolvedModelPolicy,
    StageContentRequest,
    WorkerModelResult,
    WorkerObservations,
)
from contractor_runtime.llm.image_parts import image_response_metadata
from contractor_runtime.llm.one_shot import OneShotModel, generation_config, one_shot_session
from contractor_runtime.llm.usage import project_token_usage
from contractor_runtime.reasoning import (
    MAX_REASONING_TOTAL_BYTES,
    REASONING_INPUT_INSTRUCTION,
    REASONING_TAG,
    REASONING_TRUNCATED_MEMBER,
    ReasoningBudget,
    reasoning_part,
    redact_known_values,
    tagged_size,
)

MAX_SUMMARIZER_INPUT_BYTES = 512 * 1024
_MAX_PROJECTED_TEXT_BYTES = 64 * 1024
_MAX_PROJECTED_ITEMS = 128
_MAX_PROJECTED_DEPTH = 6
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|proxy[_-]?authorization|secret|token)",
    re.IGNORECASE,
)
_SYSTEM_INSTRUCTION = (
    "Return the best final result for the supplied subtask using only the bounded task data, "
    "observations, and transcript. Preserve the exact subtask ID. Do not claim unreported "
    "artifacts. Return only the required structured result."
)
_DOCUMENT_PREAMBLE = "Contractor terminal Worker summary input (JSON):\n"


@dataclass(frozen=True, slots=True)
class SummarizerUsage:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_usage_unavailable: int = 0


class SummarizerFailure(RuntimeError):
    """Safe, content-free terminal summarizer failure."""

    def __init__(self, code: str, *, retryable: bool = True) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"Worker terminal summarizer failed ({code})")


class _OneShotModel(OneShotModel):
    """Delegate exactly one model operation while retaining only numeric usage."""

    _usage: SummarizerUsage = PrivateAttr(default_factory=SummarizerUsage)
    _telemetry_usage: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def telemetry_usage(self) -> dict[str, int]:
        return dict(self._telemetry_usage)

    @property
    def usage(self) -> SummarizerUsage:
        if self._calls == 1 and self._usage.model_calls == 0:
            return SummarizerUsage(model_calls=1, token_usage_unavailable=1)
        return self._usage

    def _call_limit_error(self) -> Exception:
        return SummarizerFailure("call_limit_exceeded")

    def _record_usage(self, usage_metadata: Any | None) -> None:
        self._usage = _usage_projection(usage_metadata)
        self._telemetry_usage = usage_attributes(usage_metadata)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        try:
            async for response in super().generate_content_async(llm_request, stream=stream):
                yield response
        except Exception as error:
            # A received response rejected by the Gateway adapter spent tokens.
            if getattr(error, "response_received", False) is True:
                self._record_usage(getattr(error, "usage_metadata", None))
            raise


@dataclass(slots=True)
class _RetainedGroup:
    events: list[dict[str, Any]]
    # Bytes of the group without reasoning parts: only these count against the
    # 512 KiB retention, so reasoning never evicts task or tool evidence.
    evidence_bytes: int
    reasoning_bytes: int


class TranscriptRecorder:
    """Retain bounded, model-visible, complete normal-loop event groups only.

    With ``include_reasoning`` a model event also keeps its provider reasoning
    as tagged parts (WRK-R181). Retained reasoning is bounded separately, to
    LIM-WRK-29 in total, by dropping the reasoning of the oldest groups first.
    """

    def __init__(self, *, include_reasoning: bool, secrets: Sequence[str] = ()) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)
        self._include_reasoning = include_reasoning
        self._groups: deque[_RetainedGroup] = deque()
        self._groups_bytes = 0
        self._reasoning = ReasoningBudget()
        self._pending: list[dict[str, Any]] = []
        self._pending_reasoning_bytes = 0
        self._pending_responses: Counter[str] = Counter()
        self._truncated = False
        self._reasoning_truncated = False

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def reasoning_truncated(self) -> bool:
        """Whether available reasoning was cut or dropped before the prompt was built."""

        return self._reasoning_truncated or self._reasoning.omitted

    def record(self, event: Event) -> None:
        projected = _project_event(event, self._secrets, include_reasoning=self._include_reasoning)
        self._truncated = self._truncated or projected.truncated
        self._reasoning_truncated = self._reasoning_truncated or projected.reasoning_truncated
        if projected.event is None:
            return
        calls, responses = projected.calls, projected.responses
        projected_bytes = _evidence_size([projected.event])
        if projected_bytes > MAX_SUMMARIZER_INPUT_BYTES:
            self._discard_pending()
            self._truncated = True
            return
        if calls and responses:
            self._discard_pending()
            self._truncated = True
            return
        if calls:
            if self._pending:
                self._discard_pending()
                self._truncated = True
            self._pending = [projected.event]
            self._pending_reasoning_bytes = projected.reasoning_bytes
            self._pending_responses = Counter(calls)
            return
        if responses and self._pending:
            observed = Counter(responses)
            if any(observed[key] > self._pending_responses[key] for key in observed):
                self._discard_pending()
                self._truncated = True
                return
            self._pending.append(projected.event)
            self._pending_reasoning_bytes += projected.reasoning_bytes
            self._pending_responses.subtract(observed)
            self._pending_responses += Counter()
            if not self._pending_responses:
                if _evidence_size(self._pending) <= MAX_SUMMARIZER_INPUT_BYTES:
                    self._append_group(self._pending, self._pending_reasoning_bytes)
                else:
                    self._truncated = True
                self._discard_pending()
            return
        if responses:
            # A tool response without its selecting model event is not a
            # complete model-visible group and cannot enter the summary.
            self._truncated = True
            return
        if self._pending:
            # A new unrelated complete event means the preceding call/response
            # group cannot be proven complete and is excluded closed.
            self._discard_pending()
            self._truncated = True
        self._append_group([projected.event], projected.reasoning_bytes)

    def finish(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Retained groups in order; reasoning parts are tagged ``reasoning``."""

        if self._pending:
            self._discard_pending()
            self._truncated = True
        return tuple(tuple(group.events) for group in self._groups)

    def _append_group(self, group: list[dict[str, Any]], reasoning_bytes: int) -> None:
        stored = list(group)
        evidence_bytes = _evidence_size(stored)
        if evidence_bytes > MAX_SUMMARIZER_INPUT_BYTES:
            self._truncated = True
            return
        if reasoning_bytes > MAX_REASONING_TOTAL_BYTES:
            # One model event is bounded to LIM-WRK-29 when it is projected,
            # so this is unreachable; drop the reasoning closed if it is not.
            stored = _without_reasoning(stored)
            reasoning_bytes = 0
            self._reasoning_truncated = True
            if not stored:
                return
        while reasoning_bytes and not self._reasoning.fits(reasoning_bytes):
            self._drop_oldest_reasoning()
        if reasoning_bytes:
            self._reasoning.admit(reasoning_bytes)
        self._groups.append(_RetainedGroup(stored, evidence_bytes, reasoning_bytes))
        self._groups_bytes += evidence_bytes
        while self._groups_bytes > MAX_SUMMARIZER_INPUT_BYTES and self._groups:
            removed = self._groups.popleft()
            self._groups_bytes -= removed.evidence_bytes
            if removed.reasoning_bytes:
                self._reasoning.release(removed.reasoning_bytes)
            self._truncated = True

    def _drop_oldest_reasoning(self) -> None:
        # Groups are bounded by the Worker's model-call limit (LIM-TPL-3), so
        # this scan stays small; it runs only while reasoning exceeds its bound.
        for index, group in enumerate(self._groups):
            if not group.reasoning_bytes:
                continue
            self._reasoning.release(group.reasoning_bytes)
            group.reasoning_bytes = 0
            group.events = _without_reasoning(group.events)
            if not group.events:
                # A reasoning-only group has no evidence left to retain.
                self._groups_bytes -= group.evidence_bytes
                del self._groups[index]
            return
        raise RuntimeError("retained reasoning has no group to release")

    def _discard_pending(self) -> None:
        self._pending.clear()
        self._pending_reasoning_bytes = 0
        self._pending_responses.clear()


class TerminalSummarizer:
    """One ephemeral, tool-free ADK agent over a separately constructed model."""

    def __init__(
        self,
        *,
        model: BaseLlm,
        policy: ResolvedModelPolicy,
        instructions: str | None = None,
        include_reasoning: bool,
        instrumentation: RuntimeInstrumentation | None = None,
        check_admission: Callable[[], None] | None = None,
    ) -> None:
        self._policy = policy
        self._instructions = _SYSTEM_INSTRUCTION if instructions is None else instructions
        if include_reasoning:
            # Runtime-owned and fixed, so reasoning stays a hypothesis even when
            # configured instructions do not mention it (WRK-R183).
            self._instructions += "\n\n" + REASONING_INPUT_INSTRUCTION
        self._model = _OneShotModel(model)
        self._instrumentation = instrumentation
        self._check_admission = check_admission

    @property
    def usage(self) -> SummarizerUsage:
        return self._model.usage

    async def run(self, *, prompt: str, invocation_id: str) -> str | None:
        span = None

        def before_model(callback_context: Any, llm_request: LlmRequest) -> None:
            nonlocal span
            if self._check_admission is not None:
                self._check_admission()
            span = start_isolated_span(
                self._instrumentation,
                "contractor.worker.model",
                {"operation.kind": "model", "model.alias": self._policy.model},
            )
            with contextlib.suppress(Exception):
                capture_span_content(span, input=lambda: model_request_content(llm_request))

        def after_model(callback_context: Any, llm_response: LlmResponse) -> None:
            capture_span_content(span, output=lambda: llm_response.content)

        agent = LlmAgent(
            name="contractor_terminal_summarizer",
            description="Produce one terminal result from bounded Worker history.",
            model=self._model,
            # A provider preserves literal braces in configured Markdown/JSON
            # instead of treating them as ADK session-state substitutions
            # (TPL-R51, WRK-R166).
            instruction=lambda _: self._instructions,
            tools=[],
            output_schema=WorkerModelResult,
            generate_content_config=generation_config(self._policy),
            before_model_callback=before_model,
            after_model_callback=after_model,
        )
        # The caller owns the isolated summary model and closes it, which
        # erases its Gateway credential; the session only owns ADK state.
        async with one_shot_session(
            agent, app_name="contractor_runtime_summarizer", session_prefix="summary"
        ) as session:
            outcome = "failed"
            try:
                try:
                    candidate = await session.run(
                        prompt=prompt, invocation_id=f"{invocation_id}-summary"
                    )
                except asyncio.CancelledError:
                    raise
                except SummarizerFailure:
                    raise
                except Exception as error:
                    # SDK workflow wrappers may carry a deadline exception.
                    # Preserve the owning invocation's expired/cancelled result
                    # instead of turning it into a generic summarizer failure.
                    if self._check_admission is not None:
                        self._check_admission()
                    code = (
                        "gateway_unavailable"
                        if isinstance(getattr(error, "provider_error_type", None), str)
                        else "execution_failed"
                    )
                    raise SummarizerFailure(
                        code, retryable=getattr(error, "retryable", True)
                    ) from None
                if self._model.output_limited:
                    raise SummarizerFailure("output_limit_exceeded")
                usage = self.usage
                if (
                    self._policy.max_total_tokens is not None
                    and usage.total_tokens > self._policy.max_total_tokens
                ):
                    raise SummarizerFailure("budget_exhausted")
                outcome = "succeeded"
                return candidate
            finally:
                end_isolated_span(span, outcome=outcome, attributes=self._model.telemetry_usage)


def build_summarizer_prompt(
    request: StageContentRequest,
    observations: WorkerObservations,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    transcript_truncated: bool,
    include_reasoning: bool,
    reasoning_truncated: bool = False,
    secrets: Sequence[str] = (),
) -> str:
    """Build a deterministic UTF-8 document no larger than 512 KiB.

    Task data, observations and the newest complete groups are selected
    first. Only then, with ``include_reasoning``, is tagged reasoning added to
    selected groups, newest first, within LIM-WRK-29 and the space left, so it
    never displaces task or tool evidence (WRK-R182).
    """

    protected_values = tuple(secret for secret in secrets if secret)
    task = {
        "subtaskId": _redact_known_values(request.subtask_id, protected_values),
        "objective": _redact_known_values(request.objective, protected_values),
        "instructions": _redact_known_values(request.instructions, protected_values),
        "parameters": {
            name: _redact_known_values(value, protected_values)
            for name, value in sorted(request.parameters.items())
        },
        "artifacts": {
            _redact_known_values(name, protected_values): {
                key: _redact_known_values(value, protected_values)
                for key, value in ref.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ).items()
            }
            for name, ref in sorted(request.artifacts.items())
        },
    }
    observation_value = observations.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload: dict[str, Any] = {
        "task": task,
        "observations": observation_value,
        "transcript": [],
        # false is one byte larger than true, so sizing against it remains safe
        # when the final projection needs to report truncation.
        "transcriptTruncated": False,
    }
    if include_reasoning:
        payload[REASONING_TRUNCATED_MEMBER] = False
    if _document_size(payload) > MAX_SUMMARIZER_INPUT_BYTES:
        workspace = observation_value.get("workspace")
        if isinstance(workspace, dict):
            workspace["filesRead"] = []
            workspace["filesReadTruncated"] = True
        observation_value["truncated"] = True
    if _document_size(payload) > MAX_SUMMARIZER_INPUT_BYTES:
        payload["observations"] = {
            "profile": observations.profile,
            "tools": {},
            "workspace": None,
            "truncated": True,
        }
    if _document_size(payload) > MAX_SUMMARIZER_INPUT_BYTES:
        raise SummarizerFailure("input_too_large")

    base_size = _document_size(payload)
    evidence = [_without_reasoning(group) for group in groups]
    evidence_sizes = [_json_size(group) if group else 0 for group in evidence]
    # Index -> the form of that group entering the transcript.
    chosen: dict[int, Sequence[Mapping[str, Any]]] = {}
    selected_size = 0
    omitted = bool(transcript_truncated)
    for index in reversed(range(len(groups))):
        if not evidence[index]:
            continue  # reasoning only; considered with the other reasoning below
        separator_size = 1 if chosen else 0
        group_size = evidence_sizes[index]
        if base_size + selected_size + separator_size + group_size <= MAX_SUMMARIZER_INPUT_BYTES:
            chosen[index] = evidence[index]
            selected_size += separator_size + group_size
        else:
            omitted = True
    reasoning_omitted = bool(reasoning_truncated)
    if include_reasoning:
        budget = ReasoningBudget()
        for index in reversed(range(len(groups))):
            cost = _reasoning_size(groups[index])
            if not cost:
                continue
            if evidence[index] and index not in chosen:
                reasoning_omitted = True  # its evidence did not fit
                continue
            full_size = _json_size(groups[index])
            if index in chosen:
                added = full_size - evidence_sizes[index]
            else:
                added = full_size + (1 if chosen else 0)
            if base_size + selected_size + added <= MAX_SUMMARIZER_INPUT_BYTES and budget.admit(
                cost
            ):
                chosen[index] = groups[index]
                selected_size += added
            else:
                reasoning_omitted = True
        reasoning_omitted = reasoning_omitted or budget.omitted
        payload[REASONING_TRUNCATED_MEMBER] = reasoning_omitted
    selected = [chosen[index] for index in sorted(chosen)]
    payload["transcript"] = selected
    payload["transcriptTruncated"] = omitted
    document = _DOCUMENT_PREAMBLE + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(document.encode("utf-8")) > MAX_SUMMARIZER_INPUT_BYTES:
        raise SummarizerFailure("input_too_large")
    return document


@dataclass(frozen=True, slots=True)
class _ProjectedEvent:
    event: dict[str, Any] | None = None
    calls: tuple[str, ...] = ()
    responses: tuple[str, ...] = ()
    truncated: bool = False
    reasoning_bytes: int = 0
    reasoning_truncated: bool = False


def _project_event(
    event: Event, secrets: tuple[str, ...], *, include_reasoning: bool
) -> _ProjectedEvent:
    if bool(getattr(event, "partial", False)) or event.content is None:
        return _ProjectedEvent()
    role = event.content.role
    if role not in {"model", "user"}:
        return _ProjectedEvent()
    source_parts = event.content.parts or []
    if len(source_parts) > _MAX_PROJECTED_ITEMS:
        return _ProjectedEvent(truncated=True)
    parts: list[dict[str, Any]] = []
    calls: list[str] = []
    responses: list[str] = []
    truncated = False
    reasoning = ReasoningBudget()
    reasoning_truncated = False
    for part in source_parts:
        if bool(getattr(part, "thought", False)):
            # Only provider reasoning text of a model turn, and only when the
            # consumer enables it; a thought signature alone is opaque.
            if not include_reasoning or role != "model" or part.text is None:
                continue
            projected_reasoning = reasoning_part(part.text, secrets)
            if projected_reasoning is None:
                continue
            reasoning_truncated = reasoning_truncated or projected_reasoning.truncated
            if reasoning.admit(projected_reasoning.size):
                parts.append(projected_reasoning.tagged())
            continue
        if part.text is not None:
            value, changed = _bound_text(part.text, secrets)
            parts.append({"text": value})
            truncated = truncated or changed
            continue
        if part.function_call is not None:
            call = part.function_call
            value, changed = _safe_json_value(call.args or {}, secrets=secrets)
            parts.append({"functionCall": {"name": str(call.name or ""), "args": value}})
            calls.append(_tool_pair_key(call.id, call.name))
            truncated = truncated or changed
            continue
        if part.function_response is not None:
            response = part.function_response
            value, changed = _safe_json_value(
                image_response_metadata(response.response or {}, attached=False), secrets=secrets
            )
            parts.append(
                {"functionResponse": {"name": str(response.name or ""), "response": value}}
            )
            responses.append(_tool_pair_key(response.id, response.name))
            truncated = truncated or changed
    reasoning_truncated = reasoning_truncated or reasoning.omitted
    if not parts:
        return _ProjectedEvent(truncated=truncated, reasoning_truncated=reasoning_truncated)
    return _ProjectedEvent(
        {"role": role, "parts": parts},
        tuple(calls),
        tuple(responses),
        truncated,
        reasoning.used,
        reasoning_truncated,
    )


def _is_reasoning(part: Mapping[str, Any]) -> bool:
    return REASONING_TAG in part


def _without_reasoning(group: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The group without reasoning parts; events left with no part are dropped."""

    result: list[dict[str, Any]] = []
    for event in group:
        parts = [part for part in event["parts"] if not _is_reasoning(part)]
        if parts:
            result.append({"role": event["role"], "parts": parts})
    return result


def _reasoning_size(group: Sequence[Mapping[str, Any]]) -> int:
    """Serialized bytes of the group's tagged reasoning parts (LIM-WRK-29)."""

    return sum(
        tagged_size(part[REASONING_TAG])
        for event in group
        for part in event["parts"]
        if _is_reasoning(part)
    )


def _evidence_size(group: Sequence[Mapping[str, Any]]) -> int:
    return _json_size(_without_reasoning(group))


def _tool_pair_key(identifier: Any, name: Any) -> str:
    value = str(identifier or "")
    if value:
        return "id:" + value[:256]
    return "name:" + str(name or "")[:256]


def _safe_json_value(
    value: Any,
    *,
    secrets: tuple[str, ...],
    key: str | None = None,
    depth: int = 0,
) -> tuple[Any, bool]:
    if key is not None and _SENSITIVE_KEY.search(key):
        return "[REDACTED]", True
    if depth >= _MAX_PROJECTED_DEPTH:
        return "[TRUNCATED]", True
    if value is None or isinstance(value, bool | int):
        return value, False
    if isinstance(value, float):
        return (value, False) if math.isfinite(value) else ("[NON_FINITE]", True)
    if isinstance(value, str):
        return _bound_text(value, secrets)
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        truncated = False
        items = sorted(((str(item_key), item_value) for item_key, item_value in value.items()))
        for item_key, item_value in items[:_MAX_PROJECTED_ITEMS]:
            projected_value, changed = _safe_json_value(
                item_value,
                secrets=secrets,
                key=item_key,
                depth=depth + 1,
            )
            projected[item_key] = projected_value
            truncated = truncated or changed
        return projected, truncated or len(items) > _MAX_PROJECTED_ITEMS
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
        projected_items: list[Any] = []
        truncated = len(value) > _MAX_PROJECTED_ITEMS
        for item in value[:_MAX_PROJECTED_ITEMS]:
            projected, changed = _safe_json_value(item, secrets=secrets, depth=depth + 1)
            projected_items.append(projected)
            truncated = truncated or changed
        return projected_items, truncated
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _safe_json_value(model_dump(mode="json"), secrets=secrets, depth=depth)
        except Exception:
            return "[OMITTED]", True
    return "[OMITTED]", True


def _bound_text(value: str, secrets: tuple[str, ...]) -> tuple[str, bool]:
    projected = value
    changed = False
    for secret in secrets:
        if secret in projected:
            projected = projected.replace(secret, "[REDACTED]")
            changed = True
    encoded = projected.encode("utf-8")
    if len(encoded) <= _MAX_PROJECTED_TEXT_BYTES:
        return projected, changed
    prefix = encoded[:_MAX_PROJECTED_TEXT_BYTES]
    while prefix:
        try:
            return prefix.decode("utf-8") + "[TRUNCATED]", True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return "[TRUNCATED]", True


def _redact_known_values(value: str, secrets: tuple[str, ...]) -> str:
    return redact_known_values(value, secrets)


def _usage_projection(usage: Any | None) -> SummarizerUsage:
    projected = project_token_usage(usage)
    return SummarizerUsage(
        model_calls=1,
        input_tokens=projected.prompt_tokens or 0,
        output_tokens=projected.output_tokens or 0,
        total_tokens=projected.total_tokens or 0,
        token_usage_unavailable=int(projected.total_unavailable),
    )


def _document_size(payload: Mapping[str, Any]) -> int:
    return len(_DOCUMENT_PREAMBLE.encode("utf-8")) + _json_size(payload)


def _json_size(value: Any) -> int:
    # Deliberately strict, unlike the telemetry measure: a value the summary
    # document cannot encode (NaN, an unsupported type, mixed key types or a
    # lone surrogate) raises instead of being counted.
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
