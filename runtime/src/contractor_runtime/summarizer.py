"""Bounded transcript projection and one-shot terminal Worker summarizer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import uuid
from collections import Counter, deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import PrivateAttr

from contractor_runtime.adapters.content import capture_span_content, model_request_content
from contractor_runtime.adapters.instrumentation import RuntimeInstrumentation
from contractor_runtime.contracts import (
    ResolvedModelPolicy,
    StageContentRequest,
    WorkerModelResult,
    WorkerObservations,
)
from contractor_runtime.token_usage import project_token_usage

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

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Worker terminal summarizer failed ({code})")


class _OneShotModel(BaseLlm):
    """Delegate exactly one model operation while retaining only numeric usage."""

    _delegate: BaseLlm = PrivateAttr()
    _calls: int = PrivateAttr(default=0)
    _usage: SummarizerUsage = PrivateAttr(default_factory=SummarizerUsage)

    def __init__(self, delegate: BaseLlm) -> None:
        super().__init__(model=delegate.model)
        self._delegate = delegate

    @property
    def capabilities(self) -> Any:
        return self._delegate.capabilities

    @property
    def usage(self) -> SummarizerUsage:
        if self._calls == 1 and self._usage.model_calls == 0:
            return SummarizerUsage(model_calls=1, token_usage_unavailable=1)
        return self._usage

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        if self._calls != 0:
            raise SummarizerFailure("call_limit_exceeded")
        self._calls = 1
        completed_usage: Any | None = None
        async for response in self._delegate.generate_content_async(llm_request, stream=stream):
            if not bool(getattr(response, "partial", False)):
                completed_usage = getattr(response, "usage_metadata", None)
                self._usage = _usage_projection(completed_usage)
            yield response


class TranscriptRecorder:
    """Retain bounded, model-visible, complete normal-loop event groups only."""

    def __init__(self, *, secrets: Sequence[str] = ()) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)
        self._groups: deque[tuple[list[dict[str, Any]], int]] = deque()
        self._groups_bytes = 0
        self._pending: list[dict[str, Any]] = []
        self._pending_responses: Counter[str] = Counter()
        self._truncated = False

    @property
    def truncated(self) -> bool:
        return self._truncated

    def record(self, event: Event) -> None:
        projected, calls, responses, truncated = _project_event(event, self._secrets)
        self._truncated = self._truncated or truncated
        if projected is None:
            return
        projected_bytes = _json_size(projected)
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
            self._pending = [projected]
            self._pending_responses = Counter(calls)
            return
        if responses and self._pending:
            observed = Counter(responses)
            if any(observed[key] > self._pending_responses[key] for key in observed):
                self._discard_pending()
                self._truncated = True
                return
            self._pending.append(projected)
            self._pending_responses.subtract(observed)
            self._pending_responses += Counter()
            if not self._pending_responses:
                if _json_size(self._pending) <= MAX_SUMMARIZER_INPUT_BYTES:
                    self._append_group(self._pending)
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
        self._append_group([projected])

    def finish(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        if self._pending:
            self._discard_pending()
            self._truncated = True
        return tuple(tuple(event for event in group) for group, _ in self._groups)

    def _append_group(self, group: list[dict[str, Any]]) -> None:
        stored = list(group)
        group_bytes = _json_size(stored)
        if group_bytes > MAX_SUMMARIZER_INPUT_BYTES:
            self._truncated = True
            return
        self._groups.append((stored, group_bytes))
        self._groups_bytes += group_bytes
        while self._groups_bytes > MAX_SUMMARIZER_INPUT_BYTES and self._groups:
            _, removed_bytes = self._groups.popleft()
            self._groups_bytes -= removed_bytes
            self._truncated = True

    def _discard_pending(self) -> None:
        self._pending.clear()
        self._pending_responses.clear()


class TerminalSummarizer:
    """One ephemeral, tool-free ADK agent over a separately constructed model."""

    def __init__(
        self,
        *,
        model: BaseLlm,
        policy: ResolvedModelPolicy,
        instrumentation: RuntimeInstrumentation | None = None,
    ) -> None:
        self._delegate = model
        self._policy = policy
        self._model = _OneShotModel(model)
        self._instrumentation = instrumentation

    @property
    def usage(self) -> SummarizerUsage:
        return self._model.usage

    async def run(self, *, prompt: str, invocation_id: str) -> str | None:
        span = None

        def before_model(callback_context: Any, llm_request: LlmRequest) -> None:
            nonlocal span
            with contextlib.suppress(Exception):
                if self._instrumentation is not None:
                    span = self._instrumentation.start_span(
                        "contractor.worker.model",
                        attributes={"operation.kind": "model", "model.alias": self._policy.model},
                    )
                    capture_span_content(span, input=lambda: model_request_content(llm_request))

        def after_model(callback_context: Any, llm_response: LlmResponse) -> None:
            capture_span_content(span, output=lambda: llm_response.content)

        generation = types.GenerateContentConfig(max_output_tokens=self._policy.max_output_tokens)
        if self._policy.temperature is not None:
            generation.temperature = self._policy.temperature
        agent = LlmAgent(
            name="contractor_terminal_summarizer",
            description="Produce one terminal result from bounded Worker history.",
            model=self._model,
            instruction=_SYSTEM_INSTRUCTION,
            tools=[],
            output_schema=WorkerModelResult,
            generate_content_config=generation,
            before_model_callback=before_model,
            after_model_callback=after_model,
        )
        app_name = "contractor_runtime_summarizer"
        user_id = "contractor_runtime"
        session_id = f"summary-{uuid.uuid4().hex}"
        service = InMemorySessionService()
        runner = Runner(
            app=App(name=app_name, root_agent=agent),
            session_service=service,
        )
        await service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        candidate: str | None = None
        outcome = "failed"
        try:
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    invocation_id=f"{invocation_id}-summary",
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                ):
                    text = _candidate_text(event)
                    if text is not None:
                        candidate = text
            except asyncio.CancelledError:
                raise
            except SummarizerFailure:
                raise
            except Exception as error:
                code = (
                    "gateway_unavailable"
                    if isinstance(getattr(error, "provider_error_type", None), str)
                    else "execution_failed"
                )
                raise SummarizerFailure(code) from None
            usage = self.usage
            if (
                self._policy.max_total_tokens is not None
                and usage.total_tokens > self._policy.max_total_tokens
            ):
                raise SummarizerFailure("budget_exhausted")
            outcome = "succeeded"
            return candidate
        finally:
            with contextlib.suppress(Exception):
                if span is not None:
                    span.end(outcome=outcome)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await runner.close()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await service.delete_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            clear = getattr(self._delegate, "clear_credentials", None)
            if callable(clear):
                clear()


def build_summarizer_prompt(
    request: StageContentRequest,
    observations: WorkerObservations,
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    transcript_truncated: bool,
    secrets: Sequence[str] = (),
) -> str:
    """Build a deterministic UTF-8 document no larger than 512 KiB."""

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
    selected_newest_first: list[Sequence[Mapping[str, Any]]] = []
    selected_size = 0
    omitted = bool(transcript_truncated)
    for group in reversed(groups):
        group_size = _json_size(group)
        separator_size = 1 if selected_newest_first else 0
        if base_size + selected_size + separator_size + group_size <= MAX_SUMMARIZER_INPUT_BYTES:
            selected_newest_first.append(group)
            selected_size += separator_size + group_size
        else:
            omitted = True
    selected = list(reversed(selected_newest_first))
    payload["transcript"] = selected
    payload["transcriptTruncated"] = omitted or len(selected) != len(groups)
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


def _project_event(
    event: Event, secrets: tuple[str, ...]
) -> tuple[dict[str, Any] | None, tuple[str, ...], tuple[str, ...], bool]:
    if bool(getattr(event, "partial", False)) or event.content is None:
        return None, (), (), False
    role = event.content.role
    if role not in {"model", "user"}:
        return None, (), (), False
    source_parts = event.content.parts or []
    if len(source_parts) > _MAX_PROJECTED_ITEMS:
        return None, (), (), True
    parts: list[dict[str, Any]] = []
    calls: list[str] = []
    responses: list[str] = []
    truncated = False
    for part in source_parts:
        if bool(getattr(part, "thought", False)):
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
            value, changed = _safe_json_value(response.response or {}, secrets=secrets)
            parts.append(
                {"functionResponse": {"name": str(response.name or ""), "response": value}}
            )
            responses.append(_tool_pair_key(response.id, response.name))
            truncated = truncated or changed
    if not parts:
        return None, (), (), truncated
    return {"role": role, "parts": parts}, tuple(calls), tuple(responses), truncated


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
    projected = value
    for secret in secrets:
        projected = projected.replace(secret, "[REDACTED]")
    return projected


def _usage_projection(usage: Any | None) -> SummarizerUsage:
    projected = project_token_usage(usage)
    return SummarizerUsage(
        model_calls=1,
        input_tokens=projected.prompt_tokens or 0,
        output_tokens=projected.output_tokens or 0,
        total_tokens=projected.total_tokens or 0,
        token_usage_unavailable=int(projected.total_unavailable),
    )


def _candidate_text(event: Event) -> str | None:
    content = event.content
    if content is None or content.role != "model" or event.partial:
        return None
    text: list[str] = []
    for part in content.parts or []:
        if bool(getattr(part, "thought", False)):
            continue
        if part.text is None:
            return None
        text.append(part.text)
    return "".join(text) if text else None


def _document_size(payload: Mapping[str, Any]) -> int:
    return len(_DOCUMENT_PREAMBLE.encode("utf-8")) + len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
