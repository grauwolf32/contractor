"""One-shot ADK boundary that serializes a completed Worker result."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Protocol

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from pydantic import PrivateAttr

from contractor_runtime.contracts import ResolvedModelPolicy, WorkerModelResult
from contractor_runtime.worker.one_shot import OneShotModel, generation_config, one_shot_session

MAX_RESULT_FINALIZER_INPUT_BYTES = 256 * 1024
_DOCUMENT_PREAMBLE = "Contractor Worker result finalization input (JSON):\n"
_SYSTEM_INSTRUCTION = (
    "You are a serialization boundary, not a task executor. You have no tools. "
    "Copy the supplied subtaskId and resultText exactly into the required "
    "WorkerModelResult fields subtaskId and result. Do not summarize, correct, "
    "interpret, add, remove, or reformat either value. Return only the required "
    "structured result."
)


class ResultFinalizerObserver(Protocol):
    """Account one auxiliary model call in its owning Worker invocation."""

    async def before_result_finalizer_call(self, *, invocation_id: str) -> None: ...

    async def after_result_finalizer_call(
        self, *, invocation_id: str, usage: Any | None
    ) -> None: ...

    async def on_result_finalizer_error(
        self, *, invocation_id: str, error: BaseException
    ) -> None: ...


class ResultFinalizerFailure(RuntimeError):
    """Safe, content-free result-finalizer failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Worker result finalizer failed ({code})")


class _OneShotFinalizerModel(OneShotModel):
    """Prevent ADK or a provider adapter from starting a repair loop."""

    _usage: Any | None = PrivateAttr(default=None)

    @property
    def usage(self) -> Any | None:
        return self._usage

    def _call_limit_error(self) -> Exception:
        return ResultFinalizerFailure("call_limit_exceeded")

    def _record_usage(self, usage_metadata: Any | None) -> None:
        self._usage = usage_metadata


class WorkerResultFinalizer:
    """Run one isolated, tool-free ADK agent over the Worker's model client."""

    def __init__(
        self,
        *,
        model: BaseLlm,
        policy: ResolvedModelPolicy,
        observer: ResultFinalizerObserver,
    ) -> None:
        self._delegate = model
        self._policy = policy
        self._observer = observer

    async def run(
        self,
        *,
        subtask_id: str,
        result_text: str,
        invocation_id: str,
    ) -> str | None:
        prompt = build_result_finalizer_prompt(
            subtask_id=subtask_id,
            result_text=result_text,
        )
        model = _OneShotFinalizerModel(self._delegate)
        agent = LlmAgent(
            name="contractor_worker_result_finalizer",
            description="Serialize one already completed Worker result.",
            model=model,
            instruction=_SYSTEM_INSTRUCTION,
            tools=[],
            output_schema=WorkerModelResult,
            generate_content_config=generation_config(self._policy),
        )
        async with one_shot_session(
            agent,
            app_name="contractor_runtime_result_finalizer",
            session_prefix="result-finalizer",
        ) as session:
            call_started = False
            try:
                await self._observer.before_result_finalizer_call(invocation_id=invocation_id)
                call_started = True
                capture = getattr(self._observer, "capture_result_finalizer_content", None)
                if callable(capture):
                    await capture(
                        invocation_id=invocation_id,
                        input={"systemInstruction": _SYSTEM_INSTRUCTION, "contents": prompt},
                    )
                try:
                    candidate = await session.run(
                        prompt=prompt, invocation_id=f"{invocation_id}-result-finalizer"
                    )
                except BaseException as error:
                    await self._observer.on_result_finalizer_error(
                        invocation_id=invocation_id,
                        error=error,
                    )
                    call_started = False
                    raise
                if callable(capture):
                    await capture(invocation_id=invocation_id, output=candidate)
                await self._observer.after_result_finalizer_call(
                    invocation_id=invocation_id,
                    usage=model.usage,
                )
                call_started = False
                if model.output_limited:
                    raise ResultFinalizerFailure("output_limit_exceeded")
                return candidate
            finally:
                if call_started:
                    await _notify_cancelled_safely(self._observer, invocation_id)


def build_result_finalizer_prompt(*, subtask_id: str, result_text: str) -> str:
    payload = {
        "resultText": result_text,
        "subtaskId": subtask_id,
    }
    document = _DOCUMENT_PREAMBLE + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(document.encode("utf-8")) > MAX_RESULT_FINALIZER_INPUT_BYTES:
        raise ResultFinalizerFailure("input_too_large")
    return document


async def _notify_cancelled_safely(
    observer: ResultFinalizerObserver,
    invocation_id: str,
) -> None:
    task = asyncio.create_task(
        observer.on_result_finalizer_error(
            invocation_id=invocation_id,
            error=asyncio.CancelledError(),
        ),
        name="worker-result-finalizer-observer-cleanup",
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
