"""One-shot ADK boundary that serializes a completed Worker result."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Protocol

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

from contractor_runtime.contracts import ResolvedModelPolicy, WorkerModelResult

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


class _OneShotFinalizerModel(BaseLlm):
    """Prevent ADK or a provider adapter from starting a repair loop."""

    _delegate: BaseLlm = PrivateAttr()
    _calls: int = PrivateAttr(default=0)
    _usage: Any | None = PrivateAttr(default=None)

    def __init__(self, delegate: BaseLlm) -> None:
        super().__init__(model=delegate.model)
        self._delegate = delegate

    @property
    def capabilities(self) -> Any:
        return self._delegate.capabilities

    @property
    def usage(self) -> Any | None:
        return self._usage

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        if self._calls != 0:
            raise ResultFinalizerFailure("call_limit_exceeded")
        self._calls = 1
        async for response in self._delegate.generate_content_async(llm_request, stream=stream):
            if not bool(getattr(response, "partial", False)):
                self._usage = getattr(response, "usage_metadata", None)
            yield response


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
        generation = types.GenerateContentConfig(max_output_tokens=self._policy.max_output_tokens)
        if self._policy.temperature is not None:
            generation.temperature = self._policy.temperature
        agent = LlmAgent(
            name="contractor_worker_result_finalizer",
            description="Serialize one already completed Worker result.",
            model=model,
            instruction=_SYSTEM_INSTRUCTION,
            tools=[],
            output_schema=WorkerModelResult,
            generate_content_config=generation,
        )
        app_name = "contractor_runtime_result_finalizer"
        user_id = "contractor_runtime"
        session_id = f"result-finalizer-{uuid.uuid4().hex}"
        service = InMemorySessionService()
        runner = Runner(
            app=App(name=app_name, root_agent=agent),
            session_service=service,
        )
        candidate: str | None = None
        call_started = False
        session_created = False
        try:
            await service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            session_created = True
            await self._observer.before_result_finalizer_call(invocation_id=invocation_id)
            call_started = True
            capture = getattr(self._observer, "capture_result_finalizer_content", None)
            if callable(capture):
                await capture(
                    invocation_id=invocation_id,
                    input={"systemInstruction": _SYSTEM_INSTRUCTION, "contents": prompt},
                )
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    invocation_id=f"{invocation_id}-result-finalizer",
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                ):
                    text = _candidate_text(event)
                    if text is not None:
                        candidate = text
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
            return candidate
        finally:
            if call_started:
                await _notify_cancelled_safely(self._observer, invocation_id)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await runner.close()
            if session_created:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await service.delete_session(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id,
                    )


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
