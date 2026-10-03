"""ADK scaffolding shared by the Worker and its one-shot auxiliary agents."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
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

from contractor_runtime.contracts import ResolvedModelPolicy
from contractor_runtime.llm.response import output_limit_reached

_USER_ID = "contractor_runtime"


def generation_config(policy: ResolvedModelPolicy) -> types.GenerateContentConfig:
    generation = types.GenerateContentConfig(max_output_tokens=policy.max_output_tokens)
    if policy.temperature is not None:
        generation.temperature = policy.temperature
    return generation


def candidate_text(event: Event) -> str | None:
    """Return the complete, non-thought model text of an event, if any."""

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


class OneShotModel(BaseLlm):
    """Delegate exactly one model operation so ADK cannot start a repair loop."""

    _delegate: BaseLlm = PrivateAttr()
    _calls: int = PrivateAttr(default=0)
    _output_limited: bool = PrivateAttr(default=False)

    def __init__(self, delegate: BaseLlm) -> None:
        super().__init__(model=delegate.model)
        self._delegate = delegate

    @property
    def capabilities(self) -> Any:
        return self._delegate.capabilities

    @property
    def output_limited(self) -> bool:
        return self._output_limited

    def _call_limit_error(self) -> Exception:
        raise NotImplementedError

    def _record_usage(self, usage_metadata: Any | None) -> None:
        raise NotImplementedError

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        if self._calls != 0:
            raise self._call_limit_error()
        self._calls = 1
        async for response in self._delegate.generate_content_async(llm_request, stream=stream):
            if not bool(getattr(response, "partial", False)):
                self._record_usage(getattr(response, "usage_metadata", None))
            if output_limit_reached(response):
                self._output_limited = True
                # ADK must never parse or repair truncated structured text.
                return
            yield response


class OneShotSession:
    """One isolated in-memory ADK session around a single tool-free agent."""

    def __init__(self, runner: Runner, session_id: str) -> None:
        self._runner = runner
        self._session_id = session_id

    async def run(self, *, prompt: str, invocation_id: str) -> str | None:
        """Send one prompt and return the last complete model text."""

        candidate: str | None = None
        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=self._session_id,
            invocation_id=invocation_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            text = candidate_text(event)
            if text is not None:
                candidate = text
        return candidate


@contextlib.asynccontextmanager
async def one_shot_session(
    agent: LlmAgent, *, app_name: str, session_prefix: str
) -> AsyncIterator[OneShotSession]:
    """Create the session on entry; close the runner and delete it on exit."""

    session_id = f"{session_prefix}-{uuid.uuid4().hex}"
    service = InMemorySessionService()
    runner = Runner(app=App(name=app_name, root_agent=agent), session_service=service)
    created = False
    try:
        await service.create_session(app_name=app_name, user_id=_USER_ID, session_id=session_id)
        created = True
        yield OneShotSession(runner, session_id)
    finally:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await runner.close()
        if created:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await service.delete_session(
                    app_name=app_name, user_id=_USER_ID, session_id=session_id
                )
