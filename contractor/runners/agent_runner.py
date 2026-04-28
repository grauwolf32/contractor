from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.artifacts import BaseArtifactService
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr

from contractor.runners.models import TaskRunnerEvent, TaskRunnerEventHandler

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AgentRunResult:
    final_text: str
    session_id: str
    final_state: dict[str, Any]


def _extract_final_text(event: Event) -> str:
    if not event.is_final_response():
        return ""
    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return "\n".join(
        text for part in parts if (text := getattr(part, "text", None))
    ).strip()


class AgentRunner(BaseModel):
    """Runs a single ADK ``LlmAgent`` end-to-end.

    Knows nothing about templates, retries, queues, or task artifacts —
    those belong to higher-level orchestrators (e.g. ``TaskRunner``) or
    to pipelines that compose ``AgentRunner`` directly.

    Emits ``TaskRunnerEvent`` so callers can reuse the same handler /
    plugin contracts as ``TaskRunner``.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Logical name for this runner (app_name)")
    artifact_service: BaseArtifactService
    session_service: InMemorySessionService = Field(
        default_factory=InMemorySessionService
    )

    _on_event: Optional[TaskRunnerEventHandler] = PrivateAttr(default=None)

    async def run(
        self,
        *,
        agent: LlmAgent,
        message: Union[str, types.Content],
        user_id: str = "cli-user",
        session_id: Optional[str] = None,
        initial_state: Optional[dict[str, Any]] = None,
        plugins: Optional[list[Any]] = None,
        on_event: Optional[TaskRunnerEventHandler] = None,
        event_name: Optional[str] = None,
    ) -> AgentRunResult:
        """Run ``agent`` against ``message`` and return final text + state.

        ``event_name`` is used as ``task_name`` on emitted events when the
        caller wants to override the agent's own name (e.g. to surface a
        higher-level identifier in the UI).
        """
        self._on_event = on_event
        emit_name = event_name or agent.name
        try:
            session_id = session_id or uuid4().hex
            await self._ensure_session(user_id, session_id, initial_state or {})

            content = (
                types.Content(role="user", parts=[types.Part(text=message)])
                if isinstance(message, str)
                else message
            )

            await self._emit(
                "agent_run_started",
                task_name=emit_name,
                agent_name=agent.name,
                session_id=session_id,
            )

            runner = Runner(
                agent=agent,
                app_name=self.name,
                session_service=self.session_service,
                artifact_service=self.artifact_service,
                plugins=plugins or [],
            )

            final_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            ):
                event_text = _extract_final_text(event)
                if not event_text:
                    continue
                final_text = event_text
                await self._emit(
                    "final_text",
                    task_name=emit_name,
                    agent_name=agent.name,
                    session_id=session_id,
                    text=event_text,
                )

            final_state = await self._get_session_state(user_id, session_id)

            await self._emit(
                "agent_run_finished",
                task_name=emit_name,
                agent_name=agent.name,
                session_id=session_id,
            )

            return AgentRunResult(
                final_text=final_text,
                session_id=session_id,
                final_state=final_state,
            )
        finally:
            self._on_event = None

    # ── Session management ────────────────────────────────────────────────

    async def _ensure_session(
        self,
        user_id: str,
        session_id: str,
        initial_state: dict[str, Any],
    ):
        existing = await self.session_service.get_session(
            app_name=self.name, user_id=user_id, session_id=session_id
        )
        if existing is not None:
            return existing
        return await self.session_service.create_session(
            app_name=self.name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state,
        )

    async def _get_session_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        session = await self.session_service.get_session(
            app_name=self.name, user_id=user_id, session_id=session_id
        )
        return dict(session.state) if session else {}

    # ── Event emission ────────────────────────────────────────────────────

    async def _emit(self, event_type: str, **payload: Any) -> None:
        if self._on_event is None:
            return
        task_name = payload.pop("task_name", self.name)
        task_id = payload.pop("task_id", 0)
        await self._on_event(
            TaskRunnerEvent(
                type=event_type,
                task_name=task_name,
                task_id=task_id,
                payload=payload,
            )
        )
