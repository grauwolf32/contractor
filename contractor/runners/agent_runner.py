from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.artifacts import BaseArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr

from contractor.runners._helpers import _extract_final_text
from contractor.runners.artifacts import (artifact_names_for_key,
                                          save_result_artifacts)
from contractor.runners.models import (ArtifactKind, TaskRunnerEvent,
                                       TaskRunnerEventHandler, TaskScopedKeys)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AgentRunResult:
    final_text: str
    session_id: str
    final_state: dict[str, Any]


class AgentRunner(BaseModel):
    """Runs a single ADK ``LlmAgent`` end-to-end.

    Knows nothing about templates, retries, queues, or task artifacts —
    those belong to higher-level orchestrators (e.g. ``TaskRunner``) or
    to workflows that compose ``AgentRunner`` directly.

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

            has_initial_state = bool(initial_state)
            if has_initial_state:
                await self.session_service.create_session(
                    app_name=self.name,
                    user_id=user_id,
                    state=initial_state,
                    session_id=session_id,
                )

            runner = Runner(
                agent=agent,
                app_name=self.name,
                session_service=self.session_service,
                artifact_service=self.artifact_service,
                plugins=plugins or [],
                auto_create_session=not has_initial_state,
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

    # ── Artifact publishing ───────────────────────────────────────────────

    async def publish_artifacts(
        self,
        *,
        user_id: str,
        key: str,
        result: AgentRunResult,
        task_id: int = 0,
    ) -> dict[ArtifactKind, str]:
        """Persist ``result``/``summary``/``records`` artifacts from a run.

        Mirrors ``TaskRunner._publish_task_artifacts``: reads task-scoped
        ``result``/``summary``/``pool`` values from ``result.final_state``
        using ``TaskScopedKeys(task_id)`` and saves each as ``{key}/{kind}``.
        """
        keys = TaskScopedKeys(task_id)
        state = result.final_state
        return await save_result_artifacts(
            artifact_service=self.artifact_service,
            app_name=self.name,
            user_id=user_id,
            key=key,
            result=state.get(keys.result, "") or "",
            summary=state.get(keys.summary, "") or "",
            records=state.get(keys.pool, []),
        )

    @staticmethod
    def artifact_names(key: str) -> dict[ArtifactKind, str]:
        """Return the ``{kind: filename}`` mapping that ``publish_artifacts`` produces."""
        return artifact_names_for_key(key)

    # ── Session management ────────────────────────────────────────────────

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
