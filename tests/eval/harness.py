from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class AgentRun:
    final_text: str
    state: dict[str, Any]
    artifacts: dict[str, str]
    tool_calls: list[ToolCall] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    def calls_named(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]


async def run_agent(
    agent: BaseAgent,
    *,
    user_message: str,
    initial_state: Optional[dict[str, Any]] = None,
    app_name: str = "eval",
    user_id: str = "eval-user",
    timeout_s: float = 600.0,
) -> AgentRun:
    """Drive `agent` end-to-end via ADK Runner and capture outputs."""
    artifact_service = InMemoryArtifactService()
    session_service = InMemorySessionService()
    session_id = uuid4().hex

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {},
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    message = types.Content(role="user", parts=[types.Part(text=user_message)])

    final_text = ""
    tool_calls: list[ToolCall] = []

    async def _consume() -> None:
        nonlocal final_text
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None and fc.name:
                    tool_calls.append(
                        ToolCall(name=fc.name, args=dict(fc.args or {}))
                    )
            if event.is_final_response():
                text = "\n".join(
                    p.text for p in parts if getattr(p, "text", None)
                ).strip()
                if text:
                    final_text = text

    await asyncio.wait_for(_consume(), timeout=timeout_s)

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    state = dict(session.state) if session is not None else {}

    artifacts: dict[str, str] = {}
    for scope_session_id in (session_id, None):
        keys = await artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=scope_session_id
        )
        for key in keys:
            if key in artifacts:
                continue
            part = await artifact_service.load_artifact(
                app_name=app_name,
                user_id=user_id,
                session_id=scope_session_id,
                filename=key,
            )
            if part is None:
                continue
            text = getattr(part, "text", None) or ""
            artifacts[key] = text

    return AgentRun(
        final_text=final_text,
        state=state,
        artifacts=artifacts,
        tool_calls=tool_calls,
    )
