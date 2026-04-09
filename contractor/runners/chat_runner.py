from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contractor.agents.chat_agent.agent import build_chat_agents
from contractor.agents.chat_agent.models import ChatContext
from contractor.tools.fs import RootedLocalFileSystem


@dataclass(slots=True)
class ChatRunnerEvent:
    type: str
    payload: dict[str, Any]
    task_name: Optional[str] = None
    task_id: Optional[str] = None


TaskRunnerEventHandler = Callable[[ChatRunnerEvent], Awaitable[None]]


class ChatRunner:
    """
    Interactive runner for the `chat` pipeline.

    It intentionally mimics the repo's batch runner surface by exposing
    `run(user_id=..., on_event=...)`, but internally it starts a REPL and uses
    ADK's Runner + InMemorySessionService for turn-based chat.
    """

    def __init__(
        self,
        *,
        project_path: str,
        folder_name: str,
        user_id: str,
        model: str,
        app_name: str,
        artifact_service: Any,
    ) -> None:
        self.ctx = ChatContext(
            project_path=str(project_path),
            folder_name=folder_name,
            user_id=user_id,
            app_name=app_name,
            model=model,
            session_id=f"chat-{uuid4().hex[:8]}",
        )
        self.artifact_service = artifact_service
        self.fs = RootedLocalFileSystem(root_path=project_path)
        self.root_agent = build_chat_agents(ctx=self.ctx, fs=self.fs)
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.root_agent,
            app_name=app_name,
            artifact_service=artifact_service,
            session_service=self.session_service,
        )

    async def _emit(
        self,
        on_event: Optional[TaskRunnerEventHandler],
        event_type: str,
        **payload: Any,
    ) -> None:
        if on_event is None:
            return
        await on_event(ChatRunnerEvent(type=event_type, payload=payload))

    async def _ensure_session(self) -> str:
        session = await self.session_service.create_session(
            app_name=self.ctx.app_name,
            user_id=self.ctx.user_id,
            session_id=self.ctx.session_id,
        )
        return session.id

    async def _run_turn(
        self,
        *,
        session_id: str,
        text: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> str:
        await self._emit(on_event, "chat_turn_started", text=text, session_id=session_id)

        content = types.Content(role="user", parts=[types.Part(text=text)])
        final_response = ""

        async for event in self.runner.run_async(
            user_id=self.ctx.user_id,
            session_id=session_id,
            new_message=content,
        ):
            # Surface a simplified event stream to the existing CLI UI/metrics hooks.
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                parts = event.content.parts
                for part in parts:
                    if getattr(part, "text", None):
                        await self._emit(
                            on_event,
                            "chat_chunk",
                            text=part.text,
                            partial=getattr(event, "partial", False),
                            author=getattr(event, "author", None),
                        )
                        if event.is_final_response():
                            final_response += part.text
            if event.is_final_response():
                await self._emit(
                    on_event,
                    "chat_turn_finished",
                    text=final_response,
                    session_id=session_id,
                )
        return final_response

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> dict[str, Any]:
        self.ctx.user_id = user_id
        session_id = await self._ensure_session()
        await self._emit(
            on_event,
            "run_started",
            session_id=session_id,
            agent_name="contractor_chat_agent",
        )

        print("Contractor Chat started. Type /exit to quit.")
        print("Try: 'analyze repo structure', 'trace /users/{id}', 'update OAS for auth routes'.")

        transcript: list[dict[str, str]] = []
        while True:
            try:
                user_text = await asyncio.to_thread(input, "\nYou> ")
            except EOFError:
                user_text = "/exit"

            if not user_text.strip():
                continue
            if user_text.strip().lower() in {"/exit", "exit", "quit"}:
                break

            transcript.append({"role": "user", "text": user_text})
            reply = await self._run_turn(session_id=session_id, text=user_text, on_event=on_event)
            transcript.append({"role": "assistant", "text": reply})
            if reply:
                print(f"\nAgent> {reply}")

        await self._emit(on_event, "run_finished", session_id=session_id, turns=len(transcript))
        return {"session_id": session_id, "transcript": transcript}