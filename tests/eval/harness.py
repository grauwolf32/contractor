from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

SetupHook = Callable[[InMemoryArtifactService, str, str], Awaitable[None]]


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
    # Function-response parts we observed from the ADK Runner. One per
    # tool finish; the count should match ``tool_calls`` and the
    # plugin's tool_call/tool_result events. Imbalance is a signal that
    # ADK batched some calls and our `function_call` reader missed
    # them. Each entry is ``{"name": str, "ok": bool}``.
    tool_responses: list[dict[str, Any]] = field(default_factory=list)
    # Agio events emitted by attached ADK plugins (e.g. AdkMetricsPlugin).
    # When the harness wires up the metrics plugin, each entry is the kwargs
    # dict of the emitted event with ``event_type`` added — same shape as a
    # row in metrics.jsonl.
    metrics_events: list[dict[str, Any]] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    def calls_named(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]

    def events_of(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.metrics_events if e.get("event_type") == event_type]

    def capture_imbalance(self) -> dict[str, int]:
        """Difference between observed tool_responses and tool_calls.

        Positive ``missing_calls`` means we captured a function_response
        without a matching function_call — a sign ADK emitted parallel
        tool invocations and our reader missed some. Cross-checks the
        plugin's metrics events: ``missing_metrics`` is the analogous
        gap between tool_result events and tool_call events.
        """
        plugin_calls = sum(
            1
            for e in self.metrics_events
            if str(e.get("event_type")) == "tool_call"
        )
        plugin_results = sum(
            1
            for e in self.metrics_events
            if str(e.get("event_type")) == "tool_result"
        )
        return {
            "tool_calls_seen": len(self.tool_calls),
            "tool_responses_seen": len(self.tool_responses),
            "missing_calls": len(self.tool_responses) - len(self.tool_calls),
            "plugin_tool_calls": plugin_calls,
            "plugin_tool_results": plugin_results,
            "missing_metrics": plugin_results - plugin_calls,
        }


async def run_agent(
    agent: BaseAgent,
    *,
    user_message: str,
    initial_state: dict[str, Any] | None = None,
    app_name: str = "eval",
    user_id: str = "eval-user",
    timeout_s: float = 600.0,
    setup: SetupHook | None = None,
    plugins: list[Any] | None = None,
    metrics_events: list[dict[str, Any]] | None = None,
    artifact_dir: Path | None = None,
) -> AgentRun:
    """Drive `agent` end-to-end via ADK Runner and capture outputs.

    `setup`, if given, is awaited with `(artifact_service, app_name, user_id)`
    after the artifact service is created but before the session is opened —
    use it to pre-populate memory/skills.

    `plugins`, if given, is forwarded to the ADK ``Runner`` (e.g. to wire up
    ``AdkMetricsPlugin`` for token/timing capture). `metrics_events`, when
    supplied, is the list those plugins should append to; it is attached to
    the returned ``AgentRun`` so callers can inspect it.

    `artifact_dir`, if given, persists the agent's artifacts to that directory
    via ``FileArtifactService`` — a live, on-disk trace (memory, saved
    artifacts) that survives the run for offline analysis. Default is an
    in-memory service (no on-disk trace).
    """
    if artifact_dir is not None:
        from google.adk.artifacts import FileArtifactService

        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        artifact_service: Any = FileArtifactService(root_dir=str(artifact_dir))
    else:
        artifact_service = InMemoryArtifactService()
    session_service = InMemorySessionService()
    session_id = uuid4().hex

    if setup is not None:
        await setup(artifact_service, app_name, user_id)

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        artifact_service=artifact_service,
        plugins=plugins or [],
        auto_create_session=True,
    )

    message = types.Content(role="user", parts=[types.Part(text=user_message)])

    final_text = ""
    tool_calls: list[ToolCall] = []
    tool_responses: list[dict[str, Any]] = []

    async def _consume() -> None:
        nonlocal final_text
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
            state_delta=initial_state or None,
        ):
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None and fc.name:
                    tool_calls.append(
                        ToolCall(name=fc.name, args=dict(fc.args or {}))
                    )
                fr = getattr(part, "function_response", None)
                if fr is not None and getattr(fr, "name", None):
                    response_payload = getattr(fr, "response", None) or {}
                    is_error = (
                        isinstance(response_payload, dict)
                        and any(
                            response_payload.get(k)
                            for k in ("error", "errors", "error_message")
                        )
                    )
                    tool_responses.append(
                        {
                            "name": fr.name,
                            "ok": not is_error,
                        }
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
        tool_responses=tool_responses,
        metrics_events=list(metrics_events) if metrics_events is not None else [],
    )
