"""ADK behavior requiring Contractor-owned exact history identity (WRK-R187, WRK-R190)."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from test_adk_gateway_conformance import chat, gateway_model

TASK = "Exact objective; parameter=value; input=source@revision; subtask=7"


def source_event(index, *, timestamp=None):
    return Event(
        id=f"source-{index}",
        author="worker",
        timestamp=timestamp or index + 1,
        content=types.Content(role="model", parts=[types.Part(text=f"raw-{index}" + "x" * 1024)]),
    )


@pytest.mark.parametrize("case", ["equal-start", "equal-end", "out-of-order", "overlap", "foreign"])
def test_native_timestamp_range_cannot_replace_exact_source_identity(case):
    async def scenario():
        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="native", user_id="user")
        source = [source_event(i) for i in range(3)]
        if case == "equal-start":
            source[0].timestamp = source[2].timestamp = 1
        elif case in {"equal-end", "foreign"}:
            source[1].timestamp = source[2].timestamp = 2
        elif case == "out-of-order":
            source[0].timestamp, source[1].timestamp, source[2].timestamp = 3, 1, 2
        if case == "foreign":
            source[2].branch = "other"
        for event in source:
            await sessions.append_event(session=current, event=event)
        if case == "overlap":
            await sessions.append_event(
                session=current,
                event=Event(
                    author="worker",
                    actions=EventActions(
                        compaction=EventCompaction(
                            start_timestamp=1,
                            end_timestamp=2,
                            compacted_content=types.Content(
                                role="model", parts=[types.Part(text="old-memory")]
                            ),
                        )
                    ),
                ),
            )
        await sessions.append_event(
            session=current,
            event=Event(
                author="worker",
                actions=EventActions(
                    compaction=EventCompaction(
                        start_timestamp=min(e.timestamp for e in source[:2]),
                        end_timestamp=max(e.timestamp for e in source[:2]),
                        compacted_content=types.Content(
                            role="model", parts=[types.Part(text="new-memory")]
                        ),
                    )
                ),
            ),
        )
        payloads = []

        async def gateway(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=chat({"content": "done"}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            model = gateway_model(http)
            runner = Runner(
                agent=LlmAgent(name="worker", model=model),
                app_name="native",
                session_service=sessions,
            )
            try:
                async with aclosing(
                    runner.run_async(
                        user_id="user",
                        session_id=current.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=TASK)]),
                    )
                ) as events:
                    async for _ in events:
                        pass
            finally:
                await runner.close()
                await model.close()
        serialized = json.dumps(payloads[0])
        # Native inclusive ranges absorb the retained neighbor. Overlap selects newer memory.
        assert ("raw-2" in serialized) is (case == "overlap")
        assert "new-memory" in serialized and "old-memory" not in serialized

    asyncio.run(scenario())


def test_adk_event_id_is_not_append_idempotency():
    async def scenario():
        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="proof", user_id="user")
        marker = Event(id="same-marker", author="worker")
        await sessions.append_event(session=current, event=marker)
        await sessions.append_event(session=current, event=marker)
        divergent = marker.model_copy(
            update={"content": types.Content(role="model", parts=[types.Part(text="other")])}
        )
        await sessions.append_event(session=current, event=divergent)
        stored = await sessions.get_session(app_name="proof", user_id="user", session_id=current.id)
        assert len([e for e in stored.events if e.id == marker.id]) == 2

    asyncio.run(scenario())
