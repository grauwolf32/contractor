"""Unit tests for the eval harness timeout path (``tests/eval/harness.py``).

Drives ``run_agent`` with dummy (non-LLM) ADK agents: a hanging agent must
raise ``AgentRunTimeout`` carrying the partial ``AgentRun`` captured before
the deadline, and a fast agent must return a normal run (``timed_out=False``).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from tests.eval.harness import AgentRunTimeout, run_agent


class _HangingAgent(BaseAgent):
    """Emits one tool-call event, then never finishes."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="probe_tool", args={"x": 1}
                        )
                    )
                ],
            ),
        )
        await asyncio.sleep(3600)


class _FastAgent(BaseAgent):
    """Finishes immediately with a final text response."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text="done")]),
        )


def test_timeout_raises_with_partial_run():
    agent = _HangingAgent(name="hanging_agent")

    with pytest.raises(AgentRunTimeout) as exc_info:
        asyncio.run(run_agent(agent, user_message="go", timeout_s=0.5))

    err = exc_info.value
    # Still a timeout failure for consumers that catch asyncio.TimeoutError.
    assert isinstance(err, asyncio.TimeoutError)
    assert err.timeout_s == 0.5

    partial = err.partial
    assert partial.timed_out is True
    # The tool call emitted before the deadline is preserved.
    assert partial.tool_names() == ["probe_tool"]
    assert partial.calls_named("probe_tool")[0].args == {"x": 1}


def test_fast_run_is_not_timed_out():
    agent = _FastAgent(name="fast_agent")

    run = asyncio.run(run_agent(agent, user_message="go", timeout_s=30.0))

    assert run.timed_out is False
    assert run.final_text == "done"
