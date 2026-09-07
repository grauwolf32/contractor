"""Real pinned ADK continuation proof; scripted outputs, no provider or Audit activation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result, tool_call
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contractor_runtime.adk_runtime import WorkerBudgetExceeded, _InvocationBudget
from contractor_runtime.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker_state import WorkerStateStore


class CountedState(WorkerStateStore):
    begins = 0
    completions = 0

    async def begin_invocation(self, **kwargs):
        self.begins += 1
        return await super().begin_invocation(**kwargs)

    async def complete_invocation(self, **kwargs):
        self.completions += 1
        return await super().complete_invocation(**kwargs)


def test_real_runner_continuation_preserves_one_invocation_and_tool_order():
    async def scenario():
        state = CountedState()
        artifact_refs, observations = [], []
        budget = _InvocationBudget(4, 2, 1000, state.metrics)
        budget.start()

        def observe(owner, cursor):
            observations.extend(artifact_refs[cursor:])

        async def probe(name: str) -> dict:
            """Retain one exact scripted artifact reference for the lifecycle proof."""
            artifact_refs.append({"namespace": "proof", "name": name, "revision": "1"})
            probe.artifact_observation_cursor = len(artifact_refs)
            return {"ok": True}

        probe.artifact_observation_cursor = 0
        plugin = WorkerInstrumentationPlugin(
            state=state,
            budget=lambda: budget,
            instrumentation=None,
            model_alias="scripted",
            observe_artifacts=observe,
        )
        model = scripted_model(
            [
                tool_call("probe", {"name": "first"}, call_id="one"),
                text_result("partial"),
                tool_call("probe", {"name": "second"}, call_id="two"),
                text_result("complete"),
            ]
        )
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="proof", user_id="user")
        runner = Runner(
            app=App(
                name="proof",
                root_agent=LlmAgent(name="worker", model=model, tools=[probe]),
                plugins=[plugin],
            ),
            session_service=sessions,
        )
        invocation = "worker-continuation"
        plugin.prepare_invocation(invocation_id=invocation, subtask_id="1")
        for turn, prompt in enumerate(
            ("inspect first item", "Runtime reminder: inspect second item")
        ):
            if turn:
                plugin.prepare_continuation(invocation_id=invocation)
            events = [
                event
                async for event in runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    invocation_id=invocation,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                )
            ]
            assert events and all(event.invocation_id == invocation for event in events)
            snapshot = await state.snapshot()
            assert snapshot["currentInvocation"]["metrics"]["modelCalls"] == 2 * (turn + 1)
            assert snapshot["currentInvocation"]["metrics"]["toolCalls"] == turn + 1
            assert plugin._next_tool_ordinal == turn + 2
            assert state.begins == 1 and state.completions == 0
        assert budget.model_calls == 4 and budget.tool_calls == 2
        with pytest.raises(WorkerBudgetExceeded):
            budget.before_model_call()
        assert len(observations) == 2 and observations == artifact_refs
        assert "Runtime reminder" in model.requests[-1]["contentText"]
        await plugin.complete_invocation(invocation_id=invocation, phase="succeeded")
        snapshot = await state.snapshot()
        assert snapshot["currentInvocation"] is None
        assert snapshot["lastCompletedInvocation"]["metrics"]["modelCalls"] == 4
        assert state.begins == state.completions == 1
        await plugin.complete_invocation(invocation_id=invocation, phase="succeeded")
        assert state.completions == 1
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(invocation_id=invocation)
        await plugin.close()
        await runner.close()

    asyncio.run(scenario())


def test_continuation_is_explicit_one_use_and_cannot_change_invocation():
    async def scenario():
        state = CountedState()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            budget=lambda: None,
            instrumentation=None,
            model_alias="scripted",
            observe_artifacts=lambda *_: None,
        )
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(invocation_id="worker-one")
        context = SimpleNamespace(invocation_id="worker-one", session=SimpleNamespace(state={}))
        plugin.prepare_invocation(invocation_id="worker-one", subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError, match="not prepared"):
            await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(invocation_id="worker-foreign")
        plugin.prepare_continuation(invocation_id="worker-one")
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(invocation_id="worker-one")
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.before_run_callback(
                invocation_context=SimpleNamespace(invocation_id="worker-foreign")
            )
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.before_run_callback(
                invocation_context=SimpleNamespace(
                    invocation_id="worker-one", session=SimpleNamespace(state={})
                )
            )
        await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError, match="not prepared"):
            await plugin.before_run_callback(invocation_context=context)
        assert state.begins == 1
        await plugin.complete_invocation(invocation_id="worker-one", phase="failed")
        await plugin.close()

    asyncio.run(scenario())
