"""Real pinned ADK continuation proof; scripted outputs, no provider or Audit activation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerStateStore
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from contractor_runtime.worker.budget import WorkerBudgetExceeded, _InvocationBudget
from contractor_runtime.worker.function_tools import ContractorFunctionTool
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin


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
        tool_invocations = []
        budget = _InvocationBudget(4, 2, 1000, state.metrics)
        budget.start()

        def observe(owner, cursor):
            observations.extend(artifact_refs[cursor:])

        async def probe(name: str, tool_context: ToolContext) -> dict:
            """Retain one exact scripted artifact reference for the lifecycle proof."""
            tool_invocations.append(tool_context.invocation_id)
            artifact_refs.append({"namespace": "proof", "name": name, "revision": "1"})
            probe.artifact_observation_cursor = len(artifact_refs)
            return {"ok": True}

        probe.artifact_observation_cursor = 0
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=None,
            model_alias="scripted",
            observe_artifacts=observe,
            allowed_tool_names=frozenset({probe.__name__}),
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
                root_agent=LlmAgent(
                    name="worker",
                    model=model,
                    tools=[
                        ContractorFunctionTool(
                            probe,
                            logical_invocation_id=plugin.logical_invocation_id_for_tool,
                        )
                    ],
                ),
                plugins=[plugin],
            ),
            session_service=sessions,
        )
        invocation = "worker-continuation"
        plugin.prepare_invocation(invocation_id=invocation, subtask_id="1")
        for turn, prompt in enumerate(
            ("inspect first item", "Runtime reminder: inspect second item")
        ):
            sdk_invocation = invocation if turn == 0 else f"{invocation}-turn-{turn}"
            if turn:
                plugin.prepare_continuation(
                    invocation_id=invocation, sdk_invocation_id=sdk_invocation
                )
            events = [
                event
                async for event in runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    invocation_id=sdk_invocation,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                )
            ]
            assert events and all(event.invocation_id == sdk_invocation for event in events)
            snapshot = await state.snapshot()
            assert snapshot["currentInvocation"]["invocationId"] == invocation
            assert snapshot["currentInvocation"]["metrics"]["modelCalls"] == 2 * (turn + 1)
            assert snapshot["currentInvocation"]["metrics"]["toolCalls"] == turn + 1
            assert plugin._next_tool_ordinal == turn + 2
            assert state.begins == 1 and state.completions == 0
        assert budget.model_calls == 4 and budget.tool_calls == 2
        with pytest.raises(WorkerBudgetExceeded):
            budget.before_model_call()
        assert len(observations) == 2 and observations == artifact_refs
        assert tool_invocations == [invocation, invocation]
        assert "Runtime reminder" in model.requests[-1]["contentText"]
        await plugin.complete_invocation(invocation_id=invocation, phase="succeeded")
        snapshot = await state.snapshot()
        assert snapshot["currentInvocation"] is None
        assert snapshot["lastCompletedInvocation"]["metrics"]["modelCalls"] == 4
        assert state.begins == state.completions == 1
        await plugin.complete_invocation(invocation_id=invocation, phase="succeeded")
        assert state.completions == 1
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(
                invocation_id=invocation, sdk_invocation_id=f"{invocation}-after-completion"
            )
        await plugin.close()
        await runner.close()

    asyncio.run(scenario())


def test_continuation_is_explicit_one_use_and_cannot_change_invocation():
    async def scenario():
        state = CountedState()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: None,
            instrumentation=None,
            model_alias="scripted",
            observe_artifacts=lambda *_: None,
            allowed_tool_names=frozenset(),
        )
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(
                invocation_id="worker-one", sdk_invocation_id="worker-one-turn-1"
            )
        session = SimpleNamespace(id="session", app_name="proof", user_id="user", state={})
        context = SimpleNamespace(invocation_id="worker-one", session=session)
        plugin.prepare_invocation(invocation_id="worker-one", subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError, match="not prepared"):
            await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(
                invocation_id="worker-foreign", sdk_invocation_id="worker-foreign-turn-1"
            )
        plugin.prepare_continuation(
            invocation_id="worker-one", sdk_invocation_id="worker-one-turn-1"
        )
        with pytest.raises(RuntimeError):
            plugin.prepare_continuation(
                invocation_id="worker-one", sdk_invocation_id="worker-one-turn-2"
            )
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.before_run_callback(
                invocation_context=SimpleNamespace(invocation_id="worker-foreign")
            )
        with pytest.raises(RuntimeError, match="stale"):
            await plugin.before_run_callback(
                invocation_context=SimpleNamespace(
                    invocation_id="worker-one-turn-1",
                    session=SimpleNamespace(
                        id="foreign-session", app_name="proof", user_id="user", state={}
                    ),
                )
            )
        context = SimpleNamespace(invocation_id="worker-one-turn-1", session=session)
        await plugin.before_run_callback(invocation_context=context)
        with pytest.raises(RuntimeError, match="not prepared"):
            await plugin.before_run_callback(invocation_context=context)
        assert state.begins == 1
        await plugin.complete_invocation(invocation_id="worker-one", phase="failed")
        await plugin.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("absent_content", [False, True])
def test_shared_empty_stop_cannot_bypass_missing_audit_results(tmp_path, absent_content):
    from test_adk_runtime import stage_request
    from test_audit_completion_runtime import runtime_for

    async def scenario():
        responses = []
        for _ in range(3):
            response = text_result("")
            response.finish_reason = types.FinishReason.STOP
            if absent_content:
                response.content = None
            responses.append(response)
        runtime, model, client, state, _ = await runtime_for(
            tmp_path, responses, model_calls=3, tool_calls=1, total_tokens=30
        )
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is None and result.failure.code == "audit_result_incomplete"
            assert result.failure.retryable
            assert not client.writes
            assert len(model.requests) == 3
            assert (
                sum(
                    "Runtime completion reminder" in request["contentText"]
                    for request in model.requests
                )
                == 2
            )
            assert state.begins == state.completions == 1
            metrics = (await state.snapshot())["lastCompletedInvocation"]["metrics"]
            assert metrics["modelCalls"] == 3 and metrics["toolCalls"] == 0
            assert metrics["totalTokens"] == 30
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
