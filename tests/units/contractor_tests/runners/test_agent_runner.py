"""Unit tests for AgentRunner — the bare single-agent runner used by
RouterWorkflow. The ADK ``Runner`` is replaced with a fake that yields
canned events, so these exercise AgentRunner's own lifecycle/emission/state
logic without an LLM."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.artifacts import BaseArtifactService
from google.genai import types

import contractor.runners.agent_runner as agent_runner_mod
from contractor.runners.agent_runner import AgentRunner, AgentRunResult
from contractor.runners.artifacts import artifact_names_for_key
from contractor.runners.models import TaskRunnerEvent, TaskScopedKeys

# ─── Fakes / helpers ──────────────────────────────────────────────────────────


def _final_event(text: str):
    ev = MagicMock()
    ev.is_final_response.return_value = True
    ev.content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
    return ev


def _nonfinal_event():
    ev = MagicMock()
    ev.is_final_response.return_value = False
    ev.content = None
    return ev


def _agent(name: str = "trace"):
    return SimpleNamespace(name=name)


def _patch_runner(monkeypatch, events: list) -> dict:
    """Replace ADK Runner with a fake that yields ``events``. Returns a dict
    recording the constructor kwargs and run_async call args."""
    rec: dict = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            rec["init_kwargs"] = kwargs

        async def run_async(self, *, user_id, session_id, new_message):
            rec["user_id"] = user_id
            rec["session_id"] = session_id
            rec["new_message"] = new_message
            for ev in events:
                yield ev

    monkeypatch.setattr(agent_runner_mod, "Runner", FakeRunner)
    return rec


def _make_runner() -> AgentRunner:
    return AgentRunner(name="app", artifact_service=MagicMock(spec=BaseArtifactService))


def _collector():
    events: list[TaskRunnerEvent] = []

    async def on_event(ev: TaskRunnerEvent) -> None:
        events.append(ev)

    return events, on_event


# ─── run() lifecycle & emission ───────────────────────────────────────────────


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_emits_started_final_text_finished_in_order(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("hello")])
        runner = _make_runner()
        events, on_event = _collector()

        await runner.run(agent=_agent(), message="hi", on_event=on_event)

        types_seen = [e.type for e in events]
        assert types_seen == ["agent_run_started", "final_text", "agent_run_finished"]

    @pytest.mark.asyncio
    async def test_returns_last_final_text(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("first"), _final_event("second")])
        runner = _make_runner()

        result = await runner.run(agent=_agent(), message="hi")

        assert isinstance(result, AgentRunResult)
        assert result.final_text == "second"

    @pytest.mark.asyncio
    async def test_skips_events_without_text(self, monkeypatch):
        _patch_runner(monkeypatch, [_nonfinal_event(), _nonfinal_event()])
        runner = _make_runner()
        events, on_event = _collector()

        result = await runner.run(agent=_agent(), message="hi", on_event=on_event)

        assert result.final_text == ""
        assert [e.type for e in events] == ["agent_run_started", "agent_run_finished"]

    @pytest.mark.asyncio
    async def test_string_message_wrapped_in_user_content(self, monkeypatch):
        rec = _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        await runner.run(agent=_agent(), message="please trace")

        msg = rec["new_message"]
        assert isinstance(msg, types.Content)
        assert msg.role == "user"
        assert msg.parts[0].text == "please trace"

    @pytest.mark.asyncio
    async def test_content_message_passed_through(self, monkeypatch):
        rec = _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()
        content = types.Content(role="user", parts=[types.Part(text="raw")])

        await runner.run(agent=_agent(), message=content)

        assert rec["new_message"] is content

    @pytest.mark.asyncio
    async def test_event_name_overrides_task_name(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()
        events, on_event = _collector()

        await runner.run(
            agent=_agent("inner-agent"),
            message="hi",
            on_event=on_event,
            event_name="router-step",
        )

        for e in events:
            assert e.task_name == "router-step"
            assert e.payload["agent_name"] == "inner-agent"


# ─── Session handling ─────────────────────────────────────────────────────────


class TestSession:
    @pytest.mark.asyncio
    async def test_generates_session_id_when_omitted(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        result = await runner.run(agent=_agent(), message="hi")

        assert result.session_id

    @pytest.mark.asyncio
    async def test_uses_provided_session_id(self, monkeypatch):
        rec = _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        result = await runner.run(agent=_agent(), message="hi", session_id="sess-x")

        assert result.session_id == "sess-x"
        assert rec["session_id"] == "sess-x"

    @pytest.mark.asyncio
    async def test_initial_state_creates_session_and_disables_autocreate(
        self, monkeypatch
    ):
        rec = _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        result = await runner.run(
            agent=_agent(),
            message="hi",
            initial_state={"seed": "value"},
        )

        # auto_create_session must be off when we pre-created the session.
        assert rec["init_kwargs"]["auto_create_session"] is False
        # State survives and is returned (real InMemorySessionService).
        assert result.final_state.get("seed") == "value"

    @pytest.mark.asyncio
    async def test_no_initial_state_enables_autocreate(self, monkeypatch):
        rec = _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        await runner.run(agent=_agent(), message="hi")

        assert rec["init_kwargs"]["auto_create_session"] is True


# ─── Emission edge cases ──────────────────────────────────────────────────────


class TestEmission:
    @pytest.mark.asyncio
    async def test_no_handler_is_noop(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()

        # Must not raise even though nothing consumes events.
        result = await runner.run(agent=_agent(), message="hi", on_event=None)
        assert result.final_text == "ok"

    @pytest.mark.asyncio
    async def test_handler_cleared_after_run(self, monkeypatch):
        _patch_runner(monkeypatch, [_final_event("ok")])
        runner = _make_runner()
        _, on_event = _collector()

        await runner.run(agent=_agent(), message="hi", on_event=on_event)

        assert runner._on_event is None

    @pytest.mark.asyncio
    async def test_handler_cleared_even_on_error(self, monkeypatch):
        class FakeRunner:
            def __init__(self, **kwargs):
                pass

            async def run_async(self, *, user_id, session_id, new_message):
                raise RuntimeError("kaboom")
                yield  # pragma: no cover  (makes this an async generator)

        monkeypatch.setattr(agent_runner_mod, "Runner", FakeRunner)
        runner = _make_runner()
        _, on_event = _collector()

        with pytest.raises(RuntimeError, match="kaboom"):
            await runner.run(agent=_agent(), message="hi", on_event=on_event)

        assert runner._on_event is None


# ─── Artifact publishing ──────────────────────────────────────────────────────


class TestPublishArtifacts:
    @pytest.mark.asyncio
    async def test_reads_task_scoped_state_and_saves(self, monkeypatch):
        save = AsyncMock(return_value={"result": "k/result"})
        monkeypatch.setattr(agent_runner_mod, "save_result_artifacts", save)
        runner = _make_runner()

        keys = TaskScopedKeys(0)
        result = AgentRunResult(
            final_text="",
            session_id="s",
            final_state={
                keys.result: "R-text",
                keys.summary: "S-text",
                keys.pool: [{"rec": 1}],
            },
        )

        out = await runner.publish_artifacts(user_id="u", key="k", result=result)

        assert out == {"result": "k/result"}
        kwargs = save.await_args.kwargs
        assert kwargs["key"] == "k"
        assert kwargs["result"] == "R-text"
        assert kwargs["summary"] == "S-text"
        assert kwargs["records"] == [{"rec": 1}]

    @pytest.mark.asyncio
    async def test_missing_state_defaults_to_empty(self, monkeypatch):
        save = AsyncMock(return_value={})
        monkeypatch.setattr(agent_runner_mod, "save_result_artifacts", save)
        runner = _make_runner()

        result = AgentRunResult(final_text="", session_id="s", final_state={})
        await runner.publish_artifacts(user_id="u", key="k", result=result)

        kwargs = save.await_args.kwargs
        assert kwargs["result"] == ""
        assert kwargs["summary"] == ""
        assert kwargs["records"] == []

    def test_artifact_names_matches_helper(self):
        assert AgentRunner.artifact_names("k") == artifact_names_for_key("k")
