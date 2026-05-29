"""Unit tests for AdkTracePlugin — converts ADK lifecycle hooks into trace
events carrying full tool-call/result/error payloads and state snapshots."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from contractor.runners.agio import AgioEventType
from contractor.runners.plugins.trace_plugin import AdkTracePlugin


class _EmitRecorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def __call__(self, event_type, **payload) -> None:
        self.events.append((event_type, payload))

    def of_type(self, event_type) -> list[dict]:
        return [p for t, p in self.events if t == event_type]


def _plugin():
    rec = _EmitRecorder()
    plugin = AdkTracePlugin(
        task_name="t", task_id=2, iteration=3, session_id="s", emit=rec
    )
    return plugin, rec


def _tool(name: str = "read_file"):
    return SimpleNamespace(name=name)


def _ctx(inv="inv1", agent="swe", state=None):
    return SimpleNamespace(invocation_id=inv, agent_name=agent, state=state or {})


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_before_and_after_run(self):
        plugin, rec = _plugin()
        inv_ctx = SimpleNamespace(invocation_id="inv1", agent_name="swe")

        await plugin.before_run_callback(invocation_context=inv_ctx)
        await plugin.after_run_callback(invocation_context=inv_ctx)

        assert rec.of_type(AgioEventType.AGENT_RUN_START)[0]["invocation_id"] == "inv1"
        assert rec.of_type(AgioEventType.AGENT_RUN_END)[0]["invocation_id"] == "inv1"


class TestToolLifecycle:
    @pytest.mark.asyncio
    async def test_before_tool_emits_call_with_args_and_state(self):
        plugin, rec = _plugin()
        ctx = _ctx(state={"k": "v"})
        await plugin.before_tool_callback(
            tool=_tool(), tool_context=ctx, args={"path": "/x"}
        )
        ev = rec.of_type(AgioEventType.ADK_TOOL_CALL)[0]
        assert ev["tool_name"] == "read_file"
        assert ev["arguments"] == {"path": "/x"}
        assert ev["state"] == {"k": "v"}
        assert ev["agent_name"] == "swe"

    @pytest.mark.asyncio
    async def test_after_tool_emits_result(self):
        plugin, rec = _plugin()
        await plugin.after_tool_callback(
            tool=_tool(), tool_context=_ctx(), args={"path": "/x"}, result={"ok": 1}
        )
        ev = rec.of_type(AgioEventType.ADK_TOOL_RESULT)[0]
        assert ev["result"] == {"ok": 1}
        assert ev["tool_name"] == "read_file"

    @pytest.mark.asyncio
    async def test_on_tool_error_emits_repr(self):
        plugin, rec = _plugin()
        await plugin.on_tool_error_callback(
            tool=_tool(), tool_context=_ctx(), args={}, error=ValueError("bad")
        )
        ev = rec.of_type(AgioEventType.ADK_TOOL_ERROR)[0]
        assert "bad" in ev["error"]

    @pytest.mark.asyncio
    async def test_on_tool_error_none_error(self):
        plugin, rec = _plugin()
        await plugin.on_tool_error_callback(
            tool=_tool(), tool_context=_ctx(), args={}, error=None
        )
        assert rec.of_type(AgioEventType.ADK_TOOL_ERROR)[0]["error"] is None


class TestEventStream:
    @pytest.mark.asyncio
    async def test_on_event_emits_and_returns_none(self):
        plugin, rec = _plugin()
        event = SimpleNamespace(author="model")
        out = await plugin.on_event_callback(
            invocation_context=_ctx(), event=event
        )
        assert out is None
        ev = rec.of_type(AgioEventType.ADK_EVENT)[0]
        assert ev["author"] == "model"
        assert ev["event"] is event


class TestContextInjection:
    @pytest.mark.asyncio
    async def test_emit_injects_plugin_context(self):
        plugin, rec = _plugin()
        await plugin.before_tool_callback(tool=_tool(), tool_context=_ctx(), args={})
        payload = rec.of_type(AgioEventType.ADK_TOOL_CALL)[0]
        assert payload["task_name"] == "t"
        assert payload["task_id"] == 2
        assert payload["iteration"] == 3
        assert payload["session_id"] == "s"
