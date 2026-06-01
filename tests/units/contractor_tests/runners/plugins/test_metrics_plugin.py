"""Unit tests for AdkMetricsPlugin — the per-agent/per-tool metrics plugin
attached to every runner. Covers the pure helpers, the call tracker that
de-dupes before/after/error callbacks, and the callback flows end-to-end with
a capturing emit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from contractor.runners.agio import AgioEventType
from contractor.runners.plugins.metrics_plugin import (
    AdkMetricsPlugin, AgentMetrics, ToolMetrics, _args_hash, _CallOutcome,
    _CallTracker, _safe_int, default_result_error_detector)

# ─── Pure helpers ──────────────────────────────────────────────────────────────


class TestResultErrorDetector:
    def test_non_dict_is_not_error(self):
        assert default_result_error_detector("oops") is False
        assert default_result_error_detector(None) is False
        assert default_result_error_detector(["error"]) is False

    @pytest.mark.parametrize("status", ["error", "failed", "failure", "ERROR"])
    def test_error_status_values(self, status):
        assert default_result_error_detector({"status": status}) is True

    def test_ok_status_is_not_error(self):
        assert default_result_error_detector({"status": "ok"}) is False
        assert default_result_error_detector({"data": 1}) is False

    def test_error_sentinel_keys(self):
        assert default_result_error_detector({"error": "boom"}) is True
        assert default_result_error_detector({"errors": ["x"]}) is True
        assert default_result_error_detector({"error_message": "bad"}) is True

    def test_empty_sentinel_values_are_not_errors(self):
        # Falsy/empty values must not be treated as an error signal.
        assert default_result_error_detector({"error": None}) is False
        assert default_result_error_detector({"errors": []}) is False
        assert default_result_error_detector({"error": ""}) is False
        assert default_result_error_detector({}) is False


class TestArgsHash:
    def test_deterministic(self):
        assert _args_hash({"a": 1}) == _args_hash({"a": 1})

    def test_key_order_independent(self):
        assert _args_hash({"a": 1, "b": 2}) == _args_hash({"b": 2, "a": 1})

    def test_distinct_args_differ(self):
        assert _args_hash({"a": 1}) != _args_hash({"a": 2})

    def test_returns_16_hex(self):
        h = _args_hash({"a": 1})
        assert len(h) == 16
        int(h, 16)  # parses as hex


class TestSafeInt:
    @pytest.mark.parametrize(
        "value,expected",
        [(None, 0), ("5", 5), (3, 3), (2.9, 2), ("x", 0), (False, 0), ([], 0)],
    )
    def test_safe_int(self, value, expected):
        assert _safe_int(value) == expected


class TestMetricContainers:
    def test_tool_metrics_record_outcome(self):
        tm = ToolMetrics()
        tm.record_outcome(_CallOutcome.SUCCESS)
        tm.record_outcome(_CallOutcome.EXCEPTION)
        tm.record_outcome(_CallOutcome.RESULT_ERROR)
        tm.record_outcome(_CallOutcome.SUCCESS)
        d = tm.as_dict()
        assert d["success_total"] == 2
        assert d["exception_errors_total"] == 1
        assert d["result_errors_total"] == 1

    def test_agent_metrics_add_usage(self):
        am = AgentMetrics()
        am.add_usage({"input": 10, "output": 5, "total": 15, "thoughts": 2, "cached": 1})
        am.add_usage({"input": 1, "output": 1, "total": 2})
        d = am.as_dict()
        assert d["llm_calls"] == 2
        assert d["input_tokens"] == 11
        assert d["output_tokens"] == 6
        assert d["total_tokens"] == 17
        assert d["thoughts_tokens"] == 2
        assert d["cached_tokens"] == 1


# ─── Call tracker ──────────────────────────────────────────────────────────────


class TestCallTracker:
    def test_register_assigns_incrementing_ids(self):
        t = _CallTracker()
        c1 = t.register("inv", "agent", "tool", {"a": 1})
        c2 = t.register("inv", "agent", "tool", {"a": 2})
        assert c1.call_id == 1
        assert c2.call_id == 2

    def test_resolve_returns_matching_pending_call(self):
        t = _CallTracker()
        c = t.register("inv", "agent", "tool", {"a": 1})
        resolved = t.resolve("inv", "agent", "tool", {"a": 1})
        assert resolved is c

    def test_resolve_skips_finished_calls(self):
        t = _CallTracker()
        c = t.register("inv", "agent", "tool", {"a": 1})
        t.finish(c)
        assert t.resolve("inv", "agent", "tool", {"a": 1}) is None

    def test_resolve_fifo_across_duplicate_fingerprints(self):
        t = _CallTracker()
        first = t.register("inv", "agent", "tool", {"a": 1})
        t.register("inv", "agent", "tool", {"a": 1})
        # Same fingerprint twice → resolve returns the oldest unfinished one.
        assert t.resolve("inv", "agent", "tool", {"a": 1}) is first
        t.finish(first)
        second_resolved = t.resolve("inv", "agent", "tool", {"a": 1})
        assert second_resolved is not None and second_resolved is not first

    def test_resolve_unknown_returns_none(self):
        t = _CallTracker()
        assert t.resolve("inv", "agent", "tool", {"a": 1}) is None

    def test_cleanup_invocation_clears_state(self):
        t = _CallTracker()
        t.register("inv1", "agent", "tool", {"a": 1})
        t.register("inv2", "agent", "tool", {"a": 1})
        t.cleanup_invocation("inv1")
        assert t.resolve("inv1", "agent", "tool", {"a": 1}) is None
        # inv2's call is untouched.
        assert t.resolve("inv2", "agent", "tool", {"a": 1}) is not None


# ─── Callback flows ─────────────────────────────────────────────────────────────


class _EmitRecorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def __call__(self, event_type, **payload) -> None:
        self.events.append((event_type, payload))

    def of_type(self, event_type) -> list[dict]:
        return [p for t, p in self.events if t == event_type]


def _plugin():
    rec = _EmitRecorder()
    plugin = AdkMetricsPlugin(
        task_name="t", task_id=0, iteration=1, session_id="s", emit=rec
    )
    return plugin, rec


def _tool(name: str = "read_file"):
    return SimpleNamespace(name=name)


def _ctx(inv: str = "inv1", agent: str = "swe", state: dict | None = None):
    return SimpleNamespace(invocation_id=inv, agent_name=agent, state=state or {})


class TestCallbackFlows:
    @pytest.mark.asyncio
    async def test_successful_call_counts_and_emits(self):
        plugin, rec = _plugin()
        tool, ctx = _tool(), _ctx()

        await plugin.before_tool_callback(tool=tool, tool_context=ctx, args={"p": 1})
        await plugin.after_tool_callback(
            tool=tool, tool_context=ctx, args={"p": 1}, result={"ok": True}
        )

        assert [t for t, _ in rec.events] == [
            AgioEventType.TOOL_CALL,
            AgioEventType.TOOL_RESULT,
        ]
        res = rec.of_type(AgioEventType.TOOL_RESULT)[0]
        assert res["successful"] is True
        assert res["result_error"] is False

        await plugin.after_run_callback(invocation_context=_ctx())
        summary = rec.of_type(AgioEventType.RUN_SUMMARY)[0]
        tool_metrics = summary["agents"]["swe"]["tools"]["read_file"]
        assert tool_metrics["calls_total"] == 1
        assert tool_metrics["success_total"] == 1
        assert summary["callback_imbalances"] == []

    @pytest.mark.asyncio
    async def test_result_error_counted(self):
        plugin, rec = _plugin()
        tool, ctx = _tool(), _ctx()

        await plugin.before_tool_callback(tool=tool, tool_context=ctx, args={"p": 1})
        await plugin.after_tool_callback(
            tool=tool, tool_context=ctx, args={"p": 1}, result={"error": "boom"}
        )

        res = rec.of_type(AgioEventType.TOOL_RESULT)[0]
        assert res["successful"] is False
        assert res["result_error"] is True

        await plugin.after_run_callback(invocation_context=_ctx())
        tm = rec.of_type(AgioEventType.RUN_SUMMARY)[0]["agents"]["swe"]["tools"][
            "read_file"
        ]
        assert tm["result_errors_total"] == 1
        assert tm["success_total"] == 0

    @pytest.mark.asyncio
    async def test_exception_counted_once_no_double_count(self):
        plugin, rec = _plugin()
        tool, ctx = _tool(), _ctx()

        await plugin.before_tool_callback(tool=tool, tool_context=ctx, args={"p": 1})
        await plugin.on_tool_error_callback(
            tool=tool, tool_context=ctx, args={"p": 1}, error=ValueError("x")
        )
        # ADK may follow on_tool_error with after_tool — must not double-count.
        await plugin.after_tool_callback(
            tool=tool, tool_context=ctx, args={"p": 1}, result={"error": "x"}
        )

        assert len(rec.of_type(AgioEventType.TOOL_EXCEPTION)) == 1
        await plugin.after_run_callback(invocation_context=_ctx())
        tm = rec.of_type(AgioEventType.RUN_SUMMARY)[0]["agents"]["swe"]["tools"][
            "read_file"
        ]
        assert tm["calls_total"] == 1
        assert tm["exception_errors_total"] == 1
        assert tm["success_total"] == 0
        assert tm["result_errors_total"] == 0

    @pytest.mark.asyncio
    async def test_after_model_records_usage(self):
        plugin, rec = _plugin()
        llm_response = SimpleNamespace(
            usage_metadata={
                "prompt_token_count": 100,
                "candidates_token_count": 20,
                "total_token_count": 120,
                "thoughts_token_count": 5,
                "cached_content_token_count": 10,
            },
            model_version="qwen3",
        )

        await plugin.after_model_callback(
            callback_context=_ctx(), llm_response=llm_response
        )

        usage_event = rec.of_type(AgioEventType.LLM_USAGE)[0]
        assert usage_event["usage"]["input"] == 100
        assert usage_event["usage"]["output"] == 20
        assert usage_event["usage"]["cached"] == 10
        assert usage_event["model"] == "qwen3"

    @pytest.mark.asyncio
    async def test_imbalance_surfaced_when_calls_unaccounted(self):
        plugin, rec = _plugin()
        tool, ctx = _tool(), _ctx()

        # Two before_tool calls, only one after_tool → one call unaccounted.
        await plugin.before_tool_callback(tool=tool, tool_context=ctx, args={"p": 1})
        await plugin.before_tool_callback(tool=tool, tool_context=ctx, args={"p": 2})
        await plugin.after_tool_callback(
            tool=tool, tool_context=ctx, args={"p": 1}, result={"ok": True}
        )

        await plugin.after_run_callback(invocation_context=_ctx())
        imbalances = rec.of_type(AgioEventType.RUN_SUMMARY)[0]["callback_imbalances"]
        assert len(imbalances) == 1
        assert imbalances[0]["missing"] == 1
        assert imbalances[0]["tool_name"] == "read_file"

    @pytest.mark.asyncio
    async def test_emit_injects_context_fields(self):
        plugin, rec = _plugin()
        await plugin.before_tool_callback(
            tool=_tool(), tool_context=_ctx(), args={"p": 1}
        )
        payload = rec.of_type(AgioEventType.TOOL_CALL)[0]
        assert payload["task_name"] == "t"
        assert payload["task_id"] == 0
        assert payload["iteration"] == 1
        assert payload["session_id"] == "s"

    @pytest.mark.asyncio
    async def test_fs_coverage_emitted_when_state_present(self):
        from contractor.tools.fs.const import FS_COVERAGE_STATE_KEY

        plugin, rec = _plugin()
        ctx = _ctx(state={FS_COVERAGE_STATE_KEY: {"visited": 3}})
        await plugin.before_tool_callback(tool=_tool(), tool_context=ctx, args={"p": 1})
        await plugin.after_tool_callback(
            tool=_tool(), tool_context=ctx, args={"p": 1}, result={"ok": True}
        )
        cov = rec.of_type(AgioEventType.FS_COVERAGE)
        assert len(cov) == 1
        assert cov[0]["fs_coverage"] == {"visited": 3}
