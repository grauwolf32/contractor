import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.artifacts import BaseArtifactService

from contractor.runners._helpers import _decode_part_text, _extract_final_text
from contractor.runners.artifacts import InvalidArtifactKeyError, artifact_names_for_key
from contractor.runners.models import (
    Checkpoint,
    CheckpointEntry,
    EventType,
    RenderedTask,
    TaskInvocation,
    TaskResult,
    TaskStatus,
    TaskTemplate,
)
from contractor.runners.task_runner import TaskNotCompletedError, TaskRunner

# ─── _extract_final_text ──────────────────────────────────────────────────────


def _mk_event(*, is_final: bool, parts=None):
    """Minimal duck-typed ADK Event substitute."""
    event = MagicMock()
    event.is_final_response.return_value = is_final
    if parts is None:
        event.content = None
    else:
        event.content = SimpleNamespace(parts=parts)
    return event


class TestExtractFinalText:
    def test_non_final_event_returns_empty(self):
        assert _extract_final_text(_mk_event(is_final=False, parts=[
            SimpleNamespace(text="ignored"),
        ])) == ""

    def test_no_content_returns_empty(self):
        assert _extract_final_text(_mk_event(is_final=True, parts=None)) == ""

    def test_joins_text_parts_strips_outer_whitespace(self):
        parts = [
            SimpleNamespace(text="  line 1"),
            SimpleNamespace(text="line 2  "),
            SimpleNamespace(text=None),
        ]
        # The function does .strip() on the joined result — outer whitespace
        # is removed but the internal newline separator is preserved.
        assert _extract_final_text(_mk_event(is_final=True, parts=parts)) == (
            "line 1\nline 2"
        )

    def test_drops_parts_without_text_attr(self):
        parts = [
            SimpleNamespace(text="kept"),
            SimpleNamespace(),  # no .text attribute at all
        ]
        assert _extract_final_text(_mk_event(is_final=True, parts=parts)) == "kept"


# ─── _decode_part_text ────────────────────────────────────────────────────────


class TestDecodePartText:
    def test_none_returns_empty(self):
        assert _decode_part_text(None) == ""

    def test_text_path(self):
        part = SimpleNamespace(text="hello", inline_data=None)
        assert _decode_part_text(part) == "hello"

    def test_inline_str_data(self):
        part = SimpleNamespace(
            text=None, inline_data=SimpleNamespace(data="payload"),
        )
        assert _decode_part_text(part) == "payload"

    def test_inline_bytes_data(self):
        part = SimpleNamespace(
            text=None, inline_data=SimpleNamespace(data=b"payload"),
        )
        assert _decode_part_text(part) == "payload"

    def test_inline_bytes_invalid_utf8_replaces_and_warns(self, caplog):
        part = SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=b"\xff\xfehello"),
        )
        # `errors="replace"` keeps the readable suffix and marks the invalid
        # bytes as U+FFFD instead of silently dropping them — with a warning.
        with caplog.at_level(logging.WARNING, logger="contractor.runners._helpers"):
            assert _decode_part_text(part) == "��hello"
        assert any("not valid UTF-8" in r.getMessage() for r in caplog.records)

    def test_inline_bytes_valid_utf8_no_warning(self, caplog):
        part = SimpleNamespace(
            text=None, inline_data=SimpleNamespace(data=b"hello"),
        )
        with caplog.at_level(logging.WARNING, logger="contractor.runners._helpers"):
            assert _decode_part_text(part) == "hello"
        assert not caplog.records

    def test_missing_inline_data_returns_empty(self):
        part = SimpleNamespace(text=None, inline_data=None)
        assert _decode_part_text(part) == ""


# ─── TaskRunner._resolve_retry_params ────────────────────────────────────────


def _make_template(
    default_iterations=1, default_artifacts=None, instructions="",
) -> TaskTemplate:
    return TaskTemplate(
        key="t",
        version="v1",
        title="T",
        objective="",
        instructions=instructions,
        output_format="",
        default_iterations=default_iterations,
        default_artifacts=default_artifacts or [],
    )


class TestResolveRetryParams:
    def test_iterations_defaults_to_template(self):
        tpl = _make_template(default_iterations=3)
        eff_iter, eff_max = TaskRunner._resolve_retry_params(tpl, None, None)
        assert eff_iter == 3
        # max_attempts defaults to max(1, iterations) when unspecified.
        assert eff_max == 3

    def test_explicit_overrides(self):
        tpl = _make_template(default_iterations=1)
        eff_iter, eff_max = TaskRunner._resolve_retry_params(tpl, 2, 5)
        assert (eff_iter, eff_max) == (2, 5)

    def test_max_attempts_defaults_to_iterations(self):
        tpl = _make_template(default_iterations=1)
        eff_iter, eff_max = TaskRunner._resolve_retry_params(tpl, 4, None)
        assert (eff_iter, eff_max) == (4, 4)

    def test_iterations_must_be_positive(self):
        tpl = _make_template()
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            TaskRunner._resolve_retry_params(tpl, 0, None)

    def test_max_attempts_must_be_ge_iterations(self):
        tpl = _make_template()
        with pytest.raises(ValueError, match="max_attempts must be >= iterations"):
            TaskRunner._resolve_retry_params(tpl, 3, 2)

    def test_zero_max_attempts_with_default_iterations_rejected(self):
        tpl = _make_template(default_iterations=2)
        with pytest.raises(ValueError, match="max_attempts must be >= iterations"):
            TaskRunner._resolve_retry_params(tpl, None, 1)


# ─── TaskNotCompletedError ───────────────────────────────────────────────────


class TestTaskNotCompletedError:
    def test_message_includes_counts(self):
        err = TaskNotCompletedError(ref="my-task", iterations=2, max_attempts=5)
        assert "my-task" in str(err)
        assert "2 time(s)" in str(err)
        assert "5 attempt(s)" in str(err)
        assert err.ref == "my-task"
        assert err.iterations == 2
        assert err.max_attempts == 5


# ─── TaskRunner retry loop ───────────────────────────────────────────────────


def _make_invocation(
    *,
    ref="task-a",
    iterations: int,
    max_attempts: int,
    template_key="t",
    template_version="v1",
    timeout_s: float | None = None,
) -> TaskInvocation:
    return TaskInvocation(
        id="inv-1",
        ref=ref,
        template_key=template_key,
        template_version=template_version,
        worker_builder=lambda **_: MagicMock(),
        iterations=iterations,
        max_attempts=max_attempts,
        timeout_s=timeout_s,
    )


def _result_for(
    template_key: str, completed: bool, idx: int, *, task_id: int = 1,
) -> TaskResult:
    return TaskResult(
        invocation_id=f"inv-{idx}",
        task_ref="task-a",
        task_key=template_key,
        task_title="t",
        template_key=template_key,
        task_id=task_id,
        session_id=f"s{idx}",
        final_response="",
        state={
            f"task::{task_id}::status": (
                TaskStatus.DONE if completed else TaskStatus.RUNNING
            ),
        },
        carry_state={},
        status="done" if completed else "running",
        result="R",
        summary="S",
        records=[],
        params={},
        input_artifacts={},
        published_artifacts={"result": "t/result", "summary": "t/summary", "records": "t/records"},
    )


@pytest.fixture()
def runner(monkeypatch):
    """A TaskRunner with all I/O-side methods stubbed.

    Tests can override `_run_single_iteration.side_effect` per case.
    """
    r = TaskRunner(name="test", artifact_service=MagicMock(spec=BaseArtifactService))

    template = _make_template(default_iterations=1)
    r.templates[("t", "v1")] = template

    # Pre-rendered task: skip the actual render step.
    rendered = RenderedTask(
        key="t",
        title="T",
        objective="",
        instructions="",
        output_format="",
        format="json",
    )
    monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())
    monkeypatch.setattr(r, "_emit", AsyncMock())
    return r


class TestRetryLoop:
    @pytest.mark.asyncio
    async def test_returns_after_required_successful_runs(self, runner, monkeypatch):
        # iterations=1, max_attempts=1, single success → return immediately.
        invocation = _make_invocation(iterations=1, max_attempts=1)
        single = AsyncMock(side_effect=[_result_for("t", True, 1)])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        result = await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert result.status == "done"
        assert single.await_count == 1
        runner._publish_task_artifacts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_streak_of_two_required(self, runner, monkeypatch):
        # iterations=2, max_attempts=3, success/success → returns on 2nd attempt.
        invocation = _make_invocation(iterations=2, max_attempts=3)
        single = AsyncMock(
            side_effect=[_result_for("t", True, 1), _result_for("t", True, 2)]
        )
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert single.await_count == 2
        assert runner._publish_task_artifacts.await_count == 2

    @pytest.mark.asyncio
    async def test_failure_does_not_increment_successful_runs(
        self, runner, monkeypatch
    ):
        # iterations=2, max_attempts=3, success/fail/success → 3 iterations to win.
        invocation = _make_invocation(iterations=2, max_attempts=3)
        single = AsyncMock(side_effect=[
            _result_for("t", True, 1),
            _result_for("t", False, 2),
            _result_for("t", True, 3),
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert single.await_count == 3
        # Publish only fires on the 2 successful iterations.
        assert runner._publish_task_artifacts.await_count == 2

    @pytest.mark.asyncio
    async def test_all_failures_raise_after_max_attempts(self, runner, monkeypatch):
        invocation = _make_invocation(iterations=1, max_attempts=3)
        single = AsyncMock(side_effect=[_result_for("t", False, i) for i in (1, 2, 3)])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        with pytest.raises(TaskNotCompletedError) as exc_info:
            await runner._run_task_with_retries(
                item=invocation, task_id=1, user_id="u", total_tasks=1,
            )
        assert exc_info.value.ref == "task-a"
        assert exc_info.value.max_attempts == 3
        assert single.await_count == 3
        runner._publish_task_artifacts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_streak_not_enough_raises(self, runner, monkeypatch):
        # iterations=3, max_attempts=4, only 2 successes → raise.
        invocation = _make_invocation(iterations=3, max_attempts=4)
        single = AsyncMock(side_effect=[
            _result_for("t", True, 1),
            _result_for("t", False, 2),
            _result_for("t", True, 3),
            _result_for("t", False, 4),
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        with pytest.raises(TaskNotCompletedError):
            await runner._run_task_with_retries(
                item=invocation, task_id=1, user_id="u", total_tasks=1,
            )
        assert single.await_count == 4
        assert runner._publish_task_artifacts.await_count == 2

    @pytest.mark.asyncio
    async def test_short_circuits_on_streak(self, runner, monkeypatch):
        # iterations=2, max_attempts=5: streak reached early, never calls remaining
        # iterations.
        invocation = _make_invocation(iterations=2, max_attempts=5)
        single = AsyncMock(side_effect=[
            _result_for("t", True, 1),
            _result_for("t", True, 2),
            _result_for("t", True, 3),  # should never be called
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert single.await_count == 2


# ─── TaskRunner retry loop: per-attempt timeout ──────────────────────────────


class TestRetryLoopTimeout:
    @pytest.mark.asyncio
    async def test_timeout_consumes_attempt_then_succeeds(self, runner, monkeypatch):
        # Attempt 1 hangs past timeout_s, attempt 2 succeeds.
        invocation = _make_invocation(iterations=1, max_attempts=2, timeout_s=0.05)
        calls = {"n": 0}

        async def single(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(5)
            return _result_for("t", True, calls["n"])

        monkeypatch.setattr(runner, "_run_single_iteration", single)

        result = await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert result.status == "done"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_all_attempts_time_out_raises(self, runner, monkeypatch):
        invocation = _make_invocation(iterations=1, max_attempts=2, timeout_s=0.05)

        async def single(**kwargs):
            await asyncio.sleep(5)
            return _result_for("t", True, 1)

        monkeypatch.setattr(runner, "_run_single_iteration", single)

        with pytest.raises(TaskNotCompletedError) as exc_info:
            await runner._run_task_with_retries(
                item=invocation, task_id=1, user_id="u", total_tasks=1,
            )
        assert "TimeoutError" in (exc_info.value.last_error or "") or isinstance(
            exc_info.value.__cause__, TimeoutError
        )

    @pytest.mark.asyncio
    async def test_no_timeout_keeps_attempts_unbounded(self, runner, monkeypatch):
        # timeout_s=None must not wrap the attempt at all.
        invocation = _make_invocation(iterations=1, max_attempts=1, timeout_s=None)
        single = AsyncMock(side_effect=[_result_for("t", True, 1)])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        result = await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert result.status == "done"


# ─── TaskRunner retry loop: exceptions ───────────────────────────────────────


class TestRetryLoopExceptions:
    @pytest.mark.asyncio
    async def test_exception_consumes_attempt_then_succeeds(
        self, runner, monkeypatch
    ):
        # A transient exception on attempt 1 must count as a failed attempt,
        # not abort the run — attempt 2 succeeds.
        invocation = _make_invocation(iterations=1, max_attempts=2)
        single = AsyncMock(side_effect=[
            RuntimeError("transient LLM error"),
            _result_for("t", True, 2),
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        result = await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )
        assert result.status == "done"
        assert single.await_count == 2
        runner._publish_task_artifacts.assert_awaited_once()

        # The exception attempt emits the same ITERATION_RESULT event as a
        # status!=DONE failure, extended with error info.
        failures = [
            c for c in runner._emit.await_args_list
            if c.args
            and c.args[0] is EventType.ITERATION_RESULT
            and c.kwargs.get("completed") is False
        ]
        assert len(failures) == 1
        assert failures[0].kwargs["error_type"] == "RuntimeError"
        assert "transient LLM error" in failures[0].kwargs["error_message"]
        assert failures[0].kwargs["successful_runs"] == 0
        assert failures[0].kwargs["iteration"] == 1

    @pytest.mark.asyncio
    async def test_exhaustion_by_exceptions_chains_last_exception(
        self, runner, monkeypatch
    ):
        invocation = _make_invocation(iterations=1, max_attempts=2)
        single = AsyncMock(side_effect=[
            RuntimeError("boom-1"),
            RuntimeError("boom-2"),
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        with pytest.raises(TaskNotCompletedError) as exc_info:
            await runner._run_task_with_retries(
                item=invocation, task_id=1, user_id="u", total_tasks=1,
            )
        assert single.await_count == 2
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "boom-2"
        assert "boom-2" in str(exc_info.value)
        assert exc_info.value.last_error == "boom-2"
        runner._publish_task_artifacts.assert_not_awaited()

        failed = [
            c for c in runner._emit.await_args_list
            if c.args and c.args[0] is EventType.TASK_FAILED
        ]
        assert len(failed) == 1
        assert failed[0].kwargs["last_error"] == "boom-2"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_without_consuming_attempts(
        self, runner, monkeypatch
    ):
        invocation = _make_invocation(iterations=1, max_attempts=3)
        single = AsyncMock(side_effect=asyncio.CancelledError())
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        with pytest.raises(asyncio.CancelledError):
            await runner._run_task_with_retries(
                item=invocation, task_id=1, user_id="u", total_tasks=1,
            )
        # No retries: cancellation unwinds immediately.
        assert single.await_count == 1
        types = [c.args[0] for c in runner._emit.await_args_list if c.args]
        assert EventType.ITERATION_RESULT not in types
        assert EventType.TASK_FAILED not in types


# ─── Missing declared input artifacts ────────────────────────────────────────


class TestMissingDeclaredArtifacts:
    def _runner(self) -> TaskRunner:
        return TaskRunner(
            name="test", artifact_service=MagicMock(spec=BaseArtifactService),
        )

    @pytest.mark.asyncio
    async def test_missing_artifact_warns_and_substitutes_empty(self, caplog):
        r = self._runner()
        r.artifact_service.load_artifact = AsyncMock(return_value=None)

        with caplog.at_level(
            logging.WARNING, logger="contractor.runners.task_runner",
        ):
            loaded = await r._load_artifacts(
                "u", ["up/result"], task_ref="task-a",
            )

        # Empty-string substitution is preserved (seeds may legitimately come
        # from the persistent store) — but it is no longer silent.
        assert loaded == {"up/result": ""}
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("task-a" in m and "up/result" in m for m in messages)

        # The empty text still renders cleanly via {artifact__*} substitution.
        template = _make_template(
            default_artifacts=["up/result"],
            instructions="ctx: [{artifact__up__result}]",
        )
        rendered = r._render_task(template, {}, loaded)
        assert "ctx: []" in rendered.instructions

    @pytest.mark.asyncio
    async def test_present_artifact_loads_without_warning(self, caplog):
        r = self._runner()
        part = SimpleNamespace(text="content", inline_data=None)
        r.artifact_service.load_artifact = AsyncMock(return_value=part)

        with caplog.at_level(
            logging.WARNING, logger="contractor.runners.task_runner",
        ):
            loaded = await r._load_artifacts(
                "u", ["up/result"], task_ref="task-a",
            )

        assert loaded == {"up/result": "content"}
        assert not caplog.records


# ─── Carry-state hygiene across attempts ─────────────────────────────────────


class TestCarryStateStripsStaleInvocationKeys:
    def _runner(self) -> TaskRunner:
        return TaskRunner(
            name="test", artifact_service=MagicMock(spec=BaseArtifactService),
        )

    def _rendered(self) -> RenderedTask:
        return RenderedTask(
            key="t", title="T", objective="obj", instructions="",
            output_format="", format="json",
        )

    def test_second_attempt_state_has_no_first_attempt_invocation_keys(self):
        # StreamlineManager keys its planner-internal state per ADK
        # invocation (task::{gid}::{invocation_id}::…). A retry gets a fresh
        # invocation_id, so the first attempt's keys are unreachable — they
        # must not be dragged into the next attempt's initial state.
        r = self._runner()
        carry = {
            "task::1::e-first::planner::tasks": [{"task_id": "0"}],
            "task::1::e-first::planner::idx": 0,
        }

        state = r._build_task_initial_state(
            task_id=1, task=self._rendered(), carry_state=carry,
        )

        assert not any("e-first" in key for key in state)

    def test_other_namespaces_and_plain_keys_survive(self):
        # Only THIS task's invocation-scoped keys are stripped: another
        # task's keys and non-task keys must pass through untouched, and the
        # task's own fixed keys are rebuilt by build_active_state.
        r = self._runner()
        carry = {
            "task::1::e-first::planner::tasks": ["stale"],
            "task::2::e-other::planner::tasks": ["other-task"],
            "task::1::status": "running",
            "callbacks": {"tokens": 5},
        }

        state = r._build_task_initial_state(
            task_id=1, task=self._rendered(), carry_state=carry,
        )

        assert "task::1::e-first::planner::tasks" not in state
        assert state["task::2::e-other::planner::tasks"] == ["other-task"]
        assert state["callbacks"] == {"tokens": 5}
        assert state["task::1::status"] == TaskStatus.RUNNING
        assert state["task::1::objective"] == "obj"


# ─── Skills: fail-fast validation + once-per-task injection ──────────────────


class TestSkillValidationAndInjection:
    def _runner_with_template(self, monkeypatch, template) -> TaskRunner:
        r = TaskRunner(
            name="test", artifact_service=MagicMock(spec=BaseArtifactService),
        )
        monkeypatch.setattr(r, "_ensure_template", MagicMock(return_value=template))
        return r

    def test_add_task_rejects_unknown_skill(self, monkeypatch):
        # A typo'd skill must fail at queue time, not when the task's first
        # iteration starts hours into a workflow.
        template = _make_template()
        r = self._runner_with_template(monkeypatch, template)

        with pytest.raises(ValueError, match="definitely_not_a_skill"):
            r.add_task(
                "t",
                worker_builder=lambda **_: MagicMock(),
                skills=["definitely_not_a_skill"],
            )
        assert r.queue == []

    def test_add_task_validates_template_default_skills(
        self, tmp_path, monkeypatch,
    ):
        import contractor.runners.skills as skills_mod

        monkeypatch.setattr(skills_mod, "SKILLS_BASE_DIR", tmp_path)
        (tmp_path / "good").mkdir()

        template = TaskTemplate(
            key="t", version="v1", title="T", objective="",
            instructions="", output_format="",
            default_skills=["good", "bad"],
        )
        r = self._runner_with_template(monkeypatch, template)

        with pytest.raises(ValueError, match="'bad'"):
            r.add_task("t", worker_builder=lambda **_: MagicMock())

    def test_add_task_accepts_existing_skill(self, tmp_path, monkeypatch):
        import contractor.runners.skills as skills_mod

        monkeypatch.setattr(skills_mod, "SKILLS_BASE_DIR", tmp_path)
        (tmp_path / "good").mkdir()

        template = _make_template()
        r = self._runner_with_template(monkeypatch, template)

        r.add_task(
            "t", worker_builder=lambda **_: MagicMock(), skills=["good"],
        )
        assert r.queue[0].skills == ["good"]

    @pytest.mark.asyncio
    async def test_injection_happens_once_across_attempts(
        self, runner, monkeypatch,
    ):
        # Skills + input artifacts are invariant across attempts; the
        # read-modify-write memory injection must run once per task, not
        # once per retry.
        inject_skills_spy = AsyncMock()
        inject_artifacts_spy = AsyncMock()
        monkeypatch.setattr(runner, "_inject_skills", inject_skills_spy)
        monkeypatch.setattr(runner, "_inject_artifacts", inject_artifacts_spy)

        invocation = _make_invocation(iterations=1, max_attempts=3)
        invocation.skills = ["some_skill"]
        single = AsyncMock(side_effect=[
            _result_for("t", False, 1),
            _result_for("t", False, 2),
            _result_for("t", True, 3),
        ])
        monkeypatch.setattr(runner, "_run_single_iteration", single)

        await runner._run_task_with_retries(
            item=invocation, task_id=1, user_id="u", total_tasks=1,
        )

        assert single.await_count == 3
        inject_skills_spy.assert_awaited_once()
        inject_artifacts_spy.assert_awaited_once()
        assert inject_skills_spy.await_args.kwargs["skills"] == ["some_skill"]


# ─── add_task artifacts override ─────────────────────────────────────────────


class TestAddTaskArtifactsOverride:
    def _runner_with_template(self, monkeypatch, template) -> TaskRunner:
        r = TaskRunner(
            name="test", artifact_service=MagicMock(spec=BaseArtifactService),
        )
        monkeypatch.setattr(r, "_ensure_template", MagicMock(return_value=template))
        return r

    def test_empty_list_overrides_template_defaults(self, monkeypatch):
        template = _make_template(default_artifacts=["up/result"])
        r = self._runner_with_template(monkeypatch, template)

        r.add_task("t", worker_builder=lambda **_: MagicMock(), artifacts=[])

        assert r.queue[0].artifacts == []

    def test_none_falls_back_to_template_defaults(self, monkeypatch):
        template = _make_template(default_artifacts=["up/result"])
        r = self._runner_with_template(monkeypatch, template)

        r.add_task("t", worker_builder=lambda **_: MagicMock())

        assert r.queue[0].artifacts == ["up/result"]


# ─── Per-invocation artifact keys (fan-out) ──────────────────────────────────


def _dict_artifact_service(store: dict[str, str]) -> MagicMock:
    """An artifact service backed by a plain dict, so publish/load round-trip."""

    async def save(*, app_name, user_id, session_id, filename, artifact):
        store[filename] = artifact.text

    async def load(*, app_name, user_id, session_id, filename):
        if filename in store:
            return SimpleNamespace(text=store[filename], inline_data=None)
        return None

    svc = MagicMock(spec=BaseArtifactService)
    svc.save_artifact = AsyncMock(side_effect=save)
    svc.load_artifact = AsyncMock(side_effect=load)
    return svc


def _fanout_runner(store, monkeypatch, **kwargs) -> TaskRunner:
    """TaskRunner with a dict-backed artifact service and a REAL
    ``_publish_task_artifacts``, so per-key publishing is exercised."""
    r = TaskRunner(
        name="test", artifact_service=_dict_artifact_service(store), **kwargs,
    )
    template = _make_template(default_iterations=1)
    r.templates[("t", "v1")] = template

    rendered = RenderedTask(
        key="t", title="T", objective="", instructions="",
        output_format="", format="json",
    )
    monkeypatch.setattr(r, "_ensure_template", MagicMock(return_value=template))
    monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    monkeypatch.setattr(r, "_emit", AsyncMock())
    return r


def _completed_result(text: str, *, task_id: int) -> TaskResult:
    result = _result_for("t", True, task_id, task_id=task_id)
    result.result = text
    return result


class TestPerInvocationArtifactKeys:
    @pytest.mark.asyncio
    async def test_distinct_artifact_keys_publish_non_colliding(self, monkeypatch):
        # Two invocations of the same template with distinct artifact_keys
        # must not overwrite each other's published artifacts.
        store: dict[str, str] = {}
        r = _fanout_runner(store, monkeypatch)

        r.add_task(
            "t", ref="t:f1", artifact_key="t/f1",
            worker_builder=lambda **_: MagicMock(),
        )
        r.add_task(
            "t", ref="t:f2", artifact_key="t/f2",
            worker_builder=lambda **_: MagicMock(),
        )
        single = AsyncMock(side_effect=[
            _completed_result("R1", task_id=0),
            _completed_result("R2", task_id=1),
        ])
        monkeypatch.setattr(r, "_run_single_iteration", single)

        results = await r.run(user_id="u")

        assert len(results) == 2
        assert store["t/f1/result"] == "R1"
        assert store["t/f2/result"] == "R2"
        assert "t/result" not in store

    def test_build_iteration_result_publishes_under_effective_key(
        self, monkeypatch,
    ):
        # The result's published_artifacts (what checkpoints record and events
        # report) must follow the invocation's artifact_key, defaulting to the
        # template key when unset.
        r = _fanout_runner({}, monkeypatch)
        rendered = RenderedTask(
            key="t", title="T", objective="", instructions="",
            output_format="", format="json",
        )
        keyed = _make_invocation(iterations=1, max_attempts=1)
        keyed.artifact_key = "t/f1"
        result = r._build_iteration_result(keyed, rendered, 0, "s", "", {}, {})
        assert result.published_artifacts == artifact_names_for_key("t/f1")

        plain = _make_invocation(iterations=1, max_attempts=1)
        result = r._build_iteration_result(plain, rendered, 0, "s", "", {}, {})
        assert result.published_artifacts == artifact_names_for_key("t")

    @pytest.mark.asyncio
    async def test_default_key_is_template_key(self, monkeypatch):
        # No artifact_key → unchanged behavior: publish under the template key.
        store: dict[str, str] = {}
        r = _fanout_runner(store, monkeypatch)

        r.add_task("t", ref="t:0", worker_builder=lambda **_: MagicMock())
        single = AsyncMock(return_value=_completed_result("R", task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        results = await r.run(user_id="u")

        assert len(results) == 1
        assert r.queue[0].artifact_key is None
        assert r.queue[0].effective_artifact_key == "t"
        assert store["t/result"] == "R"

    def test_invalid_artifact_key_rejected(self, monkeypatch):
        r = _fanout_runner({}, monkeypatch)
        with pytest.raises(InvalidArtifactKeyError):
            r.add_task(
                "t", ref="t:0", artifact_key="../escape",
                worker_builder=lambda **_: MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_restore_validates_own_key_not_siblings(
        self, tmp_path, monkeypatch,
    ):
        # Both siblings are checkpointed as done, but only f2's artifacts (and
        # a stale shared-key artifact) survive in the store. f1 must re-run —
        # neither the sibling's artifacts nor the shared key may validate its
        # restore — while f2 restores from its own key.
        cp = Checkpoint(workflow="test")
        for ref in ("t:f1", "t:f2"):
            cp.mark_done(CheckpointEntry(
                task_id=0, ref=ref, template_key="t", template_version="v1",
                published_artifacts=dict(artifact_names_for_key("t")),
            ))
        cp.save(tmp_path / "checkpoint.json")

        store = {
            "t/result": "stale shared", "t/summary": "", "t/records": "[]",
            "t/f2/result": "R2", "t/f2/summary": "", "t/f2/records": "[]",
        }
        r = _fanout_runner(
            store, monkeypatch, checkpoint_path=tmp_path / "checkpoint.json",
        )

        r.add_task(
            "t", ref="t:f1", artifact_key="t/f1",
            worker_builder=lambda **_: MagicMock(),
        )
        r.add_task(
            "t", ref="t:f2", artifact_key="t/f2",
            worker_builder=lambda **_: MagicMock(),
        )
        single = AsyncMock(return_value=_completed_result("R1", task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        results = await r.run(user_id="u")

        # f1 re-ran (own artifacts were missing) and re-published under its key.
        single.assert_awaited_once()
        assert store["t/f1/result"] == "R1"
        # f2 restored from checkpoint against its own artifacts.
        assert results[1].result == "(restored from checkpoint)"
        assert results[1].published_artifacts == artifact_names_for_key("t/f2")


# ─── Checkpoint integration ─────────────────────────────────────────────────


def _checkpoint_runner(tmp_path, monkeypatch) -> TaskRunner:
    """TaskRunner wired for checkpoint tests."""
    r = TaskRunner(
        name="test",
        artifact_service=MagicMock(spec=BaseArtifactService),
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    template = _make_template(default_iterations=1)
    r.templates[("t", "v1")] = template

    rendered = RenderedTask(
        key="t", title="T", objective="", instructions="",
        output_format="", format="json",
    )
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())
    monkeypatch.setattr(r, "_emit", AsyncMock())
    return r


class TestCheckpointIntegration:
    @pytest.mark.asyncio
    async def test_checkpoint_written_after_task_success(self, tmp_path, monkeypatch):
        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        await r.run(user_id="u")

        cp = Checkpoint.load(tmp_path / "checkpoint.json")
        assert cp is not None
        assert cp.get("a:0") is not None

    @pytest.mark.asyncio
    async def test_checkpoint_skips_completed_task(self, tmp_path, monkeypatch):
        # Seed a checkpoint with task "a:0" already done.
        cp = Checkpoint(workflow="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="t", template_version="v1",
            published_artifacts={"result": "t/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        # _load_artifact_text must return non-empty to confirm artifact exists
        monkeypatch.setattr(
            r, "_load_artifact_text", AsyncMock(return_value="content"),
        )
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        results = await r.run(user_id="u")

        # Task should be skipped — _run_single_iteration never called.
        single.assert_not_awaited()
        assert len(results) == 1
        assert results[0].status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_checkpoint_reruns_if_artifact_missing(self, tmp_path, monkeypatch):
        # Checkpoint says done, but artifact returns None → re-run.
        cp = Checkpoint(workflow="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="t", template_version="v1",
            published_artifacts={"result": "t/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        r.artifact_service.load_artifact = AsyncMock(return_value=None)
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        await r.run(user_id="u")

        # Artifact missing → task was re-executed.
        single.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_restore_skipped_on_template_key_mismatch(
        self, tmp_path, monkeypatch, caplog,
    ):
        # Entry was recorded for a different template — a stale checkpoint
        # after a workflow edit must not silently skip the task.
        cp = Checkpoint(workflow="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="other", template_version="v1",
            published_artifacts={"result": "other/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        with caplog.at_level(
            logging.WARNING, logger="contractor.runners.task_runner",
        ):
            await r.run(user_id="u")

        # Mismatch → re-run, with a warning naming both templates.
        single.assert_awaited_once()
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(
            "a:0" in m and "other@v1" in m and "t@v1" in m for m in messages
        )

    @pytest.mark.asyncio
    async def test_restore_skipped_on_template_version_mismatch(
        self, tmp_path, monkeypatch, caplog,
    ):
        cp = Checkpoint(workflow="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="t", template_version="v0",
            published_artifacts={"result": "t/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        with caplog.at_level(
            logging.WARNING, logger="contractor.runners.task_runner",
        ):
            await r.run(user_id="u")

        single.assert_awaited_once()
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("a:0" in m and "t@v0" in m and "t@v1" in m for m in messages)

    @pytest.mark.asyncio
    async def test_restore_still_happens_on_template_match(
        self, tmp_path, monkeypatch,
    ):
        # Sanity companion to the mismatch tests: same template_key AND
        # template_version → restore proceeds, task is skipped.
        cp = Checkpoint(workflow="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="t", template_version="v1",
            published_artifacts={"result": "t/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        results = await r.run(user_id="u")

        single.assert_not_awaited()
        assert results[0].result == "(restored from checkpoint)"

    @pytest.mark.asyncio
    async def test_load_checkpoint_failure_still_clears_handler(
        self, tmp_path, monkeypatch,
    ):
        # _load_checkpoint runs inside the try block, so the finally that
        # clears _on_event must run even if it raises.
        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(
            r, "_load_checkpoint",
            MagicMock(side_effect=RuntimeError("checkpoint exploded")),
        )

        async def on_event(event):
            pass

        with pytest.raises(RuntimeError, match="checkpoint exploded"):
            await r.run(user_id="u", on_event=on_event)

        assert r._on_event is None

    @pytest.mark.asyncio
    async def test_no_checkpoint_path_runs_normally(self, tmp_path, monkeypatch):
        r = TaskRunner(
            name="test",
            artifact_service=MagicMock(spec=BaseArtifactService),
        )
        template = _make_template(default_iterations=1)
        r.templates[("t", "v1")] = template

        rendered = RenderedTask(
            key="t", title="T", objective="", instructions="",
            output_format="", format="json",
        )
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))
        monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
        monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())
        monkeypatch.setattr(r, "_emit", AsyncMock())

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        results = await r.run(user_id="u")

        assert len(results) == 1
        single.assert_awaited_once()
        assert not (tmp_path / "checkpoint.json").exists()


# ─── Run lifecycle events ────────────────────────────────────────────────────


def _emitted_event_types(emit_mock) -> list:
    return [call.args[0] for call in emit_mock.await_args_list if call.args]


class TestRunLifecycleEvents:
    @pytest.mark.asyncio
    async def test_emits_run_finished_ok_on_success(self, runner, monkeypatch):
        monkeypatch.setattr(
            runner, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", True, 1, task_id=0)),
        )
        runner.queue.append(_make_invocation(ref="a:0", iterations=1, max_attempts=1))

        await runner.run(user_id="u")

        types = _emitted_event_types(runner._emit)
        assert EventType.RUN_STARTED in types
        assert EventType.RUN_FINISHED in types
        finished = next(
            c for c in runner._emit.await_args_list
            if c.args and c.args[0] is EventType.RUN_FINISHED
        )
        assert finished.kwargs["ok"] is True
        assert finished.kwargs["completed_tasks"] == 1

    @pytest.mark.asyncio
    async def test_emits_run_finished_not_ok_on_failure(self, runner, monkeypatch):
        # All attempts fail → TaskNotCompletedError propagates, but RUN_FINISHED
        # must still fire with ok=False so consumers can finalize.
        monkeypatch.setattr(
            runner, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", False, 1, task_id=0)),
        )
        runner.queue.append(_make_invocation(ref="a:0", iterations=1, max_attempts=1))

        with pytest.raises(TaskNotCompletedError):
            await runner.run(user_id="u")

        finished = [
            c for c in runner._emit.await_args_list
            if c.args and c.args[0] is EventType.RUN_FINISHED
        ]
        assert len(finished) == 1
        assert finished[0].kwargs["ok"] is False


# ─── Event handler failures (best-effort telemetry) ──────────────────────────


def _real_emit_runner(monkeypatch) -> TaskRunner:
    """A TaskRunner with I/O stubbed but a REAL _emit, so handler failures
    flow through the production guard."""
    r = TaskRunner(name="test", artifact_service=MagicMock(spec=BaseArtifactService))
    r.templates[("t", "v1")] = _make_template(default_iterations=1)
    rendered = RenderedTask(
        key="t", title="T", objective="", instructions="",
        output_format="", format="json",
    )
    monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())
    return r


class TestEmitHandlerFailures:
    @pytest.mark.asyncio
    async def test_raising_handler_does_not_abort_run(self, monkeypatch, caplog):
        # An observability failure (e.g. MetricsSink hitting a full disk) is
        # best-effort telemetry — the workflow must complete anyway.
        r = _real_emit_runner(monkeypatch)
        monkeypatch.setattr(
            r, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", True, 1, task_id=0)),
        )
        r.queue.append(_make_invocation(ref="a:0", iterations=1, max_attempts=1))

        async def bad_handler(event):
            raise OSError("No space left on device")

        with caplog.at_level(
            logging.ERROR, logger="contractor.runners.task_runner",
        ):
            results = await r.run(user_id="u", on_event=bad_handler)

        assert len(results) == 1
        assert results[0].status == "done"
        assert any(
            "event handler failed" in rec.getMessage() for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_from_handler_propagates(self, monkeypatch):
        # CancelledError must never be swallowed by the emit guard.
        r = _real_emit_runner(monkeypatch)
        monkeypatch.setattr(
            r, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", True, 1, task_id=0)),
        )
        r.queue.append(_make_invocation(ref="a:0", iterations=1, max_attempts=1))

        async def cancelling_handler(event):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await r.run(user_id="u", on_event=cancelling_handler)
        assert r._on_event is None


# ─── Per-task event payloads (characterization) ──────────────────────────────


def _capturing_runner(monkeypatch):
    """A TaskRunner with I/O stubbed but a REAL _emit feeding a capture list,
    so tests can assert the exact event sequence + payload fields."""
    r = TaskRunner(name="test", artifact_service=MagicMock(spec=BaseArtifactService))
    r.templates[("t", "v1")] = _make_template(default_iterations=1)
    rendered = RenderedTask(
        key="t", title="T", objective="", instructions="",
        output_format="", format="json",
    )
    monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())

    events: list = []

    async def on_event(ev):
        events.append(ev)

    r._on_event = on_event
    return r, events


class TestTaskEventPayloads:
    @pytest.mark.asyncio
    async def test_success_emits_started_iteration_finished(self, monkeypatch):
        r, events = _capturing_runner(monkeypatch)
        monkeypatch.setattr(
            r, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", True, 1, task_id=1)),
        )
        inv = _make_invocation(iterations=1, max_attempts=1)

        await r._run_task_with_retries(
            item=inv, task_id=1, user_id="u", total_tasks=3,
        )

        assert [e.type for e in events] == [
            EventType.TASK_STARTED,
            EventType.ITERATION_RESULT,
            EventType.TASK_FINISHED,
        ]
        # Common scoped fields must appear on every per-task event.
        for e in events:
            assert e.task_name == inv.ref
            assert e.task_id == 1
            assert e.payload["template_key"] == inv.template_key
            assert e.payload["template_version"] == inv.template_version
            assert e.payload["total_tasks"] == 3
            assert e.payload["completed_tasks"] == 1
        assert events[-1].payload["status"] == "done"
        assert events[0].payload["task_title"] == "T"

    @pytest.mark.asyncio
    async def test_failure_emits_task_failed_after_attempts(self, monkeypatch):
        r, events = _capturing_runner(monkeypatch)
        monkeypatch.setattr(
            r, "_run_single_iteration",
            AsyncMock(return_value=_result_for("t", False, 1, task_id=1)),
        )
        inv = _make_invocation(iterations=1, max_attempts=2)

        with pytest.raises(TaskNotCompletedError):
            await r._run_task_with_retries(
                item=inv, task_id=1, user_id="u", total_tasks=1,
            )

        assert [e.type for e in events] == [
            EventType.TASK_STARTED,
            EventType.ITERATION_RESULT,
            EventType.ITERATION_RESULT,
            EventType.TASK_FAILED,
        ]
        failed = events[-1]
        assert failed.payload["max_attempts"] == 2
        assert failed.payload["template_version"] == inv.template_version
        assert failed.payload["total_tasks"] == 1
