from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.artifacts import BaseArtifactService

from contractor.runners.models import (
    Checkpoint,
    CheckpointEntry,
    RenderedTask,
    TaskInvocation,
    TaskResult,
    TaskStatus,
    TaskTemplate,
)
from contractor.runners._helpers import _decode_part_text, _extract_final_text
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

    def test_inline_bytes_invalid_utf8_ignores_errors(self):
        part = SimpleNamespace(
            text=None,
            inline_data=SimpleNamespace(data=b"\xff\xfehello"),
        )
        # `errors="ignore"` drops the invalid bytes; the readable suffix survives.
        assert _decode_part_text(part) == "hello"

    def test_missing_inline_data_returns_empty(self):
        part = SimpleNamespace(text=None, inline_data=None)
        assert _decode_part_text(part) == ""


# ─── TaskRunner._resolve_retry_params ────────────────────────────────────────


def _make_template(default_iterations=1) -> TaskTemplate:
    return TaskTemplate(
        key="t",
        version="v1",
        title="T",
        objective="",
        instructions="",
        output_format="",
        default_iterations=default_iterations,
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
) -> TaskInvocation:
    return TaskInvocation(
        id="inv-1",
        ref=ref,
        template_key=template_key,
        template_version=template_version,
        worker_builder=lambda **_: MagicMock(),
        iterations=iterations,
        max_attempts=max_attempts,
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
        assert result["status"] == "done"
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
        cp = Checkpoint(pipeline="test")
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
        assert results[0]["status"] == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_checkpoint_reruns_if_artifact_missing(self, tmp_path, monkeypatch):
        # Checkpoint says done, but artifact returns empty → re-run.
        cp = Checkpoint(pipeline="test")
        cp.mark_done(CheckpointEntry(
            task_id=0, ref="a:0", template_key="t", template_version="v1",
            published_artifacts={"result": "t/result"},
        ))
        cp.save(tmp_path / "checkpoint.json")

        r = _checkpoint_runner(tmp_path, monkeypatch)
        monkeypatch.setattr(
            r, "_load_artifact_text", AsyncMock(return_value=""),
        )
        monkeypatch.setattr(r, "_load_artifacts", AsyncMock(return_value={}))

        inv = _make_invocation(ref="a:0", iterations=1, max_attempts=1)
        single = AsyncMock(return_value=_result_for("t", True, 1, task_id=0))
        monkeypatch.setattr(r, "_run_single_iteration", single)

        r.queue.append(inv)
        await r.run(user_id="u")

        # Artifact missing → task was re-executed.
        single.assert_awaited_once()

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
