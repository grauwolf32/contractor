"""Integration tests for TaskRunner.

Unlike the unit tests (which stub _load_artifacts / _render_task / _publish /
_emit), these exercise the *real* artifact-load → render → publish → checkpoint
path against a real ADK InMemoryArtifactService. Only the LLM iteration itself
is replaced with a deterministic worker, so the cross-task artifact contract —
"tasks communicate only via artifacts" — is verified end to end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.artifacts import BaseArtifactService, FileArtifactService

from contractor.runners.models import (
    Checkpoint,
    TaskInvocation,
    TaskResult,
    TaskScopedKeys,
    TaskStatus,
    TaskTemplate,
)
from contractor.runners.task_runner import TaskNotCompletedError, TaskRunner


def _template(key: str) -> TaskTemplate:
    return TaskTemplate(
        key=key,
        version="v1",
        title=f"Template {key}",
        objective="do the thing",
        instructions="follow the plan",
        output_format="text",
        default_iterations=1,
    )


def _invocation(
    *,
    ref: str,
    template_key: str,
    artifacts: list[str] | None = None,
    iterations: int = 1,
    max_attempts: int = 1,
) -> TaskInvocation:
    return TaskInvocation(
        id=f"inv-{ref}",
        ref=ref,
        template_key=template_key,
        template_version="v1",
        worker_builder=lambda **_: MagicMock(),
        artifacts=artifacts or [],
        iterations=iterations,
        max_attempts=max_attempts,
    )


class _EchoRunner(TaskRunner):
    """TaskRunner whose iteration marks the task done and echoes the artifacts
    it was given — proving _load_artifacts fed it the upstream results.

    ``complete`` controls whether the synthetic worker succeeds. ``iteration_calls``
    counts real iterations so tests can assert a checkpoint-restored task was
    NOT re-executed.
    """

    complete: bool = True
    iteration_calls: int = 0

    async def _run_single_iteration(
        self,
        *,
        item: TaskInvocation,
        rendered_task: Any,
        input_artifacts: dict[str, str],
        task_id: int,
        user_id: str,
        carry_state: dict[str, Any],
        iteration: int,
    ) -> TaskResult:
        self.iteration_calls += 1
        keys = TaskScopedKeys(task_id)
        echoed = ";".join(f"{k}={v}" for k, v in sorted(input_artifacts.items()))
        status = TaskStatus.DONE if self.complete else TaskStatus.RUNNING
        final_state = {
            keys.status: status,
            keys.result: f"{item.template_key}-result|inputs:[{echoed}]",
            keys.summary: f"{item.template_key}-summary",
            keys.pool: [{"finding": item.template_key}],
        }
        return self._build_iteration_result(
            item,
            rendered_task,
            task_id,
            session_id=f"sess-{task_id}-{iteration}",
            final_text="ok",
            final_state=final_state,
            input_artifacts=input_artifacts,
        )


def _svc(root) -> FileArtifactService:
    """A real FileArtifactService rooted at ``root`` (the production service)."""
    root.mkdir(parents=True, exist_ok=True)
    return FileArtifactService(root_dir=str(root))


def _make_runner(artifact_service: BaseArtifactService, **kwargs) -> _EchoRunner:
    r = _EchoRunner(name="contractor", artifact_service=artifact_service, **kwargs)
    r.templates[("ta", "v1")] = _template("ta")
    r.templates[("tb", "v1")] = _template("tb")
    return r


async def _load_text(svc: BaseArtifactService, filename: str) -> str | None:
    part = await svc.load_artifact(
        app_name="contractor", user_id="u", session_id=None, filename=filename
    )
    return None if part is None else part.text


class TestArtifactFlow:
    @pytest.mark.asyncio
    async def test_upstream_result_reaches_downstream_task(self, tmp_path):
        svc = _svc(tmp_path / "artifacts")
        r = _make_runner(svc)
        r.queue.append(_invocation(ref="a", template_key="ta"))
        r.queue.append(
            _invocation(ref="b", template_key="tb", artifacts=["ta/result"])
        )

        results = await r.run(user_id="u")

        assert len(results) == 2
        # Task A really published its three artifacts via the real service.
        assert await _load_text(svc, "ta/result") == "ta-result|inputs:[]"
        assert await _load_text(svc, "ta/summary") == "ta-summary"
        # Task B's loaded inputs included A's published result — the only way
        # "ta/result=..." appears in B's output is if _load_artifacts fetched it.
        b_result = await _load_text(svc, "tb/result")
        assert "ta/result=ta-result|inputs:[]" in b_result

    @pytest.mark.asyncio
    async def test_records_artifact_serialised(self, tmp_path):
        svc = _svc(tmp_path / "artifacts")
        r = _make_runner(svc)
        r.queue.append(_invocation(ref="a", template_key="ta"))

        await r.run(user_id="u")

        records = await _load_text(svc, "ta/records")
        assert "ta" in records  # JSON list with the finding dict


class TestCheckpointIntegration:
    @pytest.mark.asyncio
    async def test_restore_skips_completed_task_without_rerunning(self, tmp_path):
        svc = _svc(tmp_path / "artifacts")
        cp = tmp_path / "cp.json"

        # First run: completes task A, writes checkpoint + artifacts.
        r1 = _make_runner(svc, checkpoint_path=cp)
        r1.queue.append(_invocation(ref="a", template_key="ta"))
        await r1.run(user_id="u")
        assert r1.iteration_calls == 1
        assert Checkpoint.load(cp).get("a") is not None

        # Second run with the same checkpoint + artifact service: task A is
        # restored (its artifacts still exist), so the worker never runs again.
        r2 = _make_runner(svc, checkpoint_path=cp)
        r2.queue.append(_invocation(ref="a", template_key="ta"))
        results = await r2.run(user_id="u")

        assert r2.iteration_calls == 0
        assert results[0].status == TaskStatus.DONE
        assert results[0].result == "(restored from checkpoint)"

    @pytest.mark.asyncio
    async def test_reruns_when_checkpointed_artifact_missing(self, tmp_path):
        svc = _svc(tmp_path / "artifacts")
        cp = tmp_path / "cp.json"

        r1 = _make_runner(svc, checkpoint_path=cp)
        r1.queue.append(_invocation(ref="a", template_key="ta"))
        await r1.run(user_id="u")

        # Fresh (empty) artifact store → checkpoint exists but artifacts are gone.
        r2 = _make_runner(_svc(tmp_path / "artifacts2"), checkpoint_path=cp)
        r2.queue.append(_invocation(ref="a", template_key="ta"))
        await r2.run(user_id="u")

        assert r2.iteration_calls == 1  # re-executed


class TestFailurePath:
    @pytest.mark.asyncio
    async def test_unfinished_task_raises_and_publishes_nothing(self, tmp_path):
        svc = _svc(tmp_path / "artifacts")
        cp = tmp_path / "cp.json"
        r = _make_runner(svc, checkpoint_path=cp)
        r.complete = False  # worker never reaches DONE
        r.queue.append(
            _invocation(ref="a", template_key="ta", iterations=1, max_attempts=2)
        )

        with pytest.raises(TaskNotCompletedError):
            await r.run(user_id="u")

        # Tried max_attempts times, published no artifact, wrote no checkpoint.
        assert r.iteration_calls == 2
        assert await _load_text(svc, "ta/result") is None
        assert Checkpoint.load(cp) is None or Checkpoint.load(cp).get("a") is None
