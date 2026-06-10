"""Probe: does ADK's ``after_run_callback`` fire during a TaskRunner run?

``TaskRunner.run``'s teardown comment and ``SandboxCleanupPlugin``'s module
docstring used to contradict each other on this. This test drives a REAL
ADK ``Runner`` (no LLM — the agent is a ``BaseAgent`` that yields one final
event) through ``TaskRunner`` with a probe plugin registered, and settles
the question.

Finding (ADK 2.x): ``Runner._exec_with_plugin`` awaits
``run_after_run_callback`` after the run's event generator is exhausted —
so it DOES fire whenever the outer run is consumed to completion (which
``TaskRunner._consume_events`` always does on the happy path). It does
NOT fire when the run raises or is abandoned mid-stream, which is why
``TaskRunner.run`` keeps a run()-level sandbox sweep as a backstop.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event, EventActions
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from contractor.runners.models import (
    RenderedTask,
    TaskInvocation,
    TaskScopedKeys,
    TaskStatus,
    TaskTemplate,
)
from contractor.runners.task_runner import TaskRunner

# ─── Probe fixtures ───────────────────────────────────────────────────────────


class ProbePlugin(BasePlugin):
    """Records every before/after run callback invocation."""

    def __init__(self) -> None:
        super().__init__(name="probe")
        self.before_run_ids: list[str | None] = []
        self.after_run_ids: list[str | None] = []

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        self.before_run_ids.append(
            getattr(invocation_context, "invocation_id", None)
        )

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        self.after_run_ids.append(
            getattr(invocation_context, "invocation_id", None)
        )


class OneShotAgent(BaseAgent):
    """No-LLM agent: yields a single final event that marks the task done."""

    async def _run_async_impl(self, ctx):
        keys = TaskScopedKeys(ctx.session.state.get("_global_task_id", 0))
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="all done")],
            ),
            actions=EventActions(state_delta={keys.status: TaskStatus.DONE}),
        )


def _probe_task_runner(monkeypatch, probe: ProbePlugin) -> TaskRunner:
    """A TaskRunner whose iteration drives a REAL ADK Runner.

    Only the LLM-bound pieces are replaced: the planner is swapped for
    ``OneShotAgent`` and the plugin list for the probe. Session/artifact
    services, state seeding, and event consumption are the production path.
    """
    r = TaskRunner(name="probe_app", artifact_service=InMemoryArtifactService())
    r.templates[("t", "v1")] = TaskTemplate(
        key="t", version="v1", title="T",
        objective="", instructions="", output_format="",
    )
    rendered = RenderedTask(
        key="t", title="T", objective="", instructions="",
        output_format="", format="json",
    )
    monkeypatch.setattr(r, "_render_task", MagicMock(return_value=rendered))
    # ADK's InMemoryArtifactService rejects the session-agnostic
    # (session_id=None) saves the production artifact service supports;
    # publishing is irrelevant to the probe, so stub it.
    monkeypatch.setattr(r, "_publish_task_artifacts", AsyncMock())
    monkeypatch.setattr(
        r, "_spawn_planning_agent",
        lambda item, task: OneShotAgent(name="probe_agent"),
    )
    monkeypatch.setattr(
        r, "_build_plugins", lambda *args, **kwargs: [probe],
    )
    return r


# ─── The probe ────────────────────────────────────────────────────────────────


class TestAfterRunCallbackFires:
    @pytest.mark.asyncio
    async def test_after_run_callback_fires_for_the_outer_task_run(
        self, monkeypatch,
    ):
        probe = ProbePlugin()
        r = _probe_task_runner(monkeypatch, probe)
        r.queue.append(TaskInvocation(
            id="inv-1",
            ref="probe_task",
            template_key="t",
            template_version="v1",
            worker_builder=lambda **_: MagicMock(),
            iterations=1,
            max_attempts=1,
        ))

        results = await r.run(user_id="u")

        assert len(results) == 1
        assert results[0].status == TaskStatus.DONE

        # The load-bearing assertion: after_run_callback DOES fire when the
        # outer run is consumed to completion, and it pairs 1:1 with
        # before_run_callback for the same invocation. SandboxCleanupPlugin's
        # root-invocation teardown relies on exactly this.
        assert len(probe.before_run_ids) == 1
        assert probe.after_run_ids == probe.before_run_ids
