"""Task-level eval harness.

Runs an entire ``TaskRunner`` queue (planner + worker chain, exactly the
shape pipelines under ``cli/pipelines/`` build) against a fixture and
captures the same analytics signal trace_agent's harness produces:
tool counts, token usage, tool exec time — keyed by task ref.

Unlike ``trace_harness.run_trace_agent`` (which drives a bare ``LlmAgent``
through ``Runner``), this harness goes through ``TaskRunner``: the
planner spawns subtasks, the worker executes them, and after each task
finishes the runner publishes ``{template_key}/result|summary|records``
artifacts. We read those artifacts back and return them next to the raw
events so callers can score the produced text however they like.

Metrics capture is automatic — ``TaskRunner._build_plugins`` already
attaches ``AdkMetricsPlugin`` / ``AdkTracePlugin`` whose emitted events
flow through ``self._emit`` into the ``on_event`` callback we register.
A single event list therefore contains both runner-lifecycle events
(``task_started``, ``iteration_finished``, ``global_task_finished``,
…) and per-tool / per-LLM metrics events (``tool_call``,
``tool_result``, ``llm_usage``, …).
"""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from google.adk.artifacts import FileArtifactService

from contractor.runners.agio import AgioEventType
from contractor.runners.artifacts import artifact_filename, save_result_artifacts
from contractor.runners.models import TaskResult, TaskRunnerEvent
from contractor.runners.task_runner import TaskRunner
from contractor.utils import observability

QueueFn = Callable[[TaskRunner], Awaitable[None] | None]


@dataclass
class TaskMetrics:
    """Per-task aggregate of plugin-emitted events.

    All values are summed across iterations and attempts of the same
    ``task_ref`` (so a task with ``iterations=3`` reports the union of
    its three worker invocations). ``tool_counts`` indexes by tool
    name; ``llm_calls`` counts ``llm_usage`` events.
    """

    task_ref: str
    tool_counts: Counter[str] = field(default_factory=Counter)
    total_tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_time_ms: float = 0.0
    args_bytes: int = 0
    result_bytes: int = 0


@dataclass
class TaskAgentRun:
    """End-to-end result of one ``TaskRunner.run()`` invocation."""

    results: list[TaskResult]
    events: list[TaskRunnerEvent]
    # ``{template_key}/result`` text for every task that published an
    # artifact, keyed by the *artifact filename* (e.g. ``"likec4_build/result"``).
    artifacts: dict[str, str]
    # Per-task analytics. Keyed by ``task_ref`` (e.g. ``"likec4_build:0"``).
    metrics: dict[str, TaskMetrics]

    def events_of(self, event_type: str) -> list[TaskRunnerEvent]:
        return [e for e in self.events if str(e.type) == event_type]

    def result_text(self, template_key: str) -> str:
        """Convenience: fetch the published ``{key}/result`` body."""
        return self.artifacts.get(artifact_filename(template_key, "result"), "")


def _aggregate_metrics(events: list[TaskRunnerEvent]) -> dict[str, TaskMetrics]:
    """Bucket plugin events by ``task_name`` (== task ref).

    Plugins emit through ``TaskRunner._emit`` which stamps each event
    with ``task_name=item.ref``; we use that as the bucket key so a
    pipeline with two ``swe_agent``-backed tasks doesn't collapse into
    one row.
    """
    buckets: dict[str, TaskMetrics] = {}

    def bucket(ref: str) -> TaskMetrics:
        if ref not in buckets:
            buckets[ref] = TaskMetrics(task_ref=ref)
        return buckets[ref]

    for ev in events:
        kind = str(ev.type)
        b = bucket(ev.task_name)
        p = ev.payload

        if kind == AgioEventType.TOOL_CALL:
            name = str(p.get("tool_name", ""))
            if name:
                b.tool_counts[name] += 1
                b.total_tool_calls += 1
            b.args_bytes += int(p.get("arguments_size", 0) or 0)
        elif kind == AgioEventType.TOOL_RESULT:
            # execution_time_ms / result_size land on TOOL_RESULT, not TOOL_CALL
            b.tool_time_ms += float(p.get("execution_time_ms", 0) or 0)
            b.result_bytes += int(p.get("result_size", 0) or 0)
        elif kind == AgioEventType.LLM_USAGE:
            usage = p.get("usage") or {}
            b.llm_calls += 1
            b.input_tokens += int(usage.get("input", 0) or 0)
            b.output_tokens += int(usage.get("output", 0) or 0)
            b.total_tokens += int(usage.get("total", 0) or 0)
    return buckets


async def run_task_pipeline(
    *,
    queue_fn: QueueFn,
    artifact_keys: list[str],
    namespace: str,
    timeout_s: float = 1800.0,
    user_id: str = "eval-user",
    runner_name: str = "eval-task-runner",
    observability_tags: Optional[list[str]] = None,
    preloaded_artifacts: Optional[dict[str, str]] = None,
) -> TaskAgentRun:
    """Build a ``TaskRunner``, hand it to ``queue_fn`` for population, run it,
    and return everything the caller needs to score and analyse the run.

    ``queue_fn`` is invoked with the ``TaskRunner`` instance after the
    artifact service is wired up — its job is to call ``runner.add_task(...)``
    one or more times with whatever ``worker_builder`` partials the test
    needs. Anything ``queue_fn`` returns is ignored; sync or async is fine.

    ``artifact_keys`` is the list of ``template_key/result`` filenames to
    load back after the run (defaults to a single task means a single
    artifact). We load each via the artifact service so callers don't
    have to reach into ``TaskRunner`` internals.

    ``namespace`` is used only for the Langfuse trace context; it does
    NOT override per-task ``namespace=`` passed to ``add_task`` (those
    keep the same isolation the production pipelines use).
    """
    import asyncio

    # ``InMemoryArtifactService`` rejects saves with ``session_id=None`` on
    # filenames that aren't ``user:``-prefixed (its strict mode requires
    # one or the other). The production CLI uses ``FileArtifactService``
    # which treats ``session_id=None`` as user-scoped — the same shape
    # ``TaskRunner._publish_task_artifacts`` relies on. Mirror that here
    # over a TemporaryDirectory so each eval call is hermetic.
    with tempfile.TemporaryDirectory(prefix="task-eval-artifacts-") as tmpdir:
        artifact_service = FileArtifactService(root_dir=tmpdir)
        runner = TaskRunner(name=runner_name, artifact_service=artifact_service)

        maybe_coro = queue_fn(runner)
        if maybe_coro is not None and hasattr(maybe_coro, "__await__"):
            await maybe_coro

        if preloaded_artifacts:
            for art_filename, text in preloaded_artifacts.items():
                key, _, kind = art_filename.rpartition("/")
                if kind == "result":
                    await save_result_artifacts(
                        artifact_service=artifact_service,
                        app_name=runner_name,
                        user_id=user_id,
                        key=key,
                        result=text,
                    )

        events: list[TaskRunnerEvent] = []

        async def _on_event(event: TaskRunnerEvent) -> None:
            events.append(event)

        tags = ["eval", "agent:task_runner", f"runner:{runner_name}"]
        if observability_tags:
            tags.extend(observability_tags)

        with observability.run_context(
            name=f"eval.task_runner.{runner_name}",
            session_id=namespace,
            tags=tags,
            metadata={
                "runner_name": runner_name,
                "namespace": namespace,
                "queued_refs": [item.ref for item in runner.queue],
            },
        ):
            results = await asyncio.wait_for(
                runner.run(user_id=user_id, on_event=_on_event),
                timeout=timeout_s,
            )

        artifacts: dict[str, str] = {}
        for key in artifact_keys:
            part = await artifact_service.load_artifact(
                app_name=runner_name,
                user_id=user_id,
                session_id=None,
                filename=key,
            )
            if part is not None and getattr(part, "text", None):
                artifacts[key] = part.text

    return TaskAgentRun(
        results=results,
        events=events,
        artifacts=artifacts,
        metrics=_aggregate_metrics(events),
    )


def render_metrics_table(metrics: dict[str, TaskMetrics]) -> str:
    """Pretty-print a per-task analytics table — used by the eval failure
    message and the ``compare_task_runs`` script."""
    headers = (
        "task_ref",
        "tools",
        "llm",
        "in_tok",
        "out_tok",
        "tool_ms",
    )
    rows = [headers] + [
        (
            m.task_ref,
            str(m.total_tool_calls),
            str(m.llm_calls),
            str(m.input_tokens),
            str(m.output_tokens),
            f"{m.tool_time_ms:.0f}",
        )
        for m in sorted(metrics.values(), key=lambda x: x.task_ref)
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    return "\n".join(fmt.format(*r) for r in rows)
