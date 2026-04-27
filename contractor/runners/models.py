from __future__ import annotations
import os
import re
import yaml

from pathlib import Path
from enum import StrEnum, unique
from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict, Literal, Callable, Awaitable
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool
from google.adk.agents import LlmAgent

# ─── Enums ────────────────────────────────────────────────────────────────────


@unique
class EventType(StrEnum):
    """All event types emitted by TaskRunner, in one discoverable place."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    GLOBAL_TASK_FINISHED = "global_task_finished"
    ITERATION_STARTED = "iteration_started"
    ITERATION_FINISHED = "iteration_finished"
    ITERATION_RESULT = "iteration_result"
    FINAL_TEXT = "final_text"


@unique
class TaskStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"


# ─── Data Structures ─────────────────────────────────────────────────────────


class TaskResult(TypedDict):
    """Strongly-typed dict returned from each iteration / task."""

    invocation_id: str
    task_ref: str
    task_key: str
    task_title: str
    template_key: str
    task_id: int
    session_id: str
    final_response: str
    state: dict[str, Any]
    carry_state: dict[str, Any]
    status: str | None
    result: Any
    summary: str | None
    records: list[Any]
    params: dict[str, Any]
    input_artifacts: dict[str, str]
    published_artifacts: dict[ArtifactKind, str]


@dataclass(slots=True, frozen=True)
class TaskRunnerEvent:
    type: EventType | str
    task_name: str
    task_id: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInvocation:
    id: str
    ref: str
    template_key: str
    worker_builder: WorkerBuilder

    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    iterations: int = 1
    max_attempts: int = 1
    max_steps: int = 15

    namespace: str | None = None
    model: LiteLlm | None = None

    def effective_namespace(self, fallback: str) -> str:
        return self.namespace or fallback

    def effective_model(self, fallback: LiteLlm) -> LiteLlm:
        return self.model or fallback


# ─── Key-name helpers ────────────────────────────────────────────────────────


class TaskScopedKeys:
    """
    Generates the task::{id}::* keys for a specific task_id.

    Usage:
        keys = TaskScopedKeys(task_id)
        state[keys.status] = TaskStatus.RUNNING
        state[keys.result] = "..."
    """

    def __init__(self, task_id: int) -> None:
        self._prefix = f"task::{task_id}"

    def _k(self, suffix: str) -> str:
        return f"{self._prefix}::{suffix}"

    # ── Task-scoped keys (written by TaskRunner, read by StreamlineManager) ──

    @property
    def objective(self) -> str:
        return self._k("objective")

    @property
    def status(self) -> str:
        return self._k("status")

    @property
    def current(self) -> str:
        """Current active subtask; None while not yet assigned."""
        return self._k("current")

    @property
    def result(self) -> str:
        """Final task result text; written by StreamlineManager.finish()."""
        return self._k("result")

    @property
    def summary(self) -> str:
        """Concise handoff summary; written by StreamlineManager.finish()."""
        return self._k("summary")

    @property
    def pool(self) -> str:
        """List of task record dicts; appended by StreamlineManager.save_record()."""
        return self._k("pool")


# ─── Global key constant ─────────────────────────────────────────────────────

GLOBAL_TASK_ID_KEY = "_global_task_id"


# ─── Session-state builder ───────────────────────────────────────────────────
#
# Per-iteration state is a flat dict keyed by:
#   _global_task_id    — sentinel read by StreamlineManager
#   task::{id}::*      — per-task live state, owned by planning agent


def build_active_state(*, task_id: int, task: RenderedTask) -> dict[str, Any]:
    """Initial flat state dict for a new task iteration."""
    keys = TaskScopedKeys(task_id)
    return {
        GLOBAL_TASK_ID_KEY: task_id,
        keys.objective: task.objective,
        keys.status: TaskStatus.RUNNING,
        keys.current: None,
        keys.result: "",
        keys.summary: "",
        keys.pool: [],
    }


# ─── Constants ────────────────────────────────────────────────────────────────

ArtifactKind = Literal["result", "summary", "records"]
WorkerBuilder = Callable[..., LlmAgent | AgentTool]
TaskRunnerEventHandler = Callable[[TaskRunnerEvent], Awaitable[None]]

# --- Task Models ---------------------------------------------------------------


TASKS_BASE_DIR = Path(__file__).parent.parent / "tasks"


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return normalized or "task"


def _artifact_var_name(artifact_ref: str) -> str:
    return "artifact__" + "__".join(
        _normalize_name(part) for part in artifact_ref.split("/") if part.strip()
    )


@dataclass(slots=True, frozen=True)
class TaskTemplate:
    key: str
    title: str
    objective: str
    instructions: str
    output_format: str
    default_artifacts: list[str] = field(default_factory=list)
    default_iterations: int = 1
    format: str = "json"

    @classmethod
    def load(cls, name: str) -> TaskTemplate:
        template_key = Path(name).stem
        fname = TASKS_BASE_DIR / f"{template_key}.yml"
        if not os.path.exists(fname):
            raise ValueError(f"Task template {template_key} not found")

        with open(fname, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw = data["task"]

        return cls(
            key=template_key,
            title=raw.get("name", template_key) or template_key,
            objective=raw["objective"],
            instructions=raw["instructions"],
            output_format=raw["output_format"],
            default_artifacts=list(raw.get("artifacts", []) or []),
            default_iterations=int(raw.get("iterations", 1) or 1),
            format=raw.get("format", "json") or "json",
        )


@dataclass(slots=True, frozen=True)
class RenderedTask:
    key: str
    title: str
    objective: str
    instructions: str
    output_format: str
    format: str
    artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_template(
        cls,
        template: TaskTemplate,
        *,
        variables: Mapping[str, Any],
        params: Mapping[str, Any],
        artifacts: Mapping[str, str],
    ) -> RenderedTask:
        scope: dict[str, Any] = dict(variables)
        scope.update(params)

        scope["artifacts"] = yaml.safe_dump(
            dict(artifacts),
            allow_unicode=True,
            sort_keys=False,
        )

        for artifact_ref, value in artifacts.items():
            scope[_artifact_var_name(artifact_ref)] = value

        return cls(
            key=template.key,
            title=template.title,
            objective=template.objective.format(**scope),
            instructions=template.instructions.format(**scope),
            output_format=template.output_format.format(**scope),
            artifacts=dict(artifacts),
            format=template.format,
        )

    def _format_artifacts(self) -> str:
        fmt: str = "artifacts from previous tasks, stored as memories:\n"
        for name in self.artifacts.keys():
            fmt += f"* {name}\n"
        return fmt

    def _format_task(self) -> str:
        task: str = (
            f"TASK:\n{self.title}\n\n"
            f"OBJECTIVE:\n{self.objective}\n\n"
            f"INSTRUCTIONS:\n{self.instructions}\n\n"
            f"OUTPUT FORMAT:\n{self.output_format}\n\n"
        )
        if self.artifacts:
            task += f"INBOX:\n{self._format_artifacts()}"
        return task
