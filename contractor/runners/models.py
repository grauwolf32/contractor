from __future__ import annotations
import os
import re
import yaml
import copy

from pathlib import Path
from pydantic import BaseModel, Field
from enum import StrEnum, unique
from dataclasses import dataclass, field
from typing import Any, Optional, Mapping, TypedDict, Literal, Callable, Awaitable
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
    type: EventType
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


# ─── ActiveTaskState ─────────────────────────────────────────────────────────


class ActiveTaskState(BaseModel):
    """
    All state keys written at the START of each iteration.

    Covers both the runner:-namespaced keys (runner-wide bookkeeping)
    and the task::{id}::-namespaced keys (task-specific live state).
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    task_id: int = Field(
        description="Monotonically increasing task counter for this runner run."
    )

    # ── Runner-scoped: bookkeeping visible to all tasks in the session ────────

    last_task_id: int = Field(
        description="runner:last_task_id — same as task_id at task start; "
        "used by downstream tasks to find the predecessor."
    )
    last_task_key: str = Field(
        description="runner:last_task_key — template key of the currently active task."
    )
    last_task_title: str = Field(
        description="runner:last_task_title — human-readable title of the active task."
    )
    active_task_ref: str = Field(
        description="runner:active_task_ref — fully-qualified task ref (e.g. 'oas_update:1')."
    )
    active_template_key: str = Field(
        description="runner:active_template_key — template YAML stem (e.g. 'oas_update')."
    )
    iteration: int = Field(
        description="runner:iteration — 1-based iteration counter within max_attempts."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="runner:params — task parameters rendered into the objective template.",
    )
    input_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="runner:input_artifacts — {artifact_name: text_content} loaded for this task.",
    )

    # ── Task-scoped: live state written/updated by the planning agent ─────────

    objective: str = Field(
        description="task::{id}::objective — rendered objective string sent to the agent."
    )
    status: str = Field(
        default=TaskStatus.RUNNING,
        description="task::{id}::status — TaskStatus.RUNNING at start; DONE when finished.",
    )
    current: Optional[Any] = Field(
        default=None,
        description="task::{id}::current — active subtask; None until first subtask is dispatched.",
    )
    result: str = Field(
        default="",
        description="task::{id}::result — final result text; empty until StreamlineManager.finish().",
    )
    summary: str = Field(
        default="",
        description="task::{id}::summary — concise handoff text; empty until finish().",
    )
    pool: list[Any] = Field(
        default_factory=list,
        description="task::{id}::pool — record accumulator; appended by save_record().",
    )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_invocation(
        cls,
        task_id: int,
        task: RenderedTask,
        item: TaskInvocation,
        iteration: int,
        input_artifacts: dict[str, str],
    ) -> "ActiveTaskState":
        """Construct from the objects already available in _build_task_initial_state."""
        return cls(
            task_id=task_id,
            last_task_id=task_id,
            last_task_key=task.key,
            last_task_title=task.title,
            active_task_ref=item.ref,
            active_template_key=item.template_key,
            iteration=iteration,
            params=copy.deepcopy(item.params),
            input_artifacts=copy.deepcopy(input_artifacts),
            objective=task.objective,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_session_dict(self) -> dict[str, Any]:
        """
        Produce the flat string-keyed dict that the ADK session service stores.
        Call result is merged into the carry_state dict, overwriting any stale keys.
        """
        keys = TaskScopedKeys(self.task_id)
        return {
            # Global task ID sentinel (read by StreamlineManager._global_keys)
            GLOBAL_TASK_ID_KEY: self.task_id,
            # Runner-scoped
            "runner:last_task_id": self.last_task_id,
            "runner:last_task_key": self.last_task_key,
            "runner:last_task_title": self.last_task_title,
            "runner:active_task_ref": self.active_task_ref,
            "runner:active_template_key": self.active_template_key,
            "runner:iteration": self.iteration,
            "runner:params": copy.deepcopy(self.params),
            "runner:input_artifacts": copy.deepcopy(self.input_artifacts),
            # Task-scoped
            keys.objective: self.objective,
            keys.status: self.status,
            keys.current: self.current,
            keys.result: self.result,
            keys.summary: self.summary,
            keys.pool: list(self.pool),
        }


# ─── CarryState ──────────────────────────────────────────────────────────────


class CarryState(BaseModel):
    """
    The slice of runner:-namespaced state forwarded to the NEXT task.

    Written by _extract_carry_state after a task finishes so that the
    subsequent task can read its predecessor's outcome without knowing
    the predecessor's task_id.
    """

    previous_task_id: int = Field(
        description="runner:previous_task_id — task_id of the just-finished task."
    )
    previous_task_status: Optional[str] = Field(
        default=None,
        description="runner:previous_task_status — TaskStatus of the finished task.",
    )
    previous_task_result: Optional[str] = Field(
        default=None,
        description="runner:previous_task_result — result text from the finished task.",
    )
    previous_task_summary: Optional[str] = Field(
        default=None,
        description="runner:previous_task_summary — summary text from the finished task.",
    )
    previous_task_objective: Optional[str] = Field(
        default=None,
        description="runner:previous_task_objective — objective that was given to the finished task.",
    )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_session_dict(
        cls, state: dict[str, Any], finished_task_id: int
    ) -> "CarryState":
        """
        Read a finished task's scoped keys from the raw session state dict.
        Mirrors the logic in the original _extract_carry_state static method.
        """
        keys = TaskScopedKeys(finished_task_id)
        return cls(
            previous_task_id=finished_task_id,
            previous_task_status=state.get(keys.status),
            previous_task_result=state.get(keys.result),
            previous_task_summary=state.get(keys.summary),
            previous_task_objective=state.get(keys.objective),
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_session_dict(self) -> dict[str, Any]:
        """
        Produce the runner:-namespaced keys that get merged into the
        next task's initial session state.
        """
        return {
            "runner:previous_task_id": self.previous_task_id,
            "runner:previous_task_status": self.previous_task_status,
            "runner:previous_task_result": self.previous_task_result,
            "runner:previous_task_summary": self.previous_task_summary,
            "runner:previous_task_objective": self.previous_task_objective,
        }


# ─── SessionState (read-back helper) ─────────────────────────────────────────


class SessionState(BaseModel):
    """
    Read-back model: deserialise a raw session dict back into typed fields.

    Use this in _build_iteration_result and _is_task_completed instead of
    doing raw dict.get() calls with string literals scattered around the file.

    Usage:
        s = SessionState.from_session_dict(final_state, task_id)
        if s.task_status == TaskStatus.DONE: ...
        result = s.task_result
    """

    # ── Runner-scoped (previous task) ──────────────────────────────────────
    previous_task_id: Optional[int] = None
    previous_task_status: Optional[str] = None
    previous_task_result: Optional[str] = None
    previous_task_summary: Optional[str] = None
    previous_task_objective: Optional[str] = None

    # ── Runner-scoped (active task) ────────────────────────────────────────
    active_task_ref: Optional[str] = None
    active_template_key: Optional[str] = None
    iteration: Optional[int] = None
    params: dict[str, Any] = Field(default_factory=dict)
    input_artifacts: dict[str, str] = Field(default_factory=dict)

    # ── Task-scoped (resolved for task_id) ────────────────────────────────
    task_id: Optional[int] = None
    task_objective: Optional[str] = None
    task_status: Optional[str] = None
    task_current: Optional[Any] = None
    task_result: str = ""
    task_summary: str = ""
    task_pool: list[Any] = Field(default_factory=list)

    @classmethod
    def from_session_dict(cls, state: dict[str, Any], task_id: int) -> "SessionState":
        keys = TaskScopedKeys(task_id)
        return cls(
            # Previous task
            previous_task_id=state.get("runner:previous_task_id"),
            previous_task_status=state.get("runner:previous_task_status"),
            previous_task_result=state.get("runner:previous_task_result"),
            previous_task_summary=state.get("runner:previous_task_summary"),
            previous_task_objective=state.get("runner:previous_task_objective"),
            # Active task
            active_task_ref=state.get("runner:active_task_ref"),
            active_template_key=state.get("runner:active_template_key"),
            iteration=state.get("runner:iteration"),
            params=state.get("runner:params") or {},
            input_artifacts=state.get("runner:input_artifacts") or {},
            # Task-scoped
            task_id=task_id,
            task_objective=state.get(keys.objective),
            task_status=state.get(keys.status),
            task_current=state.get(keys.current),
            task_result=state.get(keys.result) or "",
            task_summary=state.get(keys.summary) or "",
            task_pool=state.get(keys.pool) or [],
        )

    def is_completed(self) -> bool:
        """Replaces the _is_task_completed(task_id, state) helper."""
        return self.task_status == TaskStatus.DONE


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
