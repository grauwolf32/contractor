from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Optional, Union, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError
from xml.sax.saxutils import escape as xml_escape

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext


# -------------------- CONSTANTS --------------------

EMPTY_TASKS_MGR_STR: Final[str] = (
    "No active subtasks. Use add_subtask to add new subtask."
)
TASK_LIMIT_MGR_STR: Final[str] = "You have reached the limit of available subtask."
TASK_NOT_FOUND_WRK_STR: Final[str] = (
    "Task {task_id} is not found. Check current task to get the task_id."
)
WRONG_TASK_WRK_STR: Final[str] = (
    "Task {task_id} is not current task! Check current task to get the description."
)
INCOMPLETE_NEEDS_DECOMP_STR: Final[str] = (
    "Task {task_id} is incomplete and must be decomposed before advancing."
)

TaskStatus = Literal["new", "done", "incomplete"]
OutputFormat = Literal["json", "markdown", "yaml", "xml"]

# Allowed status transitions
TASK_STATUS_TRANSITIONS: Final[dict[TaskStatus, list[TaskStatus]]] = {
    "new": ["done", "incomplete"],
    "incomplete": ["done"],
    "done": [],
}

# -------------------- MODELS --------------------


class TaskMeta(BaseModel):
    """Metadata describing a subtask to be created."""

    title: str = Field(..., description="Short subtask title")
    description: str = Field(..., description="Detailed subtask description")


class TaskDecompositionList(BaseModel):
    """A list of proposed subtasks (e.g., from a decomposition step)."""

    decomposition: list[TaskMeta] = Field(
        default_factory=list,
        description="Ordered list of subtasks produced by decomposition.",
    )


class Task(BaseModel):
    """A lightweight subtask with a status and export helpers."""

    task_id: str = Field(..., description="Unique subtask identifier (string).")
    title: str = Field(..., description="Short subtask title")
    description: str = Field(..., description="Detailed subtask description")
    status: TaskStatus = Field("new", description="Task status")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_markdown(self) -> str:
        return (
            f"### {self.title} [ID: {self.task_id}]\n"
            f"**Description**: {self.description}\n"
            f"**Status**: {self.status}\n"
        )

    def to_yaml(self) -> str:
        payload = {
            self.task_id: {
                "title": self.title,
                "description": self.description,
                "status": self.status,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False)

    def to_xml(self, indent: int = 0) -> str:
        """Simple XML serializer. Escapes values; not intended for complex XML."""
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        task_id = xml_escape(self.task_id)
        title = xml_escape(self.title)
        description = xml_escape(self.description)
        status = xml_escape(self.status)

        return (
            f'{pad}<task id="{task_id}">\n'
            f"{pad2}<title>{title}</title>\n"
            f"{pad2}<description>{description}</description>\n"
            f"{pad2}<status>{status}</status>\n"
            f"{pad}</task>"
        )

    def can_transition_to(self, new_status: TaskStatus) -> bool:
        return new_status in TASK_STATUS_TRANSITIONS[self.status]

    def transition_to(self, new_status: TaskStatus) -> None:
        if not self.can_transition_to(new_status):
            allowed = TASK_STATUS_TRANSITIONS[self.status]
            raise ValueError(
                f"Invalid transition {self.status!r} -> {new_status!r}. Allowed: {allowed}"
            )
        self.status = new_status


class TaskResult(BaseModel):
    """Result record for executing a subtask."""

    task_title: str = Field(..., description="Short title of the task/subtask")
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Final status after execution")
    result: str = Field(..., description="Detailed execution result")
    summary: str = Field(
        ...,
        description="Execution summary: steps taken, intermediate results, and brief next steps.",
    )


# -------------------- TASK MANAGER --------------------

Ctx = Union[ToolContext, CallbackContext]


@dataclass
class TaskManager:
    """
    Holds subtasks and results. Persists itself in ctx.state.

    Notes:
    - max_tasks is used as a guardrail to encourage re-decomposition.
    - current_id points at the current subtask index (0-based).
    """

    name: str
    max_tasks: int
    subtasks: list[Task] = field(default_factory=list)
    results: dict[str, TaskResult] = field(default_factory=dict)
    current_id: Optional[int] = None
    invocation_id: Optional[str] = None
    history: dict[str, Any] = field(default_factory=dict)
    _format: OutputFormat = "json"

    # ---- formatting helpers ----

    def format_task(self, task: Task) -> Union[str, dict[str, Any]]:
        match self._format:
            case "json":
                return task.to_dict()
            case "markdown":
                return task.to_markdown()
            case "yaml":
                return task.to_yaml()
            case "xml":
                return task.to_xml()
        return task.to_dict()

    def format_tasks(self, tasks: list[Task]) -> Union[str, list[dict[str, Any]]]:
        match self._format:
            case "json":
                return [t.to_dict() for t in tasks]
            case "markdown":
                return "\n".join(t.to_markdown() for t in tasks)
            case "yaml":
                return "\n".join(t.to_yaml() for t in tasks)
            case "xml":
                inner = "\n".join(t.to_xml(indent=1) for t in tasks)
                return f"<subtasks>\n{inner}\n</subtasks>"
        return [t.to_dict() for t in tasks]

    # ---- state persistence ----

    def to_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_tasks": self.max_tasks,
            "subtasks": [t.model_dump() for t in self.subtasks],
            "results": {task_id: r.model_dump() for task_id, r in self.results.items()},
            "current_id": self.current_id,
            "invocation_id": self.invocation_id,
            "format": self._format,
            "history": self.history,
        }

    def advance(self) -> Tuple[TaskResult, Optional[Task]]:
        """
        Apply the current task's reported TaskResult (must exist) and advance pointer.

        Rules:
        - If current result is done: move to next task in list (if any)
        - If current result is incomplete: must have at least one child subtask; move to first child
        """

        current = self.get_current()
        if current is None:
            raise ValueError("No current task to advance.")

        if current.task_id not in self.results:
            raise ValueError(f"No result found for task_id={current.task_id!r}.")

        result = self.results[current.task_id]

        # Apply transition via FSM
        current.transition_to(result.status)  # type: ignore[arg-type]
        self.subtasks[self.current_id] = current

        if current.status == "done":
            # advance sequentially
            if self.current_id + 1 < len(self.subtasks):
                self.current_id += 1
                return result, self.subtasks[self.current_id]
            # no more tasks
            return result, None

        # incomplete -> must be decomposed first
        if not self.next_task_id().startswith(current.task_id):
            raise ValueError(
                INCOMPLETE_NEEDS_DECOMP_STR.format(task_id=current.task_id)
            )

        self.current_id += 1
        return result, self.subtasks[self.current_id]

    @classmethod
    def _cls_key(cls) -> str:
        return cls.__name__

    @classmethod
    def _state_key(cls, name: str, invocation_id: Optional[str]) -> str:
        return name if invocation_id is None else f"{name}::{invocation_id}"

    def save_to_state(self, ctx: Ctx) -> None:
        cls_key = self._cls_key()
        ctx.state.setdefault(cls_key, {})

        cls_data: dict[str, Any] = ctx.state[cls_key]
        state_key = self._state_key(self.name, self.invocation_id)

        cls_data[state_key] = self.to_state()

        # HACK: ctx.state must be explicitly overwritten (kept from your comment)
        ctx.state[cls_key] = cls_data

    @classmethod
    def load_from_state(
        cls, ctx: Ctx, name: str, max_tasks: int, _format: str = "json"
    ) -> "TaskManager":
        cls_key = cls._cls_key()
        ctx.state.setdefault(cls_key, {})

        cls_data: dict[str, Any] = ctx.state[cls_key]
        invocation_id = getattr(ctx, "invocation_id", None)
        state_key = cls._state_key(name, invocation_id)

        raw: Optional[dict[str, Any]] = cls_data.get(state_key)

        mgr = cls(
            name=name, max_tasks=max_tasks, invocation_id=invocation_id, _format=_format
        )

        if raw is None:
            mgr.save_to_state(ctx)
            return mgr

        mgr.current_id = raw.get("current_id")
        mgr.history = raw.get("history", {})
        mgr._format = raw.get("format", "json")

        mgr.subtasks = [Task.model_validate(x) for x in raw.get("subtasks", [])]
        mgr.results = {
            task_id: TaskResult.model_validate(v)
            for task_id, v in (raw.get("results") or {}).items()
        }
        return mgr

    # ---- task operations ----

    def get_current(self) -> Optional[Task]:
        if self.current_id is None:
            return None
        if 0 <= self.current_id < len(self.subtasks):
            return self.subtasks[self.current_id]
        return None

    def next_task_id(self) -> str:
        if not self.subtasks:
            return "1"
        # assumes integer prefix before optional dotted suffix
        last_root = self.subtasks[-1].task_id.split(".")[0]
        return str(int(last_root) + 1)

    def add_task(self, title: str, description: str) -> Task:
        if len(self.subtasks) >= self.max_tasks:
            raise ValueError(TASK_LIMIT_MGR_STR)

        task = Task(
            task_id=self.next_task_id(),
            title=title,
            description=description,
            status="new",
        )
        self.subtasks.append(task)

        # initialize current task if unset
        if self.current_id is None:
            self.current_id = 0

        return task

    def decompose_incomplete_task(
        self, subtask_id: str, new_subtasks: TaskDecompositionList
    ) -> list[Task]:
        idx = next(
            (i for i, t in enumerate(self.subtasks) if t.task_id == subtask_id), None
        )
        if idx is None:
            raise KeyError(f"subtask with id:{subtask_id} is not found")

        orig = self.subtasks[idx]
        if orig.status != "incomplete":
            raise ValueError(
                "could not decompose a task unless it is marked incomplete"
            )

        if not new_subtasks.decomposition:
            raise ValueError("decomposition list is empty")

        insertion: list[Task] = []
        for sub_idx, meta in enumerate(new_subtasks.decomposition, start=1):
            insertion.append(
                Task(
                    task_id=f"{subtask_id}.{sub_idx}",
                    title=meta.title,
                    description=meta.description,
                    status="new",
                )
            )

        # insert children right after the parent (stable ordering)
        self.subtasks = self.subtasks[: idx + 1] + insertion + self.subtasks[idx + 1 :]

        # If we're currently at the parent, move current to the first new child
        if self.current_id == idx:
            self.current_id = idx + 1

        return insertion


# -------------------- TOOL FACTORIES --------------------


def manager_tools(name: str, max_tasks: int, _format: str = "json") -> list[Callable]:
    """Tools to use in an agent with task manager capabilities."""

    def _load(ctx: ToolContext) -> TaskManager:
        return TaskManager.load_from_state(
            ctx, name=name, max_tasks=max_tasks, _format=_format
        )

    def list_subtasks(tool_context: ToolContext) -> dict[str, Any]:
        """
        Return all known subtasks
        """

        mgr = _load(tool_context)
        return {"result": mgr.format_tasks(mgr.subtasks)}

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """
        Returns the current subtask
        """

        mgr = _load(tool_context)
        current = mgr.get_current()
        if current is None:
            return {"result": EMPTY_TASKS_MGR_STR}
        return {"result": mgr.format_task(current)}

    def add_subtask(
        title: str, description: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Append a new subtask to the list of subtasks
        """

        mgr = _load(tool_context)
        try:
            task = mgr.add_task(title=title, description=description)
        except ValueError as exc:
            return {"error": str(exc)}

        mgr.save_to_state(tool_context)
        return {"result": mgr.format_task(task)}

    def advance(tool_context: ToolContext) -> dict[str, Any]:
        """
        Review current task's result and advance to the next task.
        """

        mgr = _load(tool_context)
        try:
            task_result, next_task = mgr.advance()
        except ValueError as exc:
            return {"error": str(exc)}

        mgr.save_to_state(tool_context)
        return {
            "task_result": task_result.model_dump(),
            "next_task": mgr.format_task(next_task)
            if next_task
            else EMPTY_TASKS_MGR_STR,
        }

    def decompose_subtask(
        subtask_id: str,
        new_subtasks: TaskDecompositionList,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Decomposes incomplete subtask into several subtasks (best choice is 1-3 subtask)
        """
        mgr = _load(tool_context)

        if isinstance(new_subtasks, dict):
            try:
                new_subtasks = TaskDecompositionList.model_validate(new_subtasks)
            except ValidationError as exc:
                return {"error": str(exc)}

        try:
            insertion = mgr.decompose_incomplete_task(
                subtask_id=subtask_id, new_subtasks=new_subtasks
            )
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}

        mgr.save_to_state(tool_context)
        return {"result": mgr.format_tasks(insertion)}

    return [list_subtasks, get_current_subtask, add_subtask, decompose_subtask, advance]


def worker_tools(name: str, max_tasks: int, _format: str = "json") -> list[Callable]:
    """Tools meant for the 'worker' side (read current task + report result)."""

    def _load(ctx: ToolContext) -> TaskManager:
        return TaskManager.load_from_state(
            ctx, name=name, max_tasks=max_tasks, _format=_format
        )

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """
        Return the current subtask
        """

        mgr = _load(tool_context)
        current = mgr.get_current()
        if current is None:
            return {"result": EMPTY_TASKS_MGR_STR}
        return {"result": mgr.format_task(current)}

    def report(task_result: TaskResult, tool_context: ToolContext) -> dict[str, Any]:
        """
        Report complete or incomplete result with the execution summary
        """
        mgr = _load(tool_context)

        if isinstance(task_result, dict):
            try:
                task_result = TaskResult.model_validate(task_result)
            except ValidationError as exc:
                return {"error": str(exc)}

        task_id = task_result.task_id

        if not any(t.task_id == task_id for t in mgr.subtasks):
            return {"error": TASK_NOT_FOUND_WRK_STR.format(task_id=task_id)}

        current = mgr.get_current()
        if current is None or current.task_id != task_id:
            return {"error": WRONG_TASK_WRK_STR.format(task_id=task_id)}

        mgr.results[task_id] = task_result

        mgr.save_to_state(tool_context)
        return {"result": "ok"}

    return [get_current_subtask, report]
