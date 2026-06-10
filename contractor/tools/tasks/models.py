from __future__ import annotations

import json
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
NO_SUBTASKS_EXIST_MSG: Final[str] = (
    "No subtasks exist yet. You MUST call `add_subtask` first."
)
NO_ACTIVE_SUBTASKS_MSG: Final[str] = (
    "There is no current active subtask. "
    "Review existing records first. "
    "If the objective is already complete, call `finish`. "
    "Otherwise, call `add_subtask` only if genuinely new work remains."
)
TASK_LIMIT_REACHED_MSG: Final[str] = (
    "The maximum number of subtasks ({max_tasks}) has been reached. "
    "You MUST NOT create new subtasks. "
    "Execute or skip the remaining subtasks first, then call `finish` "
    "with a summary of the records collected so far."
)
SUBTASK_DECOMPOSE_OVER_CAPACITY: Final[str] = (
    "Decomposing into {requested} subtasks would exceed the subtask limit "
    "({max_tasks}): only {remaining} more subtask(s) can be added. "
    "Retry with fewer children."
)
SUBTASK_NOT_CURRENT_MSG: Final[str] = (
    "Subtask `{task_id}` is NOT the current subtask. "
    "You may only operate on the subtask returned by `get_current_subtask`. "
    "Call `get_current_subtask` first, then retry with the correct task_id."
)
SKIP_REASON_MUST_NOT_BE_EMPTY: Final[str] = (
    "The skip reason MUST NOT be empty. "
    "Provide a clear, specific explanation of why you are skipping "
    "this subtask (e.g. dependency unavailable, out of scope, duplicate)."
)
SUBTASK_REQUIRES_RESOLUTION_MSG: Final[str] = (
    "Subtask `{task_id}` has status '{status}' — it cannot be executed again "
    "directly. You MUST either decompose it by calling `decompose_subtask`, "
    "or skip it by calling `skip`."
)
SUBTASK_REQUIRES_DECOMPOSITION_MSG: Final[str] = (
    "Subtask `{task_id}` has status 'incomplete' — it was NOT fully resolved. "
    "You have to decompose it into smaller subtasks by calling `decompose_subtask` "
    "before calling `execute_current_subtask` again. You can skip task by calling "
    "`skip` if you have reached the maximum number of subtasks."
)
SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE: Final[str] = (
    "Subtask `{task_id}` has status '{status}'. "
    "Only subtasks with status 'incomplete' or 'malformed' can be decomposed. "
    "If the subtask is 'new', execute it first. "
    "If it is already resolved ('done', 'skipped', or 'decomposed'), move on "
    "to the next subtask."
)
SUBTASK_SKIP_NOT_SKIPPABLE: Final[str] = (
    "Subtask `{task_id}` has status '{status}' and was NOT skipped. "
    "Only subtasks with status 'new', 'incomplete', or 'malformed' can be "
    "skipped. It is already resolved — move on to the next subtask, or call "
    "`finish` if the objective is complete."
)
SUBTASK_RESULT_MALFORMED: Final[str] = (
    "The worker returned a result that could not be completely parsed into the "
    "expected format. The raw output has been stored for reference. "
    "The subtask is marked 'malformed' — its raw results may still contain "
    "useful information. You MUST either decompose or skip it."
)
SubtaskStatus: TypeAlias = Literal[
    "new", "done", "incomplete", "malformed", "skipped", "decomposed"
]

SUBTASK_STATUS_TRANSITIONS: Final[dict[str, list[str]]] = {
    "new": ["done", "incomplete", "malformed", "skipped"],
    "malformed": ["skipped", "decomposed"],
    "incomplete": ["skipped", "decomposed"],
    "done": [],
    "decomposed": [],
    "skipped": [],
}
DO_NOT_FINISH_WITH_NO_TASKS_DONE: Final[str] = (
    "Cannot finish with status='done' when no subtasks have been completed "
    "or there are still 'new' (unexecuted) subtasks remaining. "
    "Execute or skip all 'new' subtasks first, then call `finish`, "
    "or set the status='failed'."
)
SUBTASK_DECOMPOSE_EMPTY_LIST: Final[str] = (
    "Subtask decomposition is empty. You need to provide 1-3 subtasks as decomposition."
)

NO_REMAINING_SUBTASKS_MSG: Final[str] = (
    "No remaining subtasks exist. There is no actionable subtask in the plan. "
    "Do NOT infer new work from memories, records, or prior subtasks. "
    "If the objective is complete, call `finish`. Otherwise, add a new subtask "
    "only if genuinely new required work remains."
)

_MAX_LITERAL_EVAL_LEN: Final[int] = 50_000


# ═══════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════
class TaskManagerExecutionError(Exception):
    """Raised when task manager encounters an unrecoverable error."""


class InvalidStatusTransitionError(TaskManagerExecutionError):
    """Raised when an invalid status transition is attempted."""

    def __init__(self, current_status: str, new_status: str):
        allowed = SUBTASK_STATUS_TRANSITIONS.get(current_status, [])
        super().__init__(
            f"Invalid transition: '{current_status}' -> '{new_status}'. "
            f"Allowed transitions from '{current_status}': {allowed}"
        )


# ═══════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════
class SubtaskSpec(BaseModel):
    """Specification for creating a new subtask."""

    title: str = Field(
        ...,
        description=(
            "Concise, action-oriented subtask title. "
            "Use imperative mood (e.g. 'Extract API endpoints'). "
            "Keep under 80 characters."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Detailed description of the subtask. MUST include:\n"
            "1. What specific work needs to be done\n"
            "2. What inputs or context are available\n"
            "3. What the expected output or deliverable is\n"
            "4. Any constraints, boundaries, or edge cases to consider"
        ),
    )


class SubtaskDecomposition(BaseModel):
    """Result of decomposing an incomplete task into executable subtasks."""

    subtasks: list[SubtaskSpec] = Field(
        ...,
        min_length=1,
        max_length=3,
        description=(
            "Ordered list of 1-3 executable subtasks. Requirements:\n"
            "- Each subtask MUST be independently executable\n"
            "- Together they MUST cover ALL remaining work of the parent task\n"
            "- Order matters: subtask N may depend on results of subtask N-1\n"
            "- Prefer fewer, broader subtasks over many narrow ones"
        ),
    )


_SUBTASK_DECOMPOSITION_SCHEMA_JSON: Final[str] = json.dumps(
    SubtaskDecomposition.model_json_schema(), indent=2
)


class Subtask(BaseModel):
    """A single executable unit of work."""

    task_id: str = Field(
        ...,
        description=(
            "Unique subtask identifier using dotted numeric format. "
            "Root tasks: '0', '1', '2'. "
            "Child tasks from decomposition: '1.1', '1.2'."
        ),
        pattern=r"^\d+(\.\d+)*$",
    )
    title: str = Field(
        ...,
        description=(
            "Concise, action-oriented title in imperative mood. "
            "Single responsibility. Under 80 characters."
        ),
        min_length=1,
    )
    description: str = Field(
        ...,
        description=(
            "Detailed scope, constraints, inputs, expected outputs, "
            "and completion criteria for this subtask."
        ),
        min_length=1,
    )
    status: SubtaskStatus = Field(
        default="new",
        description=(
            "Lifecycle status of the subtask:\n"
            "- 'new': Not yet executed\n"
            "- 'done': Successfully completed\n"
            "- 'incomplete': Attempted but needs decomposition\n"
            "- 'decomposed': Replaced by child subtasks and no longer executable\n"
            "- 'malformed': Worker output could not be parsed\n"
            "- 'skipped': Deliberately skipped with reason\n"
            "Valid transitions: "
            "new->[done,incomplete,malformed,skipped], "
            "incomplete->[decomposed,skipped], "
            "malformed->[decomposed,skipped]"
        ),
    )


class SubtaskExecutionResult(BaseModel):
    """Structured result produced by the worker after executing a subtask."""

    task_id: str = Field(
        ...,
        description=(
            "Identifier of the subtask that was executed. "
            "MUST exactly match the task_id provided as input."
        ),
    )
    status: Literal["done", "incomplete"] = Field(
        ...,
        description=(
            "Execution outcome:\n"
            "- 'done': Fully completed\n"
            "- 'incomplete': Partially completed, needs decomposition\n"
        ),
    )
    output: str = Field(
        ...,
        description=(
            "Factual, detailed execution output. Include all concrete "
            "results, data, artifacts, errors, and observations."
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "Brief execution summary (2-5 sentences). Goal, what was "
            "accomplished, and if incomplete what remains."
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# Status transition validation
# ═══════════════════════════════════════════════════════════════════
def validate_status_transition(current_status: str, new_status: str) -> bool:
    allowed = SUBTASK_STATUS_TRANSITIONS.get(current_status, [])
    if new_status not in allowed:
        raise InvalidStatusTransitionError(current_status, new_status)
    return True
