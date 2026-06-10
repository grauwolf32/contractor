"""Task management: models, formatters, manager, and tool factory.

This package was split from the original ``contractor/tools/tasks.py`` module.
All previously-public names are re-exported here so that existing imports
(``from contractor.tools.tasks import ...``) continue to work unchanged.
"""

from contractor.tools.tasks.formatters import (
    SubtaskFormatter,
    _is_empty_worker_response,
    _parse_worker_output,
    _stringify_formatted,
)
from contractor.tools.tasks.manager import StreamlineManager
from contractor.tools.tasks.models import (
    _MAX_LITERAL_EVAL_LEN,
    _SUBTASK_DECOMPOSITION_SCHEMA_JSON,
    DO_NOT_FINISH_WITH_NO_TASKS_DONE,
    NO_ACTIVE_SUBTASKS_MSG,
    NO_REMAINING_SUBTASKS_MSG,
    NO_SUBTASKS_EXIST_MSG,
    SKIP_REASON_MUST_NOT_BE_EMPTY,
    SUBTASK_DECOMPOSE_EMPTY_LIST,
    SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE,
    SUBTASK_DECOMPOSE_OVER_CAPACITY,
    SUBTASK_NOT_CURRENT_MSG,
    SUBTASK_REQUIRES_DECOMPOSITION_MSG,
    SUBTASK_REQUIRES_RESOLUTION_MSG,
    SUBTASK_RESULT_MALFORMED,
    SUBTASK_STATUS_TRANSITIONS,
    TASK_LIMIT_REACHED_MSG,
    InvalidStatusTransitionError,
    Subtask,
    SubtaskDecomposition,
    SubtaskExecutionResult,
    SubtaskSpec,
    SubtaskStatus,
    TaskManagerExecutionError,
    validate_status_transition,
)
from contractor.tools.tasks.tools import (
    _INSTRUCTION_SNAPSHOT_ATTR,
    _WORKER_EXAMPLE_DONE,
    _WORKER_EXAMPLE_INCOMPLETE,
    TASK_RESULT_SUMMARIZATION_INSTRUCTIONS,
    _get_agent_ref,
    _prepare_worker_instructions,
    instrument_worker,
    task_tools,
)

__all__ = [
    # models.py
    "NO_SUBTASKS_EXIST_MSG",
    "NO_ACTIVE_SUBTASKS_MSG",
    "TASK_LIMIT_REACHED_MSG",
    "SUBTASK_NOT_CURRENT_MSG",
    "SKIP_REASON_MUST_NOT_BE_EMPTY",
    "SUBTASK_REQUIRES_RESOLUTION_MSG",
    "SUBTASK_REQUIRES_DECOMPOSITION_MSG",
    "SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE",
    "SUBTASK_DECOMPOSE_OVER_CAPACITY",
    "SUBTASK_RESULT_MALFORMED",
    "SubtaskStatus",
    "SUBTASK_STATUS_TRANSITIONS",
    "DO_NOT_FINISH_WITH_NO_TASKS_DONE",
    "SUBTASK_DECOMPOSE_EMPTY_LIST",
    "NO_REMAINING_SUBTASKS_MSG",
    "_MAX_LITERAL_EVAL_LEN",
    "TaskManagerExecutionError",
    "InvalidStatusTransitionError",
    "SubtaskSpec",
    "SubtaskDecomposition",
    "_SUBTASK_DECOMPOSITION_SCHEMA_JSON",
    "Subtask",
    "SubtaskExecutionResult",
    "validate_status_transition",
    # formatters.py
    "SubtaskFormatter",
    "_stringify_formatted",
    "_is_empty_worker_response",
    "_parse_worker_output",
    # manager.py
    "StreamlineManager",
    # tools.py
    "_INSTRUCTION_SNAPSHOT_ATTR",
    "TASK_RESULT_SUMMARIZATION_INSTRUCTIONS",
    "_WORKER_EXAMPLE_DONE",
    "_WORKER_EXAMPLE_INCOMPLETE",
    "_prepare_worker_instructions",
    "_get_agent_ref",
    "instrument_worker",
    "task_tools",
]
