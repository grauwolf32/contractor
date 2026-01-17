from __future__ import annotations

import re
import ast
import json
import yaml

from dataclasses import dataclass
from typing import Any, Callable, Final, Literal, Optional, Union

from contextlib import suppress
from pydantic import BaseModel, Field, ValidationError
from xml.sax.saxutils import escape as xml_escape

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

NO_ACTIVE_TASKS_MSG: Final[str] = (
    "There are no active subtasks. Add a subtask using `add_subtask` to begin."
)

TASK_LIMIT_REACHED_MSG: Final[str] = (
    "The maximum number of subtasks has been reached. "
    "You MUST summarize records and finish the execution."
)

TASK_ID_NOT_FOUND_MSG: Final[str] = (
    "Task with id `{task_id}` was not found. "
    "Call `get_current_subtask` to retrieve the valid task_id."
)

TASK_NOT_CURRENT_MSG: Final[str] = (
    "Task `{task_id}` is not the current task. "
    "Only the current task returned by `get_current_subtask` may be used."
)

TASK_REQUIRES_DECOMPOSITION_MSG: Final[str] = (
    "Task `{task_id}` is incomplete. "
    "Decompose this task into subtasks before calling `execute_current_subtask` again."
)

TASK_STATUS_TRANSITIONS: Final[dict[str, Any]] = {
    "new": ["done", "incomplete", "skipped"],
    "incomplete": [],
    "done": [],
    "skipped": [],
}

SKIP_REASON_MUST_NOT_BE_EMPTY: Final[str] = (
    "Skip reason MUST not be empty."
    "Describe the reason why you have decided to skip current task."
)

TASK_RESULT_MALFORMED: Final[str] = (
    "Task result has malformed format. The result stored in the output."
)

_GLOBAL_TASK_ID_KEY: Final[str] = "_global_task_id"


class TaskManagerExecutionError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class SubtaskSpec(BaseModel):
    """
    Specification for creating a new subtask.
    Used when decomposing a task into executable subtasks.
    """

    title: str = Field(..., description="Concise, action-oriented subtask title.")
    description: str = Field(
        ...,
        description="Detailed description of the subtask, including scope and expected outcome.",
    )


class SubtaskDecomposition(BaseModel):
    """
    Result of decomposing an incomplete task into executable subtasks.

    This structure defines the ordered subtasks that collectively
    replace the parent task.
    """

    subtasks: list[SubtaskSpec] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of executable subtasks. "
            "Subtasks MUST collectively cover all remaining work of the parent task."
        ),
    )


class Subtask(BaseModel):
    """
    A single executable unit of work.

    Subtasks may be created as root tasks (e.g. "3") or as children of an incomplete task
    (e.g. "2.1", "2.2").
    """

    task_id: str = Field(
        ...,
        description="Unique subtask identifier. Dotted numeric, e.g. '2' or '2.1'.",
        pattern=r"^\d+(\.\d+)*$",
    )
    title: str = Field(
        ...,
        description="Concise, action-oriented title (imperative, single responsibility).",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Detailed scope, constraints, and completion criteria for this subtask.",
        min_length=1,
    )
    status: Literal["new", "done", "incomplete", "skipped"] = Field(
        default="new",
        description="Workflow status: new -> done, or new -> incomplete -> done.",
    )


class TaskExecutionResult(BaseModel):
    """
    Result of executing the current task.
    """

    task_id: str = Field(
        ...,
        description="Identifier of the task that was executed. MUST match the current task_id.",
    )

    status: Literal["done", "incomplete", "skipped"] = Field(
        ...,
        description="Execution outcome: 'done' if fully completed, 'incomplete' if further work or decomposition is required.",
    )

    output: str = Field(
        ...,
        description="Factual execution output: what was done, produced artifacts, errors, or observations.",
    )

    summary: str = Field(
        ...,
        description=(
            "Brief execution summary. "
            "Include steps taken and, if status is 'incomplete', what remains to be done."
        ),
    )


@dataclass
class TaskFormat:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"

    @staticmethod
    def _subtask_to_json(subtask: Subtask, **kwargs) -> dict[str, Any]:
        return subtask.model_dump()

    @staticmethod
    def _subtask_to_markdown(subtask: Subtask, **kwargs) -> str:
        return (
            f"### {subtask.title} [ID: {subtask.task_id}]\n"
            f"**Description**: {subtask.description}\n"
            f"**Status**: {subtask.status}\n"
        )

    @staticmethod
    def _subtask_to_yaml(subtask: Subtask, **kwargs) -> str:
        payload = {
            f"task_{subtask.task_id}": {
                "task_id": subtask.task_id,
                "title": subtask.title,
                "description": subtask.description,
                "status": subtask.status,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False)

    @staticmethod
    def _subtask_to_xml(subtask: Subtask, indent: int = 0, **kwargs) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        task_id = xml_escape(subtask.task_id)
        title = xml_escape(subtask.title)
        description = xml_escape(subtask.description)
        status = xml_escape(subtask.status)

        return (
            f'{pad}<task id="{task_id}">\n'
            f"{pad2}<title>{title}</title>\n"
            f"{pad2}<description>{description}</description>\n"
            f"{pad2}<status>{status}</status>\n"
            f"{pad}</task>"
        )

    @staticmethod
    def _task_result_to_json(
        task_result: TaskExecutionResult, **kwargs
    ) -> dict[str, Any]:
        return task_result.model_dump()

    @staticmethod
    def _task_result_to_markdown(task_result: TaskExecutionResult, **kwargs) -> str:
        return (
            f"### RESULT [ID: {task_result.task_id}]\n"
            f"**Status**: {task_result.status}\n"
            f"**Output**: {task_result.output}\n"
            f"**Summary**: {task_result.summary}\n"
            f"---"
        )

    @staticmethod
    def _task_result_to_yaml(task_result: TaskExecutionResult, **kwargs) -> str:
        payload = {
            f"result_{task_result.task_id}": {
                "task_id": task_result.task_id,
                "status": task_result.status,
                "output": task_result.output,
                "summary": task_result.summary,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False)

    @staticmethod
    def _task_result_to_xml(
        task_result: TaskExecutionResult, indent: int = 0, **kwargs
    ) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        task_id = xml_escape(task_result.task_id)
        task_status = xml_escape(task_result.status)
        output = xml_escape(task_result.output)
        summary = xml_escape(task_result.summary)

        return (
            f'{pad}<task_result task_id="{task_id}">\n'
            f"{pad2}<status>{task_status}</status>\n"
            f"{pad2}<output>{output}</output>\n"
            f"{pad2}<summary>{summary}</summary>\n"
            f"{pad}</task_result>"
        )

    def _type_hint(
        self,
        output: Union[str, dict[str, Any], list[dict[str, Any]]],
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any], list[dict[str, Any]]]:
        if type(output) is not str or not type_hint:
            return output
        return f"```{self._format}\n{output}\n```"

    def format_subtask(
        self, subtask: Subtask, type_hint: bool = False, **kwargs
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable] = {
            "json": TaskFormat._subtask_to_json,
            "markdown": TaskFormat._subtask_to_markdown,
            "yaml": TaskFormat._subtask_to_yaml,
            "xml": TaskFormat._subtask_to_xml,
        }
        if formatter := formatters.get(self._format):
            output = formatter(subtask, **kwargs)
            return self._type_hint(output, type_hint)

        return TaskFormat._subtask_to_json(subtask, **kwargs)

    def format_subtasks(
        self, subtasks: list[Subtask], type_hint: bool = False
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join([self.format_subtask(subtask) for subtask in subtasks])
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            output = (
                "<subtasks>\n"
                + "\n".join(
                    [self.format_subtask(subtask, indent=1) for subtask in subtasks]
                )
                + "\n</subtasks>"
            )
            return self._type_hint(output, type_hint)

        return [TaskFormat._subtask_to_json(subtask) for subtask in subtasks]

    def format_task_result(
        self, task_result: TaskExecutionResult, type_hint: bool = False, **kwargs
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable] = {
            "json": TaskFormat._task_result_to_json,
            "markdown": TaskFormat._task_result_to_markdown,
            "yaml": TaskFormat._task_result_to_yaml,
            "xml": TaskFormat._task_result_to_xml,
        }

        if formatter := formatters.get(self._format):
            output = formatter(task_result, **kwargs)
            return self._type_hint(output, type_hint)

        return TaskFormat._task_result_to_json(task_result, **kwargs)

    def format_task_results(
        self, task_results: list[TaskExecutionResult], type_hint: bool = False
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join(
                [self.format_task_result(task_result) for task_result in task_results]
            )
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            output = "\n".join(
                [
                    "<results>\n"
                    + self.format_task_result(task_result, indent=1)
                    + "\n</results>"
                    for task_result in task_results
                ]
            )
            return self._type_hint(output, type_hint)

        return [
            TaskFormat._task_result_to_json(task_result) for task_result in task_results
        ]

    def format_task_record(
        self, subtask: Subtask, task_result: TaskExecutionResult
    ) -> Union[str, dict[str, Any]]:
        if self._format == "json":
            record_dict: dict[str, Any] = self._subtask_to_json(subtask)
            tr = self._task_result_to_json(task_result)
            tr.pop("task_id", None)
            record_dict |= tr
            return record_dict

        record: Union[str, dict[str, Any]] = self.format_subtask(subtask)
        record += self.format_task_result(task_result)
        return record

    @staticmethod
    def _parse_task_result_json(output: str) -> Optional[TaskExecutionResult]:
        output = output.strip()

        if not output:
            return None

        WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
        candidates = [output, WHITESPACE_RE.sub(" ", output)]

        for candidate in candidates:
            with suppress(json.JSONDecodeError, ValidationError, TypeError):
                task_result = json.loads(candidate)
                return TaskExecutionResult.model_validate(task_result)

            with suppress(
                ValueError, SyntaxError, ValidationError, TypeError, MemoryError
            ):
                task_result = ast.literal_eval(candidate)
                return TaskExecutionResult.model_validate(task_result)

        return None

    @staticmethod
    def _parse_task_result_yaml(output: str) -> Optional[TaskExecutionResult]:
        output = output.strip()

        if not output:
            return None

        with suppress(ValidationError, TypeError, yaml.YAMLError):
            task_meta = yaml.safe_load(output)
            if type(task_meta) is not dict:
                raise TypeError

            keys = list(task_meta.keys())
            if len(keys) > 1:
                return TaskExecutionResult.model_validate(task_meta)

            task_id = keys[0]
            if type(task_meta[task_id]) is not dict:
                raise TypeError

            return TaskExecutionResult.model_validate(task_meta[task_id])

        return None

    @staticmethod
    def _parse_task_result_markdown(output: str) -> Optional[TaskExecutionResult]:
        FIELD_RE = re.compile(
            r"(?im)^\s*(?:\*\*)?(status|output|summary)(?:\*\*)?\s*:?\s*(.*)\s*$"
        )
        END_RE = re.compile(r"(?m)^\s*---\s*$")
        TASK_ID_RE = re.compile(r"(?i)\[id:\s*(?P<task_id>[^\]]+)\]")

        task_result: dict[str, Optional[str]] = {
            "task_id": None,
            "status": None,
            "output": None,
            "summary": None,
        }

        m = TASK_ID_RE.search(output)
        if m:
            task_result["task_id"] = m.group("task_id").strip()

        lines = output.splitlines()

        i = 0
        while i < len(lines):
            if END_RE.match(lines[i]):
                break

            m = FIELD_RE.match(lines[i])
            if not m:
                i += 1
                continue

            key = m.group(1).lower()
            buf = [m.group(2)]

            i += 1
            while (
                i < len(lines)
                and not END_RE.match(lines[i])
                and not FIELD_RE.match(lines[i])
            ):
                if value := lines[i].strip():
                    buf.append(value)
                i += 1

            value = "\n".join(buf).strip()

            if key == "status":
                task_result[key] = value.split("\n")[0]
            else:
                task_result[key] = value if value else None

        with suppress(ValidationError):
            return TaskExecutionResult.model_validate(task_result)

        return None

    @staticmethod
    def _parse_task_result_xml(output: str) -> Optional[TaskExecutionResult]:
        task_result_re = re.compile(
            r"(?i)<task_result\s*task_id\s*=(?P<task_id>[^>]+)>(?P<result>.+?)</task_result>",
            re.DOTALL,
        )
        status_re = re.compile(r"(?i)<status>(?P<status>.+?)</status>", re.DOTALL)
        output_re = re.compile(r"(?i)<output>(?P<output>.+?)</output>", re.DOTALL)
        summary_re = re.compile(r"(?i)<summary>(?P<summary>.+?)</summary>", re.DOTALL)

        m = task_result_re.search(output)
        if not m:
            return None

        task_id = m.group("task_id").strip().replace('"', "")
        result = m.group("result").strip()

        m = status_re.search(result)
        if not m:
            return None
        status = m.group("status").strip()

        m = output_re.search(result)
        if not m:
            return None
        output = m.group("output").strip()

        m = summary_re.search(result)
        if not m:
            return None
        summary = m.group("summary").strip()

        with suppress(ValidationError):
            return TaskExecutionResult(
                task_id=task_id,
                status=status,
                output=output,
                summary=summary,
            )
        return None

    @staticmethod
    def parse_task_result(output: str) -> Optional[TaskExecutionResult]:
        parsers: dict[str, Callable] = {
            "json": TaskFormat._parse_task_result_json,
            "markdown": TaskFormat._parse_task_result_markdown,
            "yaml": TaskFormat._parse_task_result_yaml,
            "xml": TaskFormat._parse_task_result_xml,
        }

        hints_re: dict[str, re.Pattern] = {
            k: re.compile(rf"```{k}\s*(.+?)```", re.DOTALL) for k in parsers
        }

        task_result: Optional[TaskExecutionResult] = None
        for fmt_name, parser in parsers.items():
            hint_re = hints_re.get(fmt_name)
            if m := hint_re.search(output):
                task_result = parser(m.group(1).strip())
                if task_result:
                    return task_result

        for fmt_name, parser in parsers.items():
            task_result = parser(output)
            if task_result:
                return task_result

        return None

    def format_subtask_description(
        self, type_hint: bool = False
    ) -> Union[str, dict[str, Any]]:
        out: dict[str, str] = {}
        for name, finfo in Subtask.model_fields.items():
            desc = (finfo.description or "").strip()
            out[name] = desc

        stub = Subtask.model_construct(**out)
        return self.format_subtask(stub, type_hint=type_hint)

    def format_task_result_description(
        self, type_hint: bool = False
    ) -> Union[str, dict[str, Any]]:
        out: dict[str, str] = {}
        for name, finfo in TaskExecutionResult.model_fields.items():
            desc = (finfo.description or "").strip()
            out[name] = desc

        stub = TaskExecutionResult.model_construct(**out)
        return self.format_task_result(stub, type_hint=type_hint)


@dataclass
class StreamlineManager:
    name: str
    max_tasks: int
    fmt: TaskFormat

    def _state_key(self, ctx: ToolContext | CallbackContext) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY) or 0
        invocation_id = ctx.invocation_id
        return f"task::{global_task_id}::{invocation_id}::{self.name}"

    def _subtasks_key(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::tasks"

    def _records_key(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::records"

    def _current_idx(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::idx"

    @staticmethod
    def _global_pool_key(ctx: ToolContext | CallbackContext) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY) or 0
        return f"task::{global_task_id}::pool"

    @staticmethod
    def _global_summary_key(ctx: ToolContext | CallbackContext) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY) or 0
        invocation_id = ctx.invocation_id
        return f"task::{global_task_id}::{invocation_id}::summary"

    @staticmethod
    def _next_task_id(subtasks) -> str:
        if not subtasks:
            return "0"

        last_root = subtasks[-1].task_id.split(".")[0]
        return str(int(last_root) + 1)

    def get_subtasks(self, ctx: ToolContext | CallbackContext) -> list[Subtask]:
        ctx.state.setdefault(self._subtasks_key(ctx), [])
        subtasks = [Subtask(**sub) for sub in ctx.state[self._subtasks_key(ctx)]]

        return subtasks

    def add_subtask(
        self, subtask_spec: SubtaskSpec, ctx: ToolContext | CallbackContext
    ) -> Optional[Subtask]:
        subtasks = self.get_subtasks(ctx)
        if len(subtasks) >= self.max_tasks:
            return

        new = Subtask(
            task_id=self._next_task_id(subtasks),
            title=subtask_spec.title,
            description=subtask_spec.description,
            status="new",
        )

        ctx.state.setdefault(self._current_idx(ctx), -1)
        idx = ctx.state[self._current_idx(ctx)]
        if (
            idx is None
            or idx < 0
            or idx >= len(subtasks)
            or (idx >= len(subtasks) - 1 and subtasks[idx].status == "done")
        ):
            idx = len(subtasks)
        ctx.state[self._current_idx(ctx)] = idx

        subtasks.append(new)
        ctx.state[self._subtasks_key(ctx)] = [sub.model_dump() for sub in subtasks]
        return new

    def get_current_subtask(
        self, ctx: ToolContext | CallbackContext
    ) -> Optional[Subtask]:
        subtasks = self.get_subtasks(ctx)
        idx = ctx.state.get(self._current_idx(ctx))

        if idx is None or idx < 0 or idx >= len(subtasks):
            return None
        return subtasks[idx]

    def get_pool(
        self, ctx: ToolContext | CallbackContext
    ) -> list[Union[str, dict[str, Any]]]:
        ctx.state.setdefault(self._global_pool_key(ctx), [])
        return ctx.state[self._global_pool_key(ctx)]

    def save_to_pool(
        self,
        record: Union[str, dict[str, Any]],
        ctx: ToolContext | CallbackContext,
    ) -> None:
        pool = self.get_pool(ctx)
        pool.append(record)
        ctx.state[self._global_pool_key(ctx)] = pool

    def get_records(
        self,
        ctx: ToolContext | CallbackContext,
    ) -> Union[str, list[dict[str, Any]]]:
        if self.fmt._format == "json":
            ctx.state.setdefault(self._records_key(ctx), [])
        else:
            ctx.state.setdefault(self._records_key(ctx), "")

        records: Union[str, list[dict[str, Any]]] = ctx.state[self._records_key(ctx)]
        return records

    def save_record(
        self,
        record: Union[str, dict[str, Any]],
        ctx: ToolContext | CallbackContext,
    ):
        records = self.get_records(ctx)
        if type(records) is list:
            records.append(record)
        else:
            records += f"\n{record}"
        ctx.state[self._records_key(ctx)] = records
        return

    def skip(
        self, reason: str, ctx: ToolContext | CallbackContext
    ) -> Optional[Subtask]:
        idx = ctx.state.get(self._current_idx(ctx))
        if idx is None:
            return

        subtasks = self.get_subtasks(ctx)
        if idx + 1 >= len(subtasks):
            return

        current = subtasks[idx]
        next_subtask = subtasks[idx + 1]

        subtasks[idx].status = "skipped"
        ctx.state[self._subtasks_key(ctx)] = [sub.model_dump() for sub in subtasks]

        task_result: TaskExecutionResult = TaskExecutionResult(
            task_id=current.task_id, status="skipped", output=reason, summary=""
        )
        record: Union[str, dict[str, Any]] = self.fmt.format_task_record(
            current, task_result
        )
        self.save_record(record, ctx)
        ctx.state[self._current_idx(ctx)] = idx + 1

        return next_subtask

    def decompose_current_subtask(
        self, new_subtasks: list[SubtaskSpec], ctx: ToolContext | CallbackContext
    ) -> Optional[list[Subtask]]:
        subtasks = self.get_subtasks(ctx)
        idx = ctx.state.get(self._current_idx(ctx))

        if idx is None:
            return

        current_id: str = subtasks[idx].task_id
        insertion: list[Subtask] = []
        for ind, spec in enumerate(new_subtasks, start=1):
            insertion.append(
                Subtask(
                    task_id=f"{current_id}.{ind}",
                    title=spec.title,
                    description=spec.description,
                    status="new",
                )
            )

        subtasks = subtasks[: idx + 1] + insertion + subtasks[idx + 1 :]
        ctx.state[self._subtasks_key(ctx)] = [sub.model_dump() for sub in subtasks]
        ctx.state[self._current_idx(ctx)] = idx + 1
        return insertion


TASK_PLANNING_PROMPT = """
TASK PLANNING WORKFLOW

You are a task-planning agent responsible for coordinating multi-step work through explicit subtasks.
Your role is to plan, monitor, and adapt based on worker execution results.

--------------------------------------------------
1. SUBTASK MODEL
--------------------------------------------------

Each subtask has exactly one status:

- new         : planned but not yet executed
- done        : successfully completed
- incomplete  : attempted but failed or partially completed
- skip        : intentionally skipped due to irrelevance or redundancy

Valid state transitions:
- new -> done
- new -> incomplete
- new -> skip

No other transitions are allowed.

--------------------------------------------------
2. CORE INVARIANTS (MUST ALWAYS HOLD)
--------------------------------------------------

1) Single Active Subtask
   - There is exactly ONE current subtask at any time (current_id).
   - All reasoning and actions must focus only on the current subtask.

2) Worker-Driven Progress
   - The worker executes the current subtask and reports results.
   - Planning decisions must be based ONLY on reported results.

3) Strict Status Semantics
   - If execution fails or is blocked, mark the subtask as "incomplete".
   - If execution succeeds, mark the subtask as "done" and advance automatically.

--------------------------------------------------
3. WHEN TO USE THIS WORKFLOW
--------------------------------------------------

Use this workflow ONLY when the task:
- Requires multiple dependent steps
- Involves planning, execution, and verification
- May require decomposition if execution is blocked
- Is explicitly requested by the user or system

DO NOT use this workflow for:
- Single-step or trivial tasks
- Purely informational or explanatory responses

--------------------------------------------------
4. STANDARD OPERATING PROCEDURE
--------------------------------------------------

Follow this loop strictly:

1) Inspect State
   - Call list_subtasks or get_current_subtask to understand current progress.

2) Plan
   - Call add_subtask only when additional steps are required.

3) Execute
   - Call execute on the current subtask.

4) Handle Results
   - If result is "done": advancement to the next subtask is automatic.
   - If result is "incomplete": you MUST call decompose_subtask.

5) Skip (Exceptional Case)
   - Call skip only if the current subtask is clearly irrelevant or invalid.

--------------------------------------------------
5. TASK PLANNING RULES (HARD CONSTRAINTS)
--------------------------------------------------

Rule 1: Single Active Task Rule
- Do NOT work on future subtasks.
- Do NOT skip ahead without strong justification.
- Do NOT advance without a worker-reported result.

Rule 2: Advancement Rules
- If the current subtask result is "done": advance automatically.
- If the current subtask result is "incomplete": decomposition is mandatory.
- Advancing an incomplete task without decomposition is forbidden.

Rule 3: Decomposition Rules
- Only decompose the CURRENT subtask.
- Decomposition must:
  - Fully cover remaining work
  - Produce clear, actionable subtasks
  - Avoid trivial, redundant, or overly granular steps

Rule 4: Completion Rules
- Always analyze execution results before planning next steps.
- Ensure subtasks remain if the overall task is not complete.
- Never assume completion without explicit confirmation.

--------------------------------------------------
6. AGENT MINDSET
--------------------------------------------------

- Be conservative in advancing.
- Be explicit in planning.
- Prefer decomposition over guessing.
- Treat this workflow as a strict state machine, not a suggestion.

--------------------------------------------------
7. TOOLS
--------------------------------------------------

- add_subtask
- get_current_subtask
- list_subtasks
- get_records
- execute_current_subtask
- decompose_subtask
- skip 
""".strip()


def _prepare_worker_instructions(fmt: TaskFormat, type_hint: bool = False) -> str:
    example_1: TaskExecutionResult = TaskExecutionResult(
        task_id="1",
        status="done",
        output=(
            "- Reviewed source files for HTTP endpoint definitions:\n"
            "  - src/main/java/com/example/ExampleController.java\n"
            "  - src/main/java/com/example/AdminController.java\n"
            "- Identified the following endpoints:\n"
            "  - GET /example\n"
            "  - POST /example\n"
            "  - PUT /example/id\n"
            "  - DELETE /example/id\n"
            "  - GET /admin/health\n"
        ),
        summary=(
            "Task: Gather information about HTTP endpoints in the project\n"
            "Result: Completed successfully\n"
            "- All HTTP endpoints are defined in two controller classes\n"
            "- No additional endpoint definitions were found outside these files"
        ),
    )

    example_2: TaskExecutionResult = TaskExecutionResult(
        task_id="2",
        status="incomplete",
        output=(
            "- Searched for HTTP endpoint annotations in the main source directory\n"
            "- Found 2 endpoints in src/main/java/com/example/ExampleController.java:\n"
            "  - GET /example\n"
            "  - POST /example\n"
        ),
        summary=(
            "Task: Gather information about HTTP endpoints in the project\n"
            "Status: Incomplete\n"
            "Reason:\n"
            "- Only a subset of project files has been analyzed so far\n"
            "Next steps:\n"
            "- Enumerate all Java source files in the repository\n"
            "- Inspect remaining controller classes for additional endpoints"
        ),
    )

    format_description: Union[str, dict[str, Any]] = (
        fmt.format_task_result_description()
    )
    if type(format_description) is dict:
        format_description = json.dumps(format_description)

    ex1_fmt: Union[str, dict[str, Any]] = fmt.format_task_result(
        example_1, type_hint=type_hint
    )
    if type(ex1_fmt) is dict:
        ex1_fmt = json.dumps(ex1_fmt)

    ex2_fmt: Union[str, dict[str, Any]] = fmt.format_task_result(
        example_2, type_hint=type_hint
    )
    if type(ex2_fmt) is dict:
        ex2_fmt = json.dumps(ex2_fmt)

    instruction: str = "IMPORTANT: After subtask is completed, describe the results using the following structure:\n"
    instruction += f"{format_description}\n"
    instruction += "EXAMPLES:\n"
    instruction += f"{ex1_fmt}\n\n{ex2_fmt}\n\n"
    return instruction


def instrument_worker(
    worker: LlmAgent,
    fmt: TaskFormat,
    type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
) -> LlmAgent:
    if use_input_schema or fmt._format == "json":
        worker.input_schema = Subtask

    if use_output_schema:
        worker.output_schema = TaskExecutionResult

    worker.instruction += _prepare_worker_instructions(fmt, type_hint=type_hint)
    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    return worker


def task_tools(
    name: str,
    max_tasks: int,
    worker: LlmAgent | AgentTool,
    fmt: TaskFormat,
    *,
    use_skip: bool = True,
    use_summarization: bool = True,
    use_type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
    worker_instrumentation: bool = True,
) -> list[Callable]:
    if worker_instrumentation:
        worker = instrument_worker(
            worker, fmt, use_type_hint, use_input_schema, use_output_schema
        )

    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    mgr = StreamlineManager(name, max_tasks, fmt)

    def add_subtask(
        title: str, description: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Append a new subtask to the list of subtasks
        """

        subtask: Optional[Subtask] = mgr.add_subtask(
            SubtaskSpec(title=title, description=description), tool_context
        )
        if subtask is None:
            return {"error": TASK_LIMIT_REACHED_MSG}

        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """
        Return the current subtask.
        Reminder:
        - There is exactly one current task. This is the only task the worker should execute now.
        - You MUST explicitly pass both task_id and title to the worker to start execution of the current task.
        """

        subtask: Optional[Subtask] = mgr.get_current_subtask(tool_context)
        if subtask is None:
            return {
                "error": NO_ACTIVE_TASKS_MSG,
            }

        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def list_subtasks(tool_context: ToolContext) -> dict[str, Any]:
        """
        List all known subtasks.
        Reminder:
        - Use this to understand scope and ordering before doing work.
        - Do not skip tasks; there is exactly one current task at a time.
        """

        subtasks = mgr.get_subtasks(tool_context)
        return {"result": fmt.format_subtasks(subtasks, type_hint=use_type_hint)}

    def get_records(tool_context: ToolContext) -> dict[str, Any]:
        """
        List of all previous subtask with results.
        Reminder:
        - Use this to review results of the previous subtasks.
        """

        records: Union[str, list[dict[str, Any]]] = mgr.get_records(tool_context)
        return {"result": records}

    def decompose_subtask(
        task_id: str, decomposition: SubtaskDecomposition, tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Decomposes incomplete subtask into several subtasks (best choice is 1-3 subtask).
        Reminder:
        - Only current subtask with status "incomplete" could be decomposed.
        """

        if isinstance(decomposition, str):
            schema = json.dumps(SubtaskDecomposition.model_json_schema())
            return {
                "error": f"TypeError: invalid format of the decomposition. Use format of SubtaskDecomposition: {schema}"
            }
        if isinstance(decomposition, dict):
            try:
                decomposition = SubtaskDecomposition.model_validate(decomposition)
            except ValidationError as exc:
                return {"error": str(exc)}

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_TASKS_MSG}
        if str(task_id) != current.task_id:
            return {"error": TASK_NOT_CURRENT_MSG.format(task_id=task_id)}

        insertion: list[Subtask] = mgr.decompose_current_subtask(
            decomposition.subtasks, tool_context
        )
        return {"result": fmt.format_subtasks(insertion)}

    def skip(task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Skip execution of the current subtask.
        Reminder:
        - IMPORTANT: Use this tool only if you have strong reason to skip the current subtask.
        """

        if not reason.strip():
            return {"error": SKIP_REASON_MUST_NOT_BE_EMPTY}

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_TASKS_MSG}
        if str(task_id) != current.task_id:
            return {"error": TASK_NOT_CURRENT_MSG.format(task_id=task_id)}

        next_subtask = mgr.skip(reason, tool_context)
        if next_subtask is None:
            return {"result": NO_ACTIVE_TASKS_MSG}

        return {"result": fmt.format_subtask(next_subtask)}

    async def execute_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """
        Execute current subtask
        """
        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_TASKS_MSG}

        args: dict[str, Any] = {}
        if fmt._format == "json" or use_input_schema:
            args = fmt._subtask_to_json(current)
        else:
            args = {"request": fmt.format_subtask(current)}

        raw = await worker.run_async(args=args, tool_context=tool_context)
        task_result: TaskExecutionResult | None = None

        if isinstance(raw, str):
            task_result = fmt.parse_task_result(raw)

        if task_result is None:
            task_result = raw

        validated: bool = False
        if isinstance(task_result, dict):
            with suppress(ValidationError):
                task_result = TaskExecutionResult.model_validate(task_result)
                validated = True

        if isinstance(task_result, TaskExecutionResult):
            validated = True

        if not validated:
            with suppress(ValueError, TypeError):
                raw = json.dumps(raw)
            task_result = TaskExecutionResult(
                task_id=current.task_id,
                status="incomplete",
                output=raw,
                summary=TASK_RESULT_MALFORMED,
            )

        current.status = task_result.status
        record = fmt.format_task_record(current, task_result)
        mgr.save_record(record, tool_context)
        mgr.save_to_pool(record, tool_context)

        tool_context.state.setdefault(mgr._current_idx(tool_context), 0)
        idx = tool_context.state[mgr._current_idx(tool_context)]
        subtasks = mgr.get_subtasks(tool_context)

        subtasks[idx].status = task_result.status
        tool_context.state[mgr._subtasks_key(tool_context)] = [
            sub.model_dump() for sub in subtasks
        ]

        can_advance: bool = idx + 1 < len(subtasks)
        if can_advance and task_result.status != "incomplete":
            tool_context.state[mgr._current_idx(tool_context)] = idx + 1

        result: dict[str, Any] = {"record": record}
        action: str = ""

        if not can_advance:
            action += NO_ACTIVE_TASKS_MSG
        if task_result.status == "incomplete":
            action += TASK_REQUIRES_DECOMPOSITION_MSG.format(task_id=current.task_id)

        result["action"] = action
        if not validated:
            result["error"] = TASK_RESULT_MALFORMED

        return result

    tools = [
        add_subtask,
        get_current_subtask,
        list_subtasks,
        get_records,
        execute_current_subtask,
        decompose_subtask,
    ]
    if use_skip:
        tools.append(skip)

    return tools
