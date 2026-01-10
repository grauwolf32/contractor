from __future__ import annotations

import re
import ast
import json
import yaml

from dataclasses import dataclass, field
from typing import Any, Callable, Final, Literal, Optional, Union, Tuple

from contextlib import suppress
from pydantic import BaseModel, Field, ValidationError
from xml.sax.saxutils import escape as xml_escape

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext

NO_ACTIVE_TASKS_MSG: Final[str] = (
    "There are no active subtasks. Add a subtask using `add_subtask` to begin."
)

TASK_LIMIT_REACHED_MSG: Final[str] = (
    "The maximum number of subtasks has been reached. "
    "Complete or decompose existing subtasks before adding new ones."
)

TASK_ID_NOT_FOUND_MSG: Final[str] = (
    "Task with id `{task_id}` was not found. "
    "Call `get_current_subtask` to retrieve the valid task_id."
)

TASK_NOT_CURRENT_MSG: Final[str] = (
    "Task `{task_id}` is not the current task. "
    "Only the current task returned by `get_current_subtask` may be reported."
)

TASK_REQUIRES_DECOMPOSITION_MSG: Final[str] = (
    "Task `{task_id}` is incomplete. "
    "Decompose this task into subtasks before calling `advance` again."
)

TASK_STATUS_TRANSITIONS: Final[dict[str, Any]] = {
    "new": ["done", "incomplete"],
    "incomplete": ["done"],
    "done": [],
}


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

    subtask_id: str = Field(
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
    status: Literal["new", "done", "incomplete"] = Field(
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

    status: Literal["done", "incomplete"] = Field(
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
class Format:
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
            subtask.task_id: {
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
            task_result.task_id: {
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
            f"{pad2}<status>{status}</status>\n"
            f"{pad2}<output>{output}</output>\n"
            f"{pad2}<summary>{summary}</summary>\n"
            f"{pad}</task_result>"
        )

    def _type_hint(
        self,
        output: Union[str, dict[str, Any], list[dict[str, Any]]],
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any], list[dict[str, Any]]]:
        if type(output) is not str:
            return output
        return f"```{self._format}\n{output}\n```"

    def format_subtask(
        self, subtask: Subtask, type_hint: bool = False, **kwargs
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable] = {
            "json": Format._subtask_to_json,
            "markdown": Format._subtask_to_markdown,
            "yaml": Format._subtask_to_yaml,
            "xml": Format._subtask_to_xml,
        }
        if formatter := formatters.get(self._format):
            output = formatter(subtask, **kwargs)
            return self._type_hint(output, type_hint)

        return Format._subtask_to_json(subtask, **kwargs)

    def format_subtasks(
        self, subtasks: list[Subtask], type_hint: bool = False
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join([self.format_subtask(subtask) for subtask in subtasks])
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            output = "\n".join(
                [
                    "<subtasks>\n"
                    + self.format_subtask(subtask, indent=1)
                    + "\n</subtasks>"
                    for subtask in subtasks
                ]
            )
            return self._type_hint(output, type_hint)

        return [Format._subtask_to_json(subtask) for subtask in subtasks]

    def format_task_result(
        self, task_result: TaskExecutionResult, type_hint: bool = False, **kwargs
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable] = {
            "json": Format._task_result_to_json,
            "markdown": Format._task_result_to_markdown,
            "yaml": Format._task_result_to_yaml,
            "xml": Format._task_result_to_xml,
        }

        if formatter := formatters.get(self._format):
            output = formatter(task_result, **kwargs)
            return self._type_hint(output, type_hint)

        return Format._task_result_to_json(task_result, **kwargs)

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
            Format._task_result_to_json(task_result) for task_result in task_results
        ]

    @staticmethod
    def _parse_task_result_json(output: str) -> Optional[TaskExecutionResult]:
        orig_output = output
        output = output.strip()

        if not output:
            return None

        WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
        candidates = [output, _WHITESPACE_RE.sub(" ", output)]

        for candidate in candidates:
            with suppress(json.JSONDecodeError, ValidationError, TypeError):
                task_result = json.loads(output)
                return TaskExecutionResult.model_validate(task_result)

            with suppress(
                ValueError, SyntaxError, ValidationError, TypeError, MemoryError
            ):
                task_result = ast.literal_eval(output)
                return TaskExecutionResult.model_validate(task_result)

            return None

    @staticmethod
    def _parse_task_result_yaml(output: str) -> Optional[TaskExecutionResult]:
        orig_output = output
        output = output.strip()

        if not output:
            return None

        with suppress(ValidationError.TypeError):
            task_meta = yaml.safe_load(candidate)
            if type(task_meta) is not dict:
                raise TypeError

            keys = task_meta.keys()
            if len(keys) > 0:
                return TaskExecutionResult.model_validate(task_meta)

            task_id = keys[0]
            if not re.match("\d\.?[\d\.]*"):
                raise TypeError

            if type(task_meta[task_id]) is not dict:
                raise TypeError

            task_result = {"task_id": task_id, **task_meta[task_id]}

            return TaskExecutionResult.model_validate(task_result)

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

        m = TASK_ID_RE.search(text)
        if m:
            task_result["task_id"] = m.group("task_id").strip()

        lines = text.splitlines()

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
    def _parse_task_result_from_string(output: str) -> Optional[TaskExecutionResult]:
        parsers: dict[str, Callable] = {
            "json": _parse_task_result_json,
            "markdown": _parse_task_result_markdown,
            "yaml": _parse_task_result_yaml,
            "xml": _parse_task_result_xml,
        }

        hints_re: dict[str, re.Pattern] = {
            _format: re.compile("```{_format}(.+?)```") for _format in parsers.keys()
        }

        task_result: Optional[TaskExecutionResult] = None
        for _fromat, hint_re in hints_re.items():
            if m := hint_re.search(output):
                task_result = parsers[_fromat](m.group(1).strip())
                if task_result:
                    return task_result

        for _fromat, parser in parsers.keys():
            task_result = parser(output)
            if task_result:
                return task_result

        return None
