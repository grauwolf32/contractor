from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Final, Literal, Optional, Union
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field, ValidationError

NO_ACTIVE_TASKS_MSG: Final[str] = (
    "There are no active subtasks. Add a subtask using `add_subtask` to begin."
)

TASK_LIMIT_REACHED_MSG: Final[str] = (
    "The maximum number of subtasks has been reached. "
    "You MUST summarize records and finish the execution."
)

SUBTASK_NOT_CURRENT_MSG: Final[str] = (
    "Subtask `{task_id}` is not the current subtask. "
    "Only the current task returned by `get_current_subtask` may be used."
)

SUBTASK_REQUIRES_DECOMPOSITION_MSG: Final[str] = (
    "Subtask `{task_id}` is incomplete. "
    "Decompose this task into subtasks before calling `execute_current_subtask` again."
)

SUBTASK_STATUS_TRANSITIONS: Final[dict[str, Any]] = {
    "new": ["done", "incomplete", "skipped"],
    "incomplete": [],
    "done": [],
    "skipped": [],
}

SKIP_REASON_MUST_NOT_BE_EMPTY: Final[str] = (
    "Skip reason MUST not be empty."
    "Describe the reason why you have decided to skip current subtask."
)

SUBTASK_RESULT_MALFORMED: Final[str] = (
    "Subtask result has malformed format. The result stored in the output."
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


class SubtaskExecutionResult(BaseModel):
    """
    Result of executing the current task.
    """

    task_id: str = Field(
        ...,
        description="Identifier of the subtask that was executed. MUST match the current task_id.",
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
class SubtaskFormatter:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"
    _CODE_BLOCK_RE = re.compile(
        r"```(?P<lang>[a-zA-Z0-9_+-]+)?\s*\n(?P<body>.*?)\n```",
        re.DOTALL,
    )

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
    def _subtask_result_to_json(
        subtask_result: SubtaskExecutionResult, **kwargs
    ) -> dict[str, Any]:
        return subtask_result.model_dump()

    @staticmethod
    def _subtask_result_to_markdown(
        subtask_result: SubtaskExecutionResult, **kwargs
    ) -> str:
        return (
            f"### RESULT [ID: {subtask_result.task_id}]\n"
            f"**Status**: {subtask_result.status}\n"
            f"**Output**: {subtask_result.output}\n"
            f"**Summary**: {subtask_result.summary}\n"
            f"---"
        )

    @staticmethod
    def _subtask_result_to_yaml(
        subtask_result: SubtaskExecutionResult, **kwargs
    ) -> str:
        payload = {
            f"result_{subtask_result.task_id}": {
                "task_id": subtask_result.task_id,
                "status": subtask_result.status,
                "output": subtask_result.output,
                "summary": subtask_result.summary,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False)

    @staticmethod
    def _subtask_result_to_xml(
        subtask_result: SubtaskExecutionResult, indent: int = 0, **kwargs
    ) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        task_id = xml_escape(subtask_result.task_id)
        task_status = xml_escape(subtask_result.status)
        output = xml_escape(subtask_result.output)
        summary = xml_escape(subtask_result.summary)

        return (
            f'{pad}<result task_id="{task_id}">\n'
            f"{pad2}<status>{task_status}</status>\n"
            f"{pad2}<output>{output}</output>\n"
            f"{pad2}<summary>{summary}</summary>\n"
            f"{pad}</result>"
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
            "json": SubtaskFormatter._subtask_to_json,
            "markdown": SubtaskFormatter._subtask_to_markdown,
            "yaml": SubtaskFormatter._subtask_to_yaml,
            "xml": SubtaskFormatter._subtask_to_xml,
        }
        if formatter := formatters.get(self._format):
            output = formatter(subtask, **kwargs)
            return self._type_hint(output, type_hint)

        return SubtaskFormatter._subtask_to_json(subtask, **kwargs)

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

        return [SubtaskFormatter._subtask_to_json(subtask) for subtask in subtasks]

    def format_subtask_result(
        self, subtask_result: SubtaskExecutionResult, type_hint: bool = False, **kwargs
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable] = {
            "json": SubtaskFormatter._subtask_result_to_json,
            "markdown": SubtaskFormatter._subtask_result_to_markdown,
            "yaml": SubtaskFormatter._subtask_result_to_yaml,
            "xml": SubtaskFormatter._subtask_result_to_xml,
        }

        if formatter := formatters.get(self._format):
            output = formatter(subtask_result, **kwargs)
            return self._type_hint(output, type_hint)

        return SubtaskFormatter._subtask_result_to_json(subtask_result, **kwargs)

    def format_subtask_results(
        self, subtask_results: list[SubtaskExecutionResult], type_hint: bool = False
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join(
                [
                    self.format_subtask_result(subtask_result)
                    for subtask_result in subtask_results
                ]
            )
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            output = "\n".join(
                [
                    "<results>\n"
                    + self.format_subtask_result(subtask_result, indent=1)
                    + "\n</results>"
                    for subtask_result in subtask_results
                ]
            )
            return self._type_hint(output, type_hint)

        return [
            SubtaskFormatter._subtask_result_to_json(subtask_result)
            for subtask_result in subtask_results
        ]

    def format_task_record(
        self, subtask: Subtask, subtask_result: SubtaskExecutionResult
    ) -> Union[str, dict[str, Any]]:
        if self._format == "json":
            record_dict: dict[str, Any] = self._subtask_to_json(subtask)
            tr = self._subtask_result_to_json(subtask_result)
            tr.pop("task_id", None)
            record_dict |= tr
            return record_dict

        record: Union[str, dict[str, Any]] = self.format_subtask(subtask)
        record += self.format_subtask_result(subtask_result)
        return record

    @staticmethod
    def _validate_result_payload(payload: Any) -> Optional[SubtaskExecutionResult]:
        with suppress(ValidationError, TypeError):
            return SubtaskExecutionResult.model_validate(payload)
        return None

    @classmethod
    def _extract_fenced_blocks(cls, text: str) -> list[tuple[Optional[str], str]]:
        return [
            (
                (m.group("lang") or "").strip().lower() or None,
                m.group("body").strip(),
            )
            for m in cls._CODE_BLOCK_RE.finditer(text)
        ]

    @staticmethod
    def _parse_subtask_result_json(output: str) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        candidates = [output]

        # Иногда LLM возвращает python-dict вместо JSON.
        if output.startswith("{") or output.startswith("["):
            candidates.append(output)

        for candidate in candidates:
            with suppress(json.JSONDecodeError, TypeError):
                parsed = json.loads(candidate)
                result = SubtaskFormatter._validate_result_payload(parsed)
                if result:
                    return result

        # Осторожный fallback только для dict/list-подобных строк
        if output.startswith("{") or output.startswith("["):
            with suppress(ValueError, SyntaxError, TypeError, MemoryError):
                parsed = ast.literal_eval(output)
                result = SubtaskFormatter._validate_result_payload(parsed)
                if result:
                    return result

        return None

    @staticmethod
    def _parse_subtask_result_yaml(output: str) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        with suppress(yaml.YAMLError, TypeError):
            parsed = yaml.safe_load(output)

            # Вариант 1: нормальный объект результата
            result = SubtaskFormatter._validate_result_payload(parsed)
            if result:
                return result

            # Вариант 2: обёртка вида result_1: {...}
            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = next(iter(parsed.values()))
                result = SubtaskFormatter._validate_result_payload(inner)
                if result:
                    return result

        return None

    @staticmethod
    def _parse_subtask_result_markdown(output: str) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        header_re = re.compile(
            r"(?im)^\s*#{1,6}\s*result\s*\[id:\s*(?P<task_id>[^\]]+)\]\s*$"
        )
        field_re = re.compile(
            r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(status|output|summary)(?:\*\*)?\s*:\s*(.*)$"
        )
        end_re = re.compile(r"(?m)^\s*---\s*$")

        task_id = None
        header_match = header_re.search(output)
        if header_match:
            task_id = header_match.group("task_id").strip()

        end_match = end_re.search(output)
        body = output[: end_match.start()] if end_match else output

        matches = list(field_re.finditer(body))
        if not matches:
            return None

        data: dict[str, Any] = {
            "task_id": task_id,
            "status": None,
            "output": None,
            "summary": None,
        }

        for i, match in enumerate(matches):
            key = match.group(1).lower()
            first_line = match.group(2).rstrip()

            section_start = match.end()
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_tail = body[section_start:section_end]

            tail_lines = section_tail.splitlines()

            while tail_lines and not tail_lines[0].strip():
                tail_lines.pop(0)
            while tail_lines and not tail_lines[-1].strip():
                tail_lines.pop()

            if tail_lines:
                value = "\n".join([first_line, *tail_lines]).strip()
            else:
                value = first_line.strip()

            if key == "status":
                value = value.splitlines()[0].strip()

            data[key] = value

        with suppress(ValidationError, TypeError):
            return SubtaskExecutionResult.model_validate(data)

        return None

    @staticmethod
    def _extract_nested_result_xml(text: str) -> Optional[str]:
        m = re.search(r"(<result\b.*?</result>)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        return None

    @staticmethod
    def _parse_subtask_result_xml(output: str) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        output = SubtaskFormatter._extract_nested_result_xml(output)
        if not output:
            return None

        with suppress(ET.ParseError, AttributeError, TypeError):
            root = ET.fromstring(output)

            if root.tag.lower() == "results":
                result_el = root.find("./result")
            elif root.tag.lower() == "result":
                result_el = root
            else:
                return None

            if result_el is None:
                return None

            payload = {
                "task_id": (result_el.attrib.get("task_id") or "").strip(),
                "status": (result_el.findtext("status") or "").strip(),
                "output": (result_el.findtext("output") or "").strip(),
                "summary": (result_el.findtext("summary") or "").strip(),
            }

            return SubtaskFormatter._validate_result_payload(payload)

        return None

    @staticmethod
    def _sanitize_llm_output(text: str) -> str:
        text = text.strip()

        # Убираем случайные think-теги
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

        # Убираем обрамляющие кавычки вокруг всего payload
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()

        return text

    def parse_subtask_result(self, output: str) -> Optional[SubtaskExecutionResult]:
        output = SubtaskFormatter._sanitize_llm_output(output)
        output = output.strip()
        if not output:
            return None

        parsers: dict[str, Callable[[str], Optional[SubtaskExecutionResult]]] = {
            "json": self._parse_subtask_result_json,
            "markdown": self._parse_subtask_result_markdown,
            "yaml": self._parse_subtask_result_yaml,
            "xml": self._parse_subtask_result_xml,
        }

        parse_order = [self._format] + [fmt for fmt in parsers if fmt != self._format]

        # 1. Сначала fenced code blocks
        fenced_blocks = self._extract_fenced_blocks(output)
        for lang, body in fenced_blocks:
            if lang in parsers:
                result = parsers[lang](body)
                if result:
                    return result

        # 2. Если блоки были, но язык не помог — пробуем expected format на теле блоков
        for _, body in fenced_blocks:
            result = parsers[self._format](body)
            if result:
                return result

        # 3. Потом пробуем весь текст: сначала ожидаемый формат, потом fallback
        for fmt_name in parse_order:
            result = parsers[fmt_name](output)
            if result:
                return result

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

    def format_subtask_result_description(
        self, type_hint: bool = False
    ) -> Union[str, dict[str, Any]]:
        out: dict[str, str] = {}
        for name, finfo in SubtaskExecutionResult.model_fields.items():
            desc = (finfo.description or "").strip()
            out[name] = desc

        stub = SubtaskExecutionResult.model_construct(**out)
        return self.format_subtask_result(stub, type_hint=type_hint)


@dataclass
class StreamlineManager:
    name: str
    max_tasks: int
    fmt: SubtaskFormatter

    def _state_key(self, ctx: ToolContext | CallbackContext) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY) or 0
        invocation_id = ctx.invocation_id
        return f"task::{global_task_id}::{invocation_id}::{self.name}"

    def _subtasks_key(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::tasks"

    def _current_idx(self, ctx: ToolContext | CallbackContext) -> str:
        return self._state_key(ctx) + "::idx"

    @staticmethod
    def _global_keys(
        ctx: ToolContext | CallbackContext,
        key: Literal["", "pool", "summary", "result", "status"],
    ) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY) or 0
        if key == "":
            return f"task::{global_task_id}"
        return f"task::{global_task_id}::{key}"

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

    def get_records(
        self,
        ctx: ToolContext | CallbackContext,
    ) -> list[dict[str, Any]]:
        pool_key = self._global_keys(ctx, "pool")
        ctx.state.setdefault(pool_key, [])
        records: list[dict[str, Any]] = ctx.state[pool_key]
        return records

    def save_record(
        self,
        record: Union[str, dict[str, Any]],
        ctx: ToolContext | CallbackContext,
    ):
        records = self.get_records(ctx)
        records.append(record)
        pool_key = self._global_keys(ctx, "pool")
        ctx.state[pool_key] = records
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

        subtask_result: SubtaskExecutionResult = SubtaskExecutionResult(
            task_id=current.task_id, status="skipped", output=reason, summary=""
        )
        record: Union[str, dict[str, Any]] = self.fmt.format_task_record(
            current, subtask_result
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

    def finish(
        self, status: str, result: str, summary: str, ctx: ToolContext | CallbackContext
    ):
        result_key = self._global_keys(ctx, "result")
        summary_key = self._global_keys(ctx, "summary")
        status_key = self._global_keys(ctx, "status")
        ctx.state[result_key] = result
        ctx.state[summary_key] = summary
        ctx.state[status_key] = status
        return


SUBTASK_PLANNING_PROMPT: Final[str] = """
SUBTASK PLANNING WORKFLOW

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

Rule 1: Task description must be clear and concise.
- Workers are not completeley aware of the full task context.
- All required information must be provided within the task description.

Rule 2: Single Active Task Rule
- Do NOT work on future subtasks.
- Do NOT skip ahead without strong justification.
- Do NOT advance without a worker-reported result.

Rule 3: Advancement Rules
- If the current subtask result is "done": advance automatically.
- If the current subtask result is "incomplete": decomposition is mandatory.
- Advancing an incomplete task without decomposition is forbidden.

Rule 4: Decomposition Rules
- Only decompose the CURRENT subtask.
- Decomposition must:
  - Fully cover remaining work
  - Produce clear, actionable subtasks
  - Avoid trivial, redundant, or overly granular steps

Rule 5: Completion Rules
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

TASK_RESULT_SUMMARIZATION_INSTRUCTIONS: Final[str] = """
OBJECTIVE:
Analyze the task objective, execution records, final result, and final status.
Provide a detailed summary that explains the goal of the task, the main actions taken, the outcome, and the final status.
Include any important notes, warnings, blockers, or other relevant observations if present.

OUTPUT FORMAT:
Detailed summary
""".strip()


def _prepare_worker_instructions(fmt: SubtaskFormatter, type_hint: bool = False) -> str:
    example_1: SubtaskExecutionResult = SubtaskExecutionResult(
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

    example_2: SubtaskExecutionResult = SubtaskExecutionResult(
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
        fmt.format_subtask_result_description()
    )
    if type(format_description) is dict:
        format_description = json.dumps(format_description)

    ex1_fmt: Union[str, dict[str, Any]] = fmt.format_subtask_result(
        example_1, type_hint=type_hint
    )
    if type(ex1_fmt) is dict:
        ex1_fmt = json.dumps(ex1_fmt)

    ex2_fmt: Union[str, dict[str, Any]] = fmt.format_subtask_result(
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
    fmt: SubtaskFormatter,
    type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
) -> LlmAgent:
    if use_input_schema or fmt._format == "json":
        worker.input_schema = Subtask

    if use_output_schema:
        worker.output_schema = SubtaskExecutionResult

    worker.instruction += _prepare_worker_instructions(fmt, type_hint=type_hint)
    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    return worker


def task_tools(
    name: str,
    max_tasks: int,
    worker: LlmAgent | AgentTool,
    fmt: SubtaskFormatter,
    *,
    use_skip: bool = True,
    use_summarization: bool = True,
    use_type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
    worker_instrumentation: bool = True,
    max_records: int = 20,
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

        records: Union[str, list[dict[str, Any]]] = mgr.get_records(tool_context)[
            :max_records
        ]
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
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}

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
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}

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

        subtask_result: SubtaskExecutionResult | dict[str, Any] | None = None

        if isinstance(raw, SubtaskExecutionResult):
            subtask_result = raw
        elif isinstance(raw, dict):
            with suppress(ValidationError, TypeError):
                subtask_result = SubtaskExecutionResult.model_validate(raw)
        elif isinstance(raw, str):
            subtask_result = fmt.parse_subtask_result(raw)

        validated = isinstance(subtask_result, SubtaskExecutionResult)

        if validated and subtask_result.task_id != current.task_id:
            subtask_result = SubtaskExecutionResult(
                task_id=current.task_id,
                status="incomplete",
                output=(
                    f"Worker returned mismatched task_id={subtask_result.task_id!r}. "
                    f"Expected {current.task_id!r}.\n\n"
                    f"Original output:\n{subtask_result.output}"
                ),
                summary="Worker returned a result for the wrong subtask.",
            )
            validated = False

        if not validated:
            raw_dump = raw
            with suppress(ValueError, TypeError):
                raw_dump = json.dumps(raw, ensure_ascii=False)

            subtask_result = SubtaskExecutionResult(
                task_id=current.task_id,
                status="incomplete",
                output=str(raw_dump),
                summary=SUBTASK_RESULT_MALFORMED,
            )

        current.status = subtask_result.status
        record = fmt.format_task_record(current, subtask_result)
        mgr.save_record(record, tool_context)

        tool_context.state.setdefault(mgr._current_idx(tool_context), 0)
        idx = tool_context.state[mgr._current_idx(tool_context)]
        subtasks = mgr.get_subtasks(tool_context)

        subtasks[idx].status = subtask_result.status
        tool_context.state[mgr._subtasks_key(tool_context)] = [
            sub.model_dump() for sub in subtasks
        ]

        can_advance: bool = idx + 1 < len(subtasks)
        if can_advance and subtask_result.status != "incomplete":
            tool_context.state[mgr._current_idx(tool_context)] = idx + 1

        result: dict[str, Any] = {"record": record}
        action: str = ""

        if not can_advance:
            action += NO_ACTIVE_TASKS_MSG
        if subtask_result.status == "incomplete":
            action += SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(task_id=current.task_id)

        result["action"] = action
        if not validated:
            result["error"] = SUBTASK_RESULT_MALFORMED

        return result

    async def finish(
        status: Literal["done", "failed"], result: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Finalize the overall task and report the final outcome.

        Args:
            status: Final task status. Must be either "done" or "failed".
            result: A detailed description of the outcome, including what was completed,
                what failed, and any important context.

        Behavior:
            - Always report the final global task status before exiting.
            - Use "done" only if the global objective has been fully completed.
            - Use "failed" only if the global objective cannot be completed.

        Prefer this tool when:
            - You have fully achieved the global goal.
            - You have definitively failed to achieve the global goal.
            - You have completed all planned work and there is nothing left to do.
        """

        tools: list[Callable] = []
        model: LiteLlm | None = None

        summarizer: LlmAgent = LlmAgent(
            name="task_summarizer",
            description="text summarization agent",
            instruction=TASK_RESULT_SUMMARIZATION_INSTRUCTIONS,
            tools=worker.agent.tools,
            model=worker.agent.model,
        )

        objective_key: str = mgr._global_keys(tool_context, "objective")
        objective: str = tool_context.state.get(objective_key, "")

        args = {
            "objective": objective,
            "records": mgr.get_records(tool_context),
            "result": result,
            "status": status.lower() if status.lower() == "done" else "failed",
        }

        summarizer_tool: AgentTool(summarizer)
        raw = await summarizer_tool.run_async(args=args, tool_context=tool_context)
        mgr.finish(status=status, result=result, summary=summary, ctx=tool_context)
        return {"result": "ok"}

    tools = [
        add_subtask,
        get_current_subtask,
        list_subtasks,
        get_records,
        execute_current_subtask,
        decompose_subtask,
        finish,
    ]
    if use_skip:
        tools.append(skip)

    return tools
