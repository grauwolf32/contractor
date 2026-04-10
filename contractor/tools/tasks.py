from __future__ import annotations

import ast
import json
import logging
import re
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Final,
    Generator,
    Literal,
    Optional,
    Union,
)
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Constants – User-facing messages
# ═══════════════════════════════════════════════════════════════════

NO_ACTIVE_TASKS_MSG: Final[str] = (
    "No active subtasks exist. "
    "You MUST call `add_subtask` to create one before proceeding."
)

TASK_LIMIT_REACHED_MSG: Final[str] = (
    "The maximum number of subtasks ({max_tasks}) has been reached. "
    "You MUST NOT create new subtasks. "
    "Summarize the records collected so far and call `finish` immediately."
)

SUBTASK_NOT_CURRENT_MSG: Final[str] = (
    "Subtask `{task_id}` is NOT the current subtask. "
    "You may only operate on the subtask returned by `get_current_subtask`. "
    "Call `get_current_subtask` first, then retry with the correct task_id."
)

SUBTASK_REQUIRES_DECOMPOSITION_MSG: Final[str] = (
    "Subtask `{task_id}` has status 'incomplete' — it was NOT fully resolved. "
    "You MUST decompose it into smaller subtasks by calling `decompose_subtask` "
    "before calling `execute_current_subtask` again."
)

SUBTASK_DECOMPOSE_NOT_INCOMPLETE: Final[str] = (
    "Subtask `{task_id}` has status '{status}'. "
    "Only subtasks with status 'incomplete' can be decomposed. "
    "If the subtask is 'new', execute it first. "
    "If it is 'done' or 'skipped', move on to the next subtask."
)

SKIP_REASON_MUST_NOT_BE_EMPTY: Final[str] = (
    "The skip reason MUST NOT be empty. "
    "Provide a clear, specific explanation of why you are skipping "
    "this subtask (e.g. dependency unavailable, out of scope, duplicate)."
)

SUBTASK_RESULT_MALFORMED: Final[str] = (
    "The worker returned a result that could not be parsed into the "
    "expected format. The raw output has been stored. "
    "The subtask is marked 'incomplete' — decompose or retry."
)

SUBTASK_STATUS_TRANSITIONS: Final[dict[str, list[str]]] = {
    "new": ["done", "incomplete", "skipped"],
    "incomplete": ["done", "skipped"],
    "done": [],
    "skipped": [],
}

_GLOBAL_TASK_ID_KEY: Final[str] = "_global_task_id"
_MAX_LITERAL_EVAL_LEN: Final[int] = 50_000


# ═══════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════


class TaskManagerExecutionError(Exception):
    """Raised when task manager encounters an unrecoverable error."""

    def __init__(self, message: str = ""):
        super().__init__(message)


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
    """
    Specification for creating a new subtask.
    Used when decomposing a task into executable subtasks.
    """

    title: str = Field(
        ...,
        description=(
            "Concise, action-oriented subtask title. "
            "Use imperative mood (e.g. 'Extract API endpoints', "
            "'Validate input schema'). Keep under 80 characters."
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
    """
    Result of decomposing an incomplete task into executable subtasks.

    This structure defines the ordered subtasks that collectively
    replace the parent task. Each subtask must be independently
    executable and together they must cover all remaining work.
    """

    subtasks: list[SubtaskSpec] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of 1-3 executable subtasks. Requirements:\n"
            "- Each subtask MUST be independently executable\n"
            "- Together they MUST cover ALL remaining work of the parent task\n"
            "- Order matters: subtask N may depend on results of subtask N-1\n"
            "- Prefer fewer, broader subtasks over many narrow ones"
        ),
    )


class Subtask(BaseModel):
    """
    A single executable unit of work.

    Subtasks may be created as root tasks (e.g. '3') or as children
    of an incomplete task (e.g. '2.1', '2.2').
    """

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
    status: Literal["new", "done", "incomplete", "skipped"] = Field(
        default="new",
        description=(
            "Lifecycle status of the subtask:\n"
            "- 'new': Not yet executed\n"
            "- 'done': Successfully completed\n"
            "- 'incomplete': Attempted but needs decomposition\n"
            "- 'skipped': Deliberately skipped with reason\n"
            "Valid transitions: new->{done,incomplete,skipped}, "
            "incomplete->{done,skipped}"
        ),
    )


class SubtaskExecutionResult(BaseModel):
    """
    Structured result produced after executing a subtask.
    The worker MUST return this exact structure after every execution.
    """

    task_id: str = Field(
        ...,
        description=(
            "Identifier of the subtask that was executed. "
            "MUST exactly match the task_id that was provided as input."
        ),
    )

    status: Literal["done", "incomplete", "skipped"] = Field(
        ...,
        description=(
            "Execution outcome. Choose exactly one:\n"
            "- 'done': The subtask was fully completed. All deliverables "
            "are present in the output.\n"
            "- 'incomplete': The subtask was partially completed. Further "
            "decomposition or work is required. Explain what remains "
            "in the summary.\n"
            "- 'skipped': The subtask was skipped. Explain why in the output."
        ),
    )

    output: str = Field(
        ...,
        description=(
            "Factual, detailed execution output. Include ALL of the following "
            "that apply:\n"
            "- Concrete results, data, or artifacts produced\n"
            "- Files read, created, or modified (with paths)\n"
            "- Commands executed and their results\n"
            "- Errors encountered and how they were handled\n"
            "- Key observations or findings\n"
            "Do NOT include opinions, plans, or next steps here — "
            "those belong in the summary."
        ),
    )

    summary: str = Field(
        ...,
        description=(
            "Brief execution summary (2-5 sentences). MUST include:\n"
            "- What was the goal of this subtask\n"
            "- What was accomplished\n"
            "- If status is 'incomplete': what specifically remains to be done "
            "and why it could not be completed\n"
            "- If status is 'done': confirmation that all deliverables are "
            "present in the output"
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# Status transition validation
# ═══════════════════════════════════════════════════════════════════


def validate_status_transition(current_status: str, new_status: str) -> bool:
    """
    Validate that a status transition is allowed.

    Returns True if valid, raises InvalidStatusTransitionError otherwise.
    """
    allowed = SUBTASK_STATUS_TRANSITIONS.get(current_status, [])
    if new_status not in allowed:
        raise InvalidStatusTransitionError(current_status, new_status)
    return True


# ═══════════════════════════════════════════════════════════════════
# SubtaskFormatter
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SubtaskFormatter:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"

    _CODE_BLOCK_RE = re.compile(
        r"```(?P<lang>[a-zA-Z0-9_+-]+)?\s*\n(?P<body>.*?)\n```",
        re.DOTALL,
    )

    # ── Internal dispatcher ─────────────────────────────────────────

    def _dispatch(
        self,
        formatters: dict[str, Callable[..., Any]],
        *args: Any,
        type_hint: bool = False,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        formatter = formatters.get(self._format, formatters["json"])
        output = formatter(*args, **kwargs)
        return self._type_hint(output, type_hint)

    # ── Subtask formatters ──────────────────────────────────────────

    @staticmethod
    def _subtask_to_json(subtask: Subtask, **kwargs: Any) -> dict[str, Any]:
        return subtask.model_dump()

    @staticmethod
    def _subtask_to_markdown(subtask: Subtask, **kwargs: Any) -> str:
        return (
            f"### {subtask.title} [ID: {subtask.task_id}]\n"
            f"**Description**: {subtask.description}\n"
            f"**Status**: {subtask.status}\n"
        )

    @staticmethod
    def _subtask_to_yaml(subtask: Subtask, **kwargs: Any) -> str:
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
    def _subtask_to_xml(subtask: Subtask, indent: int = 0, **kwargs: Any) -> str:
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

    # ── Subtask-result formatters ───────────────────────────────────

    @staticmethod
    def _subtask_result_to_json(
        subtask_result: SubtaskExecutionResult, **kwargs: Any
    ) -> dict[str, Any]:
        return subtask_result.model_dump()

    @staticmethod
    def _subtask_result_to_markdown(
        subtask_result: SubtaskExecutionResult, **kwargs: Any
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
        subtask_result: SubtaskExecutionResult, **kwargs: Any
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
        subtask_result: SubtaskExecutionResult, indent: int = 0, **kwargs: Any
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

    # ── Helpers ─────────────────────────────────────────────────────

    def _type_hint(
        self,
        output: Union[str, dict[str, Any], list[dict[str, Any]]],
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(output, str) or not type_hint:
            return output
        return f"```{self._format}\n{output}\n```"

    # ── Public formatting API ───────────────────────────────────────

    def format_subtask(
        self, subtask: Subtask, type_hint: bool = False, **kwargs: Any
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable[..., Any]] = {
            "json": self._subtask_to_json,
            "markdown": self._subtask_to_markdown,
            "yaml": self._subtask_to_yaml,
            "xml": self._subtask_to_xml,
        }
        return self._dispatch(formatters, subtask, type_hint=type_hint, **kwargs)

    def format_subtasks(
        self, subtasks: list[Subtask], type_hint: bool = False
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join(self.format_subtask(subtask) for subtask in subtasks)
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            inner = "\n".join(
                self.format_subtask(subtask, indent=1) for subtask in subtasks
            )
            output = f"<subtasks>\n{inner}\n</subtasks>"
            return self._type_hint(output, type_hint)

        # json (default)
        return [self._subtask_to_json(subtask) for subtask in subtasks]

    def format_subtask_result(
        self,
        subtask_result: SubtaskExecutionResult,
        type_hint: bool = False,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        formatters: dict[str, Callable[..., Any]] = {
            "json": self._subtask_result_to_json,
            "markdown": self._subtask_result_to_markdown,
            "yaml": self._subtask_result_to_yaml,
            "xml": self._subtask_result_to_xml,
        }
        return self._dispatch(formatters, subtask_result, type_hint=type_hint, **kwargs)

    def format_subtask_results(
        self,
        subtask_results: list[SubtaskExecutionResult],
        type_hint: bool = False,
    ) -> Union[str, list[dict[str, Any]]]:
        if self._format in {"markdown", "yaml"}:
            output = "\n".join(self.format_subtask_result(r) for r in subtask_results)
            return self._type_hint(output, type_hint)

        if self._format == "xml":
            parts: list[str] = []
            for subtask_result in subtask_results:
                inner = self.format_subtask_result(subtask_result, indent=1)
                parts.append(f"<results>\n{inner}\n</results>")
            output = "\n".join(parts)
            return self._type_hint(output, type_hint)

        # json
        return [self._subtask_result_to_json(r) for r in subtask_results]

    def format_task_record(
        self, subtask: Subtask, subtask_result: SubtaskExecutionResult
    ) -> Union[str, dict[str, Any]]:
        if self._format == "json":
            record_dict: dict[str, Any] = self._subtask_to_json(subtask)
            tr = self._subtask_result_to_json(subtask_result)
            tr.pop("task_id", None)
            record_dict.update(tr)
            return record_dict

        # String-based formats
        task_str: str = self.format_subtask(subtask)  # type: ignore[assignment]
        result_str: str = self.format_subtask_result(subtask_result)  # type: ignore[assignment]
        return task_str + result_str

    # ── Result parsing ──────────────────────────────────────────────

    @staticmethod
    def _validate_result_payload(
        payload: Any,
    ) -> Optional[SubtaskExecutionResult]:
        with suppress(ValidationError, TypeError):
            return SubtaskExecutionResult.model_validate(payload)
        return None

    def _extract_fenced_blocks(self, text: str) -> list[tuple[Optional[str], str]]:
        return [
            (
                (m.group("lang") or "").strip().lower() or None,
                m.group("body").strip(),
            )
            for m in self._CODE_BLOCK_RE.finditer(text)
        ]

    @staticmethod
    def _parse_subtask_result_json(
        output: str,
    ) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        with suppress(json.JSONDecodeError, TypeError):
            parsed = json.loads(output)
            result = SubtaskFormatter._validate_result_payload(parsed)
            if result:
                return result

        if len(output) <= _MAX_LITERAL_EVAL_LEN and (
            output.startswith("{") or output.startswith("[")
        ):
            with suppress(ValueError, SyntaxError, TypeError, MemoryError):
                parsed = ast.literal_eval(output)
                result = SubtaskFormatter._validate_result_payload(parsed)
                if result:
                    return result

        return None

    @staticmethod
    def _parse_subtask_result_yaml(
        output: str,
    ) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        with suppress(yaml.YAMLError, TypeError):
            parsed = yaml.safe_load(output)

            result = SubtaskFormatter._validate_result_payload(parsed)
            if result:
                return result

            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = next(iter(parsed.values()))
                result = SubtaskFormatter._validate_result_payload(inner)
                if result:
                    return result

        return None

    @staticmethod
    def _parse_subtask_result_markdown(
        output: str,
    ) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        header_re = re.compile(
            r"(?im)^\s*#{1,6}\s*result\s*\[id:\s*(?P<task_id>[^\]]+)\]\s*$"
        )
        field_re = re.compile(
            r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?"
            r"(status|output|summary)(?:\*\*)?\s*:\s*(.*)$"
        )
        end_re = re.compile(r"(?m)^\s*---\s*$")

        task_id: Optional[str] = None
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
        m = re.search(
            r"(<result\b.*?</result>)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _parse_subtask_result_xml(
        output: str,
    ) -> Optional[SubtaskExecutionResult]:
        output = output.strip()
        if not output:
            return None

        extracted = SubtaskFormatter._extract_nested_result_xml(output)
        if not extracted:
            return None

        with suppress(ET.ParseError, AttributeError, TypeError):
            root = ET.fromstring(extracted)

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
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        return text

    @staticmethod
    def _apply_fallback_task_id(
        result: SubtaskExecutionResult,
        fallback_task_id: Optional[str],
    ) -> SubtaskExecutionResult:
        if not result.task_id and fallback_task_id:
            return result.model_copy(update={"task_id": fallback_task_id})
        return result

    def parse_subtask_result(
        self,
        output: str,
        fallback_task_id: Optional[str] = None,
    ) -> Optional[SubtaskExecutionResult]:
        output = self._sanitize_llm_output(output).strip()
        if not output:
            return None

        parsers: dict[str, Callable[[str], Optional[SubtaskExecutionResult]]] = {
            "json": self._parse_subtask_result_json,
            "markdown": self._parse_subtask_result_markdown,
            "yaml": self._parse_subtask_result_yaml,
            "xml": self._parse_subtask_result_xml,
        }

        parse_order = [self._format] + [fmt for fmt in parsers if fmt != self._format]

        # 1. Try fenced code blocks first
        fenced_blocks = self._extract_fenced_blocks(output)
        for lang, body in fenced_blocks:
            if lang in parsers:
                result = parsers[lang](body)
                if result:
                    return self._apply_fallback_task_id(result, fallback_task_id)

        # 2. If blocks exist but language didn't help, try expected format
        for _, body in fenced_blocks:
            result = parsers[self._format](body)
            if result:
                return self._apply_fallback_task_id(result, fallback_task_id)

        # 3. Try full text: expected format first, then fallback
        for fmt_name in parse_order:
            result = parsers[fmt_name](output)
            if result:
                return self._apply_fallback_task_id(result, fallback_task_id)

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


# ═══════════════════════════════════════════════════════════════════
# StreamlineManager
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StreamlineManager:
    name: str
    max_tasks: int
    fmt: SubtaskFormatter

    # ── State key helpers ───────────────────────────────────────────

    def _state_key(self, ctx: Union[ToolContext, CallbackContext]) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY, 0)
        invocation_id = ctx.invocation_id
        return f"task::{global_task_id}::{invocation_id}::{self.name}"

    def _subtasks_key(self, ctx: Union[ToolContext, CallbackContext]) -> str:
        return self._state_key(ctx) + "::tasks"

    def _current_idx_key(self, ctx: Union[ToolContext, CallbackContext]) -> str:
        return self._state_key(ctx) + "::idx"

    @staticmethod
    def _global_keys(
        ctx: Union[ToolContext, CallbackContext],
        key: Literal["", "pool", "summary", "result", "status", "objective"],
    ) -> str:
        global_task_id = ctx.state.get(_GLOBAL_TASK_ID_KEY, 0)
        if key == "":
            return f"task::{global_task_id}"
        return f"task::{global_task_id}::{key}"

    # ── ID generation ───────────────────────────────────────────────

    @staticmethod
    def _next_task_id(subtasks: list[Subtask]) -> str:
        """Return the next root-level task ID (0-based)."""
        if not subtasks:
            return "0"
        max_root = max(int(s.task_id.split(".")[0]) for s in subtasks)
        return str(max_root + 1)

    # ── Subtask persistence ─────────────────────────────────────────

    def get_subtasks(self, ctx: Union[ToolContext, CallbackContext]) -> list[Subtask]:
        key = self._subtasks_key(ctx)
        ctx.state.setdefault(key, [])
        return [Subtask(**sub) for sub in ctx.state[key]]

    def _save_subtasks(
        self,
        subtasks: list[Subtask],
        ctx: Union[ToolContext, CallbackContext],
    ) -> None:
        ctx.state[self._subtasks_key(ctx)] = [sub.model_dump() for sub in subtasks]

    @contextmanager
    def _locked_subtasks(
        self, ctx: Union[ToolContext, CallbackContext]
    ) -> Generator[list[Subtask], None, None]:
        """Load subtasks, yield mutable list, auto-save on exit."""
        subtasks = self.get_subtasks(ctx)
        yield subtasks
        self._save_subtasks(subtasks, ctx)

    # ── Index helpers ───────────────────────────────────────────────

    def _get_idx(self, ctx: Union[ToolContext, CallbackContext]) -> Optional[int]:
        return ctx.state.get(self._current_idx_key(ctx))

    def _set_idx(self, ctx: Union[ToolContext, CallbackContext], idx: int) -> None:
        ctx.state[self._current_idx_key(ctx)] = idx

    # ── Status transition ───────────────────────────────────────────

    @staticmethod
    def _apply_status_transition(subtask: Subtask, new_status: str) -> None:
        """
        Validate and apply a status transition on the given subtask.
        Raises InvalidStatusTransitionError if the transition is not allowed.
        """
        validate_status_transition(subtask.status, new_status)
        subtask.status = new_status

    # ── Core operations ─────────────────────────────────────────────

    def add_subtask(
        self,
        subtask_spec: SubtaskSpec,
        ctx: Union[ToolContext, CallbackContext],
    ) -> Optional[Subtask]:
        with self._locked_subtasks(ctx) as subtasks:
            if len(subtasks) >= self.max_tasks:
                logger.warning(
                    "Task limit reached",
                    extra={"max_tasks": self.max_tasks, "current": len(subtasks)},
                )
                return None

            new = Subtask(
                task_id=self._next_task_id(subtasks),
                title=subtask_spec.title,
                description=subtask_spec.description,
                status="new",
            )

            idx_key = self._current_idx_key(ctx)
            ctx.state.setdefault(idx_key, -1)
            idx = ctx.state[idx_key]

            should_advance = (
                idx is None
                or idx < 0
                or idx >= len(subtasks)
                or subtasks[idx].status in ("done", "skipped")
            )
            if should_advance:
                self._set_idx(ctx, len(subtasks))

            subtasks.append(new)

            logger.info(
                "Subtask added",
                extra={"task_id": new.task_id, "title": new.title},
            )

        return new

    def get_current_subtask(
        self, ctx: Union[ToolContext, CallbackContext]
    ) -> Optional[Subtask]:
        subtasks = self.get_subtasks(ctx)
        idx = self._get_idx(ctx)

        if idx is None or idx < 0 or idx >= len(subtasks):
            return None
        return subtasks[idx]

    def get_records(
        self,
        ctx: Union[ToolContext, CallbackContext],
    ) -> list[Any]:
        pool_key = self._global_keys(ctx, "pool")
        ctx.state.setdefault(pool_key, [])
        return ctx.state[pool_key]

    def save_record(
        self,
        record: Union[str, dict[str, Any]],
        ctx: Union[ToolContext, CallbackContext],
    ) -> None:
        records = self.get_records(ctx)
        records.append(record)
        pool_key = self._global_keys(ctx, "pool")
        ctx.state[pool_key] = records

    def skip(
        self,
        reason: str,
        ctx: Union[ToolContext, CallbackContext],
    ) -> Optional[Subtask]:
        idx = self._get_idx(ctx)
        if idx is None:
            return None

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return None

            if idx + 1 >= len(subtasks):
                return None

            current = subtasks[idx]

            try:
                self._apply_status_transition(current, "skipped")
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid skip transition",
                    extra={
                        "task_id": current.task_id,
                        "current_status": current.status,
                        "error": str(exc),
                    },
                )
                return None

        subtask_result = SubtaskExecutionResult(
            task_id=current.task_id,
            status="skipped",
            output=reason,
            summary="",
        )
        record = self.fmt.format_task_record(current, subtask_result)
        self.save_record(record, ctx)
        self._set_idx(ctx, idx + 1)

        logger.info(
            "Subtask skipped",
            extra={"task_id": current.task_id, "reason": reason},
        )

        return self.get_subtasks(ctx)[idx + 1]

    def decompose_current_subtask(
        self,
        new_subtasks: list[SubtaskSpec],
        ctx: Union[ToolContext, CallbackContext],
    ) -> Optional[list[Subtask]]:
        idx = self._get_idx(ctx)

        if idx is None:
            return []

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return []

            if len(subtasks) + len(new_subtasks) > self.max_tasks:
                logger.warning(
                    "Decomposition would exceed task limit",
                    extra={
                        "current_count": len(subtasks),
                        "new_count": len(new_subtasks),
                        "max_tasks": self.max_tasks,
                    },
                )
                return None

            current = subtasks[idx]

            if current.status != "incomplete":
                logger.warning(
                    "Cannot decompose non-incomplete subtask",
                    extra={
                        "task_id": current.task_id,
                        "status": current.status,
                    },
                )
                return None

            current_id: str = current.task_id

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

            for i, sub in enumerate(insertion):
                subtasks.insert(idx + 1 + i, sub)

        self._set_idx(ctx, idx + 1)

        logger.info(
            "Subtask decomposed",
            extra={
                "parent_task_id": current_id,
                "child_count": len(insertion),
                "child_ids": [s.task_id for s in insertion],
            },
        )

        return insertion

    def complete_current_subtask(
        self,
        subtask_result: SubtaskExecutionResult,
        ctx: Union[ToolContext, CallbackContext],
    ) -> tuple[bool, Optional[str]]:
        """
        Apply the execution result to the current subtask.

        Returns (success, error_message).
        """
        idx = self._get_idx(ctx)
        if idx is None:
            return False, NO_ACTIVE_TASKS_MSG

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return False, NO_ACTIVE_TASKS_MSG

            current = subtasks[idx]

            try:
                self._apply_status_transition(current, subtask_result.status)
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid status transition during completion",
                    extra={
                        "task_id": current.task_id,
                        "from_status": current.status,
                        "to_status": subtask_result.status,
                        "error": str(exc),
                    },
                )
                return False, str(exc)

        record = self.fmt.format_task_record(current, subtask_result)
        self.save_record(record, ctx)

        subtasks = self.get_subtasks(ctx)
        can_advance = idx + 1 < len(subtasks)
        if can_advance and subtask_result.status not in ("incomplete",):
            self._set_idx(ctx, idx + 1)

        logger.info(
            "Subtask completed",
            extra={
                "task_id": current.task_id,
                "status": subtask_result.status,
                "advanced": can_advance and subtask_result.status != "incomplete",
            },
        )

        return True, None

    def finish(
        self,
        status: str,
        result: str,
        summary: str,
        ctx: Union[ToolContext, CallbackContext],
    ) -> None:
        result_key = self._global_keys(ctx, "result")
        summary_key = self._global_keys(ctx, "summary")
        status_key = self._global_keys(ctx, "status")
        ctx.state[result_key] = result
        ctx.state[summary_key] = summary
        ctx.state[status_key] = status

        logger.info("Task finished", extra={"status": status})


# ═══════════════════════════════════════════════════════════════════
# Instructions & instrumentation
# ═══════════════════════════════════════════════════════════════════

TASK_RESULT_SUMMARIZATION_INSTRUCTIONS: Final[str] = """\
You are a precise technical summarizer. Your job is to produce a clear, \
factual summary of a completed task.

INPUT:
You will receive a JSON object with:
- "objective": The original goal of the task
- "records": A list of subtask execution records (each with task details and results)
- "result": The final reported result
- "status": The final task status ("done" or "failed")

OUTPUT REQUIREMENTS:
Write a single, structured summary that covers ALL of the following:

1. **Objective**: What was the task trying to achieve? (1 sentence)
2. **Approach**: What major steps were taken? (bullet list, 2-5 items)
3. **Outcome**: What was the final result? Was the objective met? (1-2 sentences)
4. **Status**: Final status and justification (1 sentence)
5. **Notable issues** (only if applicable): Blockers, warnings, partial \
failures, or important caveats

RULES:
- Be factual. Do NOT speculate or add information not present in the input.
- Be concise. Total length should be 100-300 words.
- Use past tense for completed actions.
- If status is "failed", clearly explain WHAT failed and WHY.
""".strip()


def _prepare_worker_instructions(fmt: SubtaskFormatter, type_hint: bool = False) -> str:
    example_done = SubtaskExecutionResult(
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
            "  - GET /admin/health"
        ),
        summary=(
            "Goal: Identify all HTTP endpoints in the project. "
            "Result: Found 5 endpoints across 2 controller classes. "
            "All endpoint definitions have been captured — no additional "
            "controllers exist in the codebase."
        ),
    )

    example_incomplete = SubtaskExecutionResult(
        task_id="2",
        status="incomplete",
        output=(
            "- Searched for HTTP endpoint annotations in the main source "
            "directory.\n"
            "- Found 2 endpoints in "
            "src/main/java/com/example/ExampleController.java:\n"
            "  - GET /example\n"
            "  - POST /example\n"
            "- Did not yet examine: AdminController.java, "
            "HealthController.java, and 3 other controller files."
        ),
        summary=(
            "Goal: Identify all HTTP endpoints in the project. "
            "Status: Incomplete — only 1 of 5 controller files examined. "
            "Remaining work: Inspect AdminController.java, "
            "HealthController.java, and the 3 remaining controller files "
            "for endpoint definitions."
        ),
    )

    format_description: Union[str, dict[str, Any]] = (
        fmt.format_subtask_result_description()
    )
    if isinstance(format_description, dict):
        format_description = json.dumps(format_description, indent=2)

    ex_done_fmt: Union[str, dict[str, Any]] = fmt.format_subtask_result(
        example_done, type_hint=type_hint
    )
    if isinstance(ex_done_fmt, dict):
        ex_done_fmt = json.dumps(ex_done_fmt, indent=2)

    ex_incomplete_fmt: Union[str, dict[str, Any]] = fmt.format_subtask_result(
        example_incomplete, type_hint=type_hint
    )
    if isinstance(ex_incomplete_fmt, dict):
        ex_incomplete_fmt = json.dumps(ex_incomplete_fmt, indent=2)

    return f"""\

RESPONSE FORMAT (MANDATORY):
After completing the subtask, you MUST return your result using EXACTLY \
the following structure. Do NOT add any text before or after.

FIELD DESCRIPTIONS:
{format_description}

GUIDELINES:
- task_id: Copy the exact task_id from the input. Do NOT invent a new one.
- status: Use 'done' ONLY if the subtask is 100% complete. If you could not \
finish everything, use 'incomplete'.
- output: Include concrete, factual results. List files, data, artifacts, or \
errors. No opinions or plans.
- summary: Keep it to 2-5 sentences. State the goal, what was done, and \
(if incomplete) what remains.

EXAMPLE — Completed subtask:
{ex_done_fmt}

EXAMPLE — Incomplete subtask:
{ex_incomplete_fmt}
"""


def _get_agent_ref(worker: Union[LlmAgent, AgentTool]) -> LlmAgent:
    """Extract the underlying LlmAgent from an AgentTool or return as-is."""
    if isinstance(worker, AgentTool):
        return worker.agent
    return worker


def instrument_worker(
    worker: LlmAgent,
    fmt: SubtaskFormatter,
    type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
) -> AgentTool:
    if use_input_schema or fmt._format == "json":
        worker.input_schema = Subtask

    if use_output_schema:
        worker.output_schema = SubtaskExecutionResult

    worker.instruction += _prepare_worker_instructions(fmt, type_hint=type_hint)

    return AgentTool(worker) if not isinstance(worker, AgentTool) else worker


# ═══════════════════════════════════════════════════════════════════
# Tool factory
# ═══════════════════════════════════════════════════════════════════


def task_tools(
    name: str,
    max_tasks: int,
    worker: Union[LlmAgent, AgentTool],
    fmt: SubtaskFormatter,
    *,
    use_skip: bool = True,
    use_summarization: bool = True,
    use_type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
    worker_instrumentation: bool = True,
    max_records: int = 20,
) -> list[Callable[..., Any]]:
    if worker_instrumentation:
        agent_ref = _get_agent_ref(worker)
        worker = instrument_worker(
            agent_ref, fmt, use_type_hint, use_input_schema, use_output_schema
        )

    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    mgr = StreamlineManager(name, max_tasks, fmt)

    # Pre-create summarizer tool if summarization is enabled
    summarizer_tool: Optional[AgentTool] = None
    if use_summarization:
        agent_ref = _get_agent_ref(worker)
        summarizer_agent = LlmAgent(
            name="task_summarizer",
            description="Produces structured summaries of completed task executions.",
            instruction=TASK_RESULT_SUMMARIZATION_INSTRUCTIONS,
            tools=agent_ref.tools,
            model=agent_ref.model,
        )
        summarizer_tool = AgentTool(summarizer_agent)

    # ── Tool functions ──────────────────────────────────────────────

    def add_subtask(
        title: str, description: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """Add a new subtask to the execution plan.

        Creates a subtask with the given title and description and appends it
        to the ordered task list. The subtask will be executed when it becomes
        the current subtask (i.e. all preceding subtasks are done or skipped).

        Args:
            title: Concise, action-oriented title in imperative mood.
                   Example: "Extract API endpoint definitions from source code"
            description: Detailed description including:
                - What specific work needs to be done
                - What inputs or context are available
                - What the expected output is
                - Any constraints or edge cases

        Returns:
            The created subtask on success, or an error if the task limit
            has been reached.

        When to use:
            - At the beginning, to plan the execution into ordered subtasks
            - During execution, if additional work is identified
        """
        subtask: Optional[Subtask] = mgr.add_subtask(
            SubtaskSpec(title=title, description=description), tool_context
        )
        if subtask is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}

        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """Get the single current subtask that should be executed next.

        Returns the one subtask that the worker should focus on right now.
        There is always exactly zero or one current subtask.

        Returns:
            The current subtask, or an error if no subtasks exist.

        Important:
            - You MUST call this before `execute_current_subtask` to know
              which task to execute.
            - You MUST pass the returned task_id and description to the
              worker verbatim.
            - Do NOT skip ahead or execute a different subtask.
        """
        subtask: Optional[Subtask] = mgr.get_current_subtask(tool_context)
        if subtask is None:
            return {"error": NO_ACTIVE_TASKS_MSG}

        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def list_subtasks(tool_context: ToolContext) -> dict[str, Any]:
        """List all subtasks in the execution plan with their current status.

        Use this to understand the full scope, ordering, and progress of
        the execution plan.

        Returns:
            Ordered list of all subtasks with their status.
            Status values: 'new' (pending), 'done', 'incomplete', 'skipped'.

        When to use:
            - Before adding subtasks, to see what already exists
            - To check overall progress
            - To understand context before executing the current subtask
        """
        subtasks = mgr.get_subtasks(tool_context)
        return {"result": fmt.format_subtasks(subtasks, type_hint=use_type_hint)}

    def get_records(tool_context: ToolContext) -> dict[str, Any]:
        """Retrieve execution records from previously completed subtasks.

        Each record contains the subtask definition and its execution result
        (output, summary, status). Use this to review what has already been
        done and what information has been gathered.

        Returns:
            List of task records (most recent last), capped at {max_records}.

        When to use:
            - Before executing a subtask that depends on prior results
            - When preparing the final result for `finish`
            - To avoid re-doing work that was already completed
        """
        records = mgr.get_records(tool_context)[:max_records]
        return {"result": records}

    def decompose_subtask(
        task_id: str,
        decomposition: SubtaskDecomposition,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Break an incomplete subtask into smaller, executable subtasks.

        When a subtask execution returns status 'incomplete', it means the
        work was too large or complex for a single execution. Use this tool
        to decompose it into 1-3 smaller subtasks that collectively cover
        all remaining work.

        Args:
            task_id: The task_id of the incomplete subtask. MUST match the
                     current subtask's task_id exactly.
            decomposition: A SubtaskDecomposition containing 1-3 ordered
                          subtask specifications.

        Returns:
            The list of newly created child subtasks, or an error.

        Rules:
            - Only the CURRENT subtask with status 'incomplete' can be
              decomposed.
            - Child subtasks get IDs like '{parent_id}.1', '{parent_id}.2'.
            - The first child becomes the new current subtask.
            - Prefer fewer subtasks (1-3) over many small ones.
            - Each child must be independently executable.
            - Together, children MUST cover ALL remaining work.
        """
        if isinstance(decomposition, str):
            schema = json.dumps(SubtaskDecomposition.model_json_schema(), indent=2)
            return {
                "error": (
                    "TypeError: 'decomposition' must be a SubtaskDecomposition "
                    f"object, not a string. Expected schema:\n{schema}"
                )
            }
        if isinstance(decomposition, dict):
            try:
                decomposition = SubtaskDecomposition.model_validate(decomposition)
            except ValidationError as exc:
                return {"error": f"Validation error in decomposition: {exc}"}

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_TASKS_MSG}
        if str(task_id) != current.task_id:
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}
        if current.status != "incomplete":
            return {
                "error": SUBTASK_DECOMPOSE_NOT_INCOMPLETE.format(
                    task_id=task_id, status=current.status
                )
            }

        insertion: Optional[list[Subtask]] = mgr.decompose_current_subtask(
            decomposition.subtasks, tool_context
        )

        if insertion is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}

        if len(insertion) == 0:
            return {"error": NO_ACTIVE_TASKS_MSG}

        return {"result": fmt.format_subtasks(insertion)}

    def skip(task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """Skip execution of the current subtask.

        Marks the current subtask as 'skipped' and advances to the next one.
        The skip reason is recorded for posterity.

        Args:
            task_id: The task_id of the subtask to skip. MUST match the
                     current subtask's task_id exactly.
            reason: Clear explanation of why this subtask is being skipped.
                    Examples: "Duplicate of subtask 0", "Dependency X is
                    unavailable", "Out of scope for the current objective".

        Returns:
            The next subtask (which becomes the new current subtask),
            or a message if no more subtasks remain.

        Rules:
            - Use this ONLY when you have a strong, justifiable reason.
            - Do NOT skip subtasks simply because they are difficult.
            - The reason MUST be specific and non-empty.
            - You cannot skip the last remaining subtask.
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

    async def execute_current_subtask(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Execute the current subtask using the worker agent.

        Sends the current subtask to the worker for execution, parses the
        result, records it, and advances the task pointer if appropriate.

        Prerequisites:
            - At least one subtask must exist (call `add_subtask` first)
            - The current subtask must have status 'new'
            - If the current subtask has status 'incomplete', you MUST call
              `decompose_subtask` first — do NOT call this tool again directly

        Returns:
            A dict containing:
            - 'record': The execution record (subtask + result)
            - 'action' (optional): Required next step if the subtask is
              incomplete or no more subtasks remain
            - 'error' (optional): If the worker output could not be parsed

        After this tool returns:
            - If status is 'done': The next subtask becomes current
              automatically. Call `get_current_subtask` to see it.
            - If status is 'incomplete': You MUST call `decompose_subtask`
              to break it down before proceeding.
            - If no more subtasks remain: Call `finish` to complete the task.
        """
        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_TASKS_MSG}

        logger.info(
            "Executing subtask",
            extra={"task_id": current.task_id, "title": current.title},
        )

        # Prepare worker input
        if fmt._format == "json" or use_input_schema:
            args: dict[str, Any] = fmt._subtask_to_json(current)
        else:
            args = {"request": fmt.format_subtask(current)}

        raw = await worker.run_async(args=args, tool_context=tool_context)

        # ── Parse worker output ─────────────────────────────────────
        subtask_result: Optional[SubtaskExecutionResult] = None

        if isinstance(raw, SubtaskExecutionResult):
            subtask_result = raw
        elif isinstance(raw, dict):
            with suppress(ValidationError, TypeError):
                subtask_result = SubtaskExecutionResult.model_validate(raw)
        elif isinstance(raw, str):
            subtask_result = fmt.parse_subtask_result(
                raw, fallback_task_id=current.task_id
            )

        validated = isinstance(subtask_result, SubtaskExecutionResult)

        # Fix mismatched task_id
        if validated and subtask_result.task_id != current.task_id:
            logger.warning(
                "Worker returned mismatched task_id",
                extra={
                    "expected": current.task_id,
                    "got": subtask_result.task_id,
                },
            )
            subtask_result = SubtaskExecutionResult(
                task_id=current.task_id,
                status="incomplete",
                output=(
                    f"Worker returned result for task_id="
                    f"'{subtask_result.task_id}' but expected "
                    f"'{current.task_id}'.\n\n"
                    f"Original output:\n{subtask_result.output}"
                ),
                summary=(
                    "Worker returned a result for the wrong subtask. "
                    "Marking as incomplete for retry or decomposition."
                ),
            )
            validated = False

        # Fallback for unparseable output
        if not validated:
            raw_dump = raw
            with suppress(ValueError, TypeError):
                raw_dump = json.dumps(raw, ensure_ascii=False)

            logger.warning(
                "Failed to parse worker output",
                extra={
                    "task_id": current.task_id,
                    "raw_type": type(raw).__name__,
                },
            )

            subtask_result = SubtaskExecutionResult(
                task_id=current.task_id,
                status="incomplete",
                output=str(raw_dump),
                summary=SUBTASK_RESULT_MALFORMED,
            )

        # ── Apply result via manager ────────────────────────────────
        success, error_msg = mgr.complete_current_subtask(subtask_result, tool_context)

        # ── Build response ──────────────────────────────────────────
        record = fmt.format_task_record(current, subtask_result)
        response: dict[str, Any] = {"record": record}

        if not success and error_msg:
            response["error"] = error_msg

        idx = mgr._get_idx(tool_context)
        subtasks = mgr.get_subtasks(tool_context)
        can_advance = idx is not None and idx + 1 < len(subtasks)

        action: str = ""
        if not can_advance and subtask_result.status != "incomplete":
            action += NO_ACTIVE_TASKS_MSG
        if subtask_result.status == "incomplete":
            action += SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(task_id=current.task_id)

        if action:
            response["action"] = action
        if not validated:
            response["error"] = SUBTASK_RESULT_MALFORMED

        return response

    async def finish(
        status: Literal["done", "failed"],
        result: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Finalize the overall task and report the final outcome.

        Call this when all subtasks are complete (or when you determine
        the task cannot be completed). This records the final status and
        result, optionally generates a summary, and signals the end of
        execution.

        Args:
            status: Final task status. Exactly one of:
                - "done": The global objective has been FULLY achieved.
                  All required deliverables are present in 'result'.
                - "failed": The global objective CANNOT be achieved.
                  Explain what went wrong and what was attempted in 'result'.
            result: Comprehensive, self-contained description of the outcome.
                MUST include:
                - What was accomplished (specific deliverables, data, changes)
                - What was NOT accomplished (if status is "failed")
                - All information required by the original task description
                - Follows the OUTPUT FORMAT specified in the task, if any
                This result must be understandable WITHOUT access to
                intermediate notes or execution records.

        Returns:
            Confirmation with instruction to stop execution.

        When to use:
            - All subtasks are done/skipped and the goal is achieved → "done"
            - A critical blocker prevents completion → "failed"
            - You have exhausted all approaches → "failed"

        When NOT to use:
            - There are still pending subtasks to execute
            - The current subtask is 'incomplete' (decompose it first)
        """
        summary = ""

        if use_summarization and summarizer_tool is not None:
            objective_key = StreamlineManager._global_keys(tool_context, "objective")
            objective = tool_context.state.get(objective_key, "")

            payload = {
                "objective": objective,
                "records": mgr.get_records(tool_context),
                "result": result,
                "status": status,
            }

            sum_args = {"request": json.dumps(payload, ensure_ascii=False, indent=2)}

            raw = await summarizer_tool.run_async(
                args=sum_args, tool_context=tool_context
            )

            summary = (
                raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            )

        mgr.finish(
            status=status,
            result=result,
            summary=summary,
            ctx=tool_context,
        )
        
        # Force quit
        tool_context.actions.end_of_agent = True
        return {"result": "ok", "instructions": "stop the execution now"}

    # ── Assemble tool list ──────────────────────────────────────────

    tools: list[Callable[..., Any]] = [
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
