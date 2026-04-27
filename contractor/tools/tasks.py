from __future__ import annotations

import ast
import json
import logging
import re
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
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
    "Summarize the records collected so far and call `finish` immediately."
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
SUBTASK_RESULT_MALFORMED: Final[str] = (
    "The worker returned a result that could not be completely parsed into the "
    "expected format. The raw output has been stored for reference. "
    "The subtask is marked 'malformed' — its raw results may still contain "
    "useful information. You MUST either decompose or skip it."
)
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

_GLOBAL_TASK_ID_KEY: Final[str] = "_global_task_id"
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
        description=(
            "Ordered list of 1-3 executable subtasks. Requirements:\n"
            "- Each subtask MUST be independently executable\n"
            "- Together they MUST cover ALL remaining work of the parent task\n"
            "- Order matters: subtask N may depend on results of subtask N-1\n"
            "- Prefer fewer, broader subtasks over many narrow ones"
        ),
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
    status: Literal[
        "new", "done", "incomplete", "malformed", "skipped", "decomposed"
    ] = Field(
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


# ═══════════════════════════════════════════════════════════════════
# SubtaskFormatter
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SubtaskFormatter:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"

    _CODE_BLOCK_RE: re.Pattern[str] = field(
        default_factory=lambda: re.compile(
            r"```(?P<lang>[a-zA-Z0-9_+-]+)?\s*\n(?P<body>.*?)\n```",
            re.DOTALL,
        ),
        init=False,
        repr=False,
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
            output = "\n".join(
                str(self.format_subtask(subtask)) for subtask in subtasks
            )
            return self._type_hint(output, type_hint)
        if self._format == "xml":
            inner = "\n".join(
                str(self.format_subtask(subtask, indent=1)) for subtask in subtasks
            )
            output = f"<subtasks>\n{inner}\n</subtasks>"
            return self._type_hint(output, type_hint)
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
            output = "\n".join(
                str(self.format_subtask_result(r)) for r in subtask_results
            )
            return self._type_hint(output, type_hint)
        if self._format == "xml":
            inner = "\n".join(
                str(self.format_subtask_result(r, indent=1)) for r in subtask_results
            )
            output = f"<results>\n{inner}\n</results>"
            return self._type_hint(output, type_hint)
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
        task_str = str(self.format_subtask(subtask))
        result_str = str(self.format_subtask_result(subtask_result))
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
        return m.group(1).strip() if m else None

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

        task_id_match = re.search(
            r'<result\b[^>]*\btask_id="([^"]+)"[^>]*>',
            extracted,
            flags=re.IGNORECASE | re.DOTALL,
        )
        status_match = re.search(
            r"<status>(.*?)</status>",
            extracted,
            flags=re.IGNORECASE | re.DOTALL,
        )
        output_match = re.search(
            r"<output>(.*?)</output>",
            extracted,
            flags=re.IGNORECASE | re.DOTALL,
        )
        summary_match = re.search(
            r"<summary>(.*?)</summary>",
            extracted,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not (task_id_match and status_match and output_match and summary_match):
            return None

        payload = {
            "task_id": task_id_match.group(1).strip(),
            "status": status_match.group(1).strip(),
            "output": output_match.group(1).strip(),
            "summary": summary_match.group(1).strip(),
        }
        return SubtaskFormatter._validate_result_payload(payload)

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

        # 3. Try full text in priority order
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

            idx = self._get_idx(ctx)
            should_advance = (
                idx is None
                or idx < 0
                or idx >= len(subtasks)
                or subtasks[idx].status in ("done", "skipped", "decomposed")
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
        """Skip the current subtask. Returns the next subtask or None."""
        idx = self._get_idx(ctx)
        if idx is None:
            return None

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
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

            # Determine the next subtask, if any
            next_subtask: Optional[Subtask] = None
            if idx + 1 < len(subtasks):
                self._set_idx(ctx, idx + 1)
                next_subtask = subtasks[idx + 1]

        # Build record directly — SubtaskExecutionResult doesn't allow "skipped"
        record: dict[str, Any] = {
            **current.model_dump(),
            "status": "skipped",
            "output": reason,
            "summary": f"Skipped: {reason}",
        }
        self.save_record(record, ctx)

        logger.info(
            "Subtask skipped",
            extra={"task_id": current.task_id, "reason": reason},
        )
        return next_subtask

    def decompose_current_subtask(
        self,
        new_subtasks: list[SubtaskSpec],
        ctx: Union[ToolContext, CallbackContext],
    ) -> Optional[list[Subtask]]:
        """
        Returns:
            list[Subtask] on success,
            None if preconditions not met or task limit would be exceeded.
        """
        idx = self._get_idx(ctx)
        if idx is None:
            return None

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0 or idx >= len(subtasks):
                return None

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
            if current.status not in ("incomplete", "malformed"):
                logger.warning(
                    "Cannot decompose non-decomposable subtask",
                    extra={
                        "task_id": current.task_id,
                        "status": current.status,
                    },
                )
                return None

            current_id = current.task_id
            try:
                self._apply_status_transition(current, "decomposed")
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid decompose transition",
                    extra={
                        "task_id": current.task_id,
                        "current_status": current.status,
                        "error": str(exc),
                    },
                )
                return None

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

            parent_record = {
                **current.model_dump(),
                "status": "decomposed",
                "output": (
                    f"Decomposed into {len(insertion)} child subtasks: "
                    + ", ".join(s.task_id for s in insertion)
                ),
                "summary": (
                    f"Subtask {current_id} was decomposed into "
                    f"{len(insertion)} child subtasks."
                ),
            }

        self._set_idx(ctx, idx + 1)
        self.save_record(parent_record, ctx)

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
        """Apply execution result. Returns (success, error_message)."""
        idx = self._get_idx(ctx)
        if idx is None:
            return False, NO_SUBTASKS_EXIST_MSG

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0:
                return False, NO_SUBTASKS_EXIST_MSG
            if idx >= len(subtasks):
                return False, NO_ACTIVE_SUBTASKS_MSG

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

            # Advance inside the lock for consistency
            can_advance = idx + 1 < len(subtasks)
            if can_advance and subtask_result.status not in ("incomplete",):
                self._set_idx(ctx, idx + 1)

        record = self.fmt.format_task_record(current, subtask_result)
        self.save_record(record, ctx)

        logger.info(
            "Subtask completed",
            extra={
                "task_id": current.task_id,
                "status": subtask_result.status,
                "advanced": can_advance and subtask_result.status != "incomplete",
            },
        )
        return True, None

    def complete_current_subtask_from_runtime_result(
        self,
        runtime_result: dict[str, Any],
        ctx: Union[ToolContext, CallbackContext],
    ) -> tuple[bool, Optional[str]]:
        """Apply a runtime-generated result (e.g. malformed)."""
        idx = self._get_idx(ctx)
        if idx is None:
            return False, NO_SUBTASKS_EXIST_MSG

        with self._locked_subtasks(ctx) as subtasks:
            if idx < 0:
                return False, NO_SUBTASKS_EXIST_MSG
            if idx >= len(subtasks):
                return False, NO_ACTIVE_SUBTASKS_MSG

            current = subtasks[idx]
            new_status = runtime_result["status"]
            try:
                self._apply_status_transition(current, new_status)
            except InvalidStatusTransitionError as exc:
                logger.warning(
                    "Invalid status transition during runtime completion",
                    extra={
                        "task_id": current.task_id,
                        "from_status": current.status,
                        "to_status": new_status,
                        "error": str(exc),
                    },
                )
                return False, str(exc)

            # Advance inside the lock for consistency
            can_advance = idx + 1 < len(subtasks)
            if can_advance and new_status not in ("incomplete", "malformed"):
                self._set_idx(ctx, idx + 1)

        record: dict[str, Any] = {
            **current.model_dump(),
            "status": runtime_result["status"],
            "output": runtime_result["output"],
            "summary": runtime_result["summary"],
        }
        self.save_record(record, ctx)

        logger.info(
            "Subtask completed from runtime result",
            extra={
                "task_id": current.task_id,
                "status": new_status,
                "advanced": can_advance
                and new_status not in ("incomplete", "malformed"),
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
- "records": A list of subtask execution records
- "result": The final reported result
- "status": The final task status ("done" or "failed")

OUTPUT REQUIREMENTS:
Write a single, structured summary covering:
1. **Objective**: What was the task trying to achieve?
2. **Approach**: What major steps were taken? (bullet list)
3. **Outcome**: What was the final result? Was the objective met?
4. **Status**: Final status and justification
5. **Notable issues** (only if applicable): Blockers, warnings, partial \
failures, or important caveats

RULES:
- Be factual. Do NOT speculate or add information not present in the input.
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
            "- Identified endpoints:\n"
            "  - GET /example\n"
            "  - POST /example\n"
            "  - PUT /example/id\n"
            "  - DELETE /example/id\n"
            "  - GET /admin/health"
        ),
        summary=(
            "Goal: Identify all HTTP endpoints in the project. "
            "Result: Found 5 endpoints across 2 controller classes."
        ),
    )
    example_incomplete = SubtaskExecutionResult(
        task_id="2",
        status="incomplete",
        output=(
            "- Found 2 endpoints in ExampleController.java:\n"
            "  - GET /example\n"
            "  - POST /example\n"
            "- Did not yet examine: AdminController.java, "
            "HealthController.java, and 3 other controller files."
        ),
        summary=(
            "Goal: Identify all HTTP endpoints. "
            "Status: Incomplete — only 1 of 5 controller files examined."
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
CORE RULE:
Finish the requested subtask. Avoid premature termination.
If information is missing, try to obtain or infer it using available tools.
Only stop when the final deliverable is produced or you are genuinely blocked.

STATUS RULES:
- task_id: Copy the exact task_id from the input.
- status:
  - Use 'done' ONLY if the requested deliverable is fully produced and no obvious requested work remains.
  - Do NOT return 'incomplete' status, make you best effor to complete the assigned task.
- output: Include only concrete results from work actually performed (not plans or intentions).
- summary: State the goal, what was completed, and, if incomplete, exactly what remains and why.

OUTPUT RULES:
- The output must contain findings, results, or observations — not a plan.
- Provide concrete evidence from work performed.
- Be specific about what was examined, what was found, and what was not found.
- If incomplete, explicitly state:
  - what has been completed so far
  - what remains unresolved
  - why it remains unresolved (blocking reason)
- Do NOT invent facts, findings, paths, entities, or results.
- If something could not be verified, say so explicitly.
- Use only information supported by the work you actually performed.
- Return ONLY the structured result.

RESPONSE FORMAT (MANDATORY):
After completing the subtask, you MUST return your result using EXACTLY \
the following structure. Do NOT add any text before or after.

FIELD DESCRIPTIONS:
{format_description}

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
    use_type_hint: bool = False,
    use_input_schema: bool = True,
    use_output_schema: bool = True,
    use_summarization: bool = True,
    worker_instrumentation: bool = True,
    max_records: int = 20,
    n_retries: int = 3,
) -> list[Callable[..., Any]]:
    if worker_instrumentation:
        agent_ref = _get_agent_ref(worker)
        worker = instrument_worker(
            agent_ref, fmt, use_type_hint, use_input_schema, use_output_schema
        )

    if not isinstance(worker, AgentTool):
        worker = AgentTool(worker)

    mgr = StreamlineManager(name, max_tasks, fmt)

    # Pre-create summarizer if needed
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

        Creates a subtask with the given title and description. The subtask
        will be executed when it becomes current (all preceding subtasks
        are done or skipped).

        Args:
            title: Concise, action-oriented title in imperative mood.
            description: Detailed description including what work to do,
                available inputs, expected output, and constraints.

        Returns:
            The created subtask on success, or an error if the task limit
            has been reached.

        Before calling:
        - Review `get_records` to confirm work is not already done
        - Confirm the subtask produces NEW information
        """
        subtask: Optional[Subtask] = mgr.add_subtask(
            SubtaskSpec(title=title, description=description), tool_context
        )
        if subtask is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}
        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def get_current_subtask(tool_context: ToolContext) -> dict[str, Any]:
        """Get the current subtask to execute next.

        Returns:
            The current subtask, or an error if none exist.

        You MUST call this before `execute_current_subtask`.
        """
        subtask: Optional[Subtask] = mgr.get_current_subtask(tool_context)
        if subtask is None:
            return {"error": NO_SUBTASKS_EXIST_MSG}
        return {"result": fmt.format_subtask(subtask, type_hint=use_type_hint)}

    def list_subtasks(
        tool_context: ToolContext,
        view: Literal["remaining", "all"] = "remaining",
    ) -> dict[str, Any]:
        """Inspect the execution plan without taking action.

        DEFAULT:
        - `view="remaining"` returns only the remaining planned subtasks:
        the current subtask and any later subtasks.

        OPTIONAL:
        - `view="all"` returns the full subtask history, including resolved, decomposed and
        historical subtasks.

        Args:
            view:
                - "remaining": current and future subtasks only
                - "all": full ordered history

        Returns:
            Ordered list of visible subtasks, or an explicit no-remaining-work
            message when the remaining plan is empty.
        """
        subtasks = mgr.get_subtasks(tool_context)

        if view == "all":
            visible_subtasks = subtasks
        else:
            idx = mgr._get_idx(tool_context)
            if idx is None or idx < 0 or idx >= len(subtasks):
                visible_subtasks = []
            elif idx == len(subtasks) - 1 and subtasks[idx].status != "new":
                visible_subtasks = []
            else:
                visible_subtasks = subtasks[idx:]

        if view == "remaining" and not visible_subtasks:
            return {"result": NO_REMAINING_SUBTASKS_MSG}

        return {
            "result": fmt.format_subtasks(
                visible_subtasks,
                type_hint=use_type_hint,
            )
        }

    def get_records(tool_context: ToolContext) -> dict[str, Any]:
        """Retrieve execution records from completed subtasks.

        Returns:
            List of task records (most recent last), capped at max_records.
        """
        records = mgr.get_records(tool_context)[-max_records:]
        return {"result": records}

    def decompose_subtask(
        task_id: str,
        decomposition: SubtaskDecomposition,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Break the current subtask into 1-3 smaller executable subtasks.

        Use only when the current subtask has status 'incomplete' or 'malformed'
        and multiple distinct steps remain.

        Args:
            task_id: MUST match the current subtask exactly.
            decomposition: Ordered list of 1-3 subtasks covering all
                remaining work.
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
            return {"error": NO_SUBTASKS_EXIST_MSG}
        if str(task_id) != current.task_id:
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}
        if current.status not in ("incomplete", "malformed"):
            return {
                "error": SUBTASK_DECOMPOSE_NOT_DECOMPOSABLE.format(
                    task_id=task_id, status=current.status
                )
            }

        insertion: Optional[list[Subtask]] = mgr.decompose_current_subtask(
            decomposition.subtasks, tool_context
        )
        if insertion is None:
            return {"error": TASK_LIMIT_REACHED_MSG.format(max_tasks=max_tasks)}
        if len(insertion) == 0:
            return {"error": SUBTASK_DECOMPOSE_EMPTY_LIST}
        return {"result": fmt.format_subtasks(insertion)}

    def skip(task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """Skip execution of the current subtask.

        Marks the current subtask as 'skipped' and advances to the next one.
        If all objectives are achived, use `finish` tool instead.

        Args:
            task_id: Must match the current subtask's task_id exactly.
            reason: Clear, specific explanation of why this subtask is being
                    skipped. Generic reasons like "not needed" or "too hard"
                    are NOT acceptable.

        Returns:
            The next subtask, or a message if no more remain.

        IMPORTANT CONSTRAINTS:
            - You MUST have attempted execution first or have clear evidence
              the subtask cannot produce useful results.
            - Valid reasons: duplicate of another subtask, dependency
              unavailable, provably out of scope, already covered by
              another subtask's output.
            - INVALID reasons: "difficult", "not sure", "might not work",
              "seems unnecessary".
            - For 'malformed' subtasks: prefer `decompose_subtask` over skip,
              since malformed output may contain useful partial information.
            - For 'incomplete' subtasks: you MUST decompose unless this is the
              last remaining subtask, in which case skip is allowed.
        """
        if not reason.strip():
            return {"error": SKIP_REASON_MUST_NOT_BE_EMPTY}

        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"result": NO_ACTIVE_SUBTASKS_MSG}
        if str(task_id) != current.task_id:
            return {"error": SUBTASK_NOT_CURRENT_MSG.format(task_id=task_id)}

        subtasks = mgr.get_subtasks(tool_context)
        is_last_subtask = len(subtasks) > 0 and subtasks[-1].task_id == current.task_id
        limit_reached = len(subtasks) >= mgr.max_tasks

        if current.status == "incomplete" and not is_last_subtask and not limit_reached:
            return {
                "error": (
                    f"Subtask `{task_id}` has status 'incomplete' and cannot be "
                    f"skipped unless it is the last remaining subtask. Call `decompose_subtask` on {task_id}."
                )
            }

        next_subtask = mgr.skip(reason, tool_context)
        if next_subtask is None:
            return {"result": NO_ACTIVE_SUBTASKS_MSG}
        return {"result": "ok", "next-subtask": fmt.format_subtask(next_subtask)}

    async def execute_current_subtask(
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Execute the current subtask using the worker agent.

        Prerequisites:
            - At least one subtask must exist
            - Current subtask must have status 'new'
            - If 'incomplete'/'malformed', resolve it by decomposing or skipping first

        Returns:
            Record of execution with optional action guidance.

        After this tool returns:
            - If status is 'done': The next subtask becomes current automatically when one exists.
            - If status is 'incomplete': You MUST call `decompose_subtask` or `skip` before proceeding.
            - If status is 'malformed': The raw output has been stored, but the
            result could not be fully parsed. You MUST call `decompose_subtask`
            or `skip` before proceeding.
        """
        current = mgr.get_current_subtask(tool_context)
        if current is None:
            return {"error": NO_ACTIVE_SUBTASKS_MSG}

        match current.status:
            case "malformed":
                return {
                    "error": SUBTASK_REQUIRES_RESOLUTION_MSG.format(
                        task_id=current.task_id,
                        status=current.status,
                    )
                }
            case "incomplete":
                return {
                    "error": SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
                        task_id=current.task_id,
                    )
                }
            case "done" | "skipped" | "decomposed":
                return {"error": NO_ACTIVE_SUBTASKS_MSG}
            case "new":
                pass
            case _:
                logger.warning(
                    "Unknown subtask status",
                    extra={
                        "task_id": current.task_id,
                        "status": current.status,
                    },
                )
                return {
                    "error": (
                        f"Subtask `{current.task_id}` has unsupported "
                        f"status '{current.status}'."
                    )
                }

        logger.info(
            "Executing subtask",
            extra={"task_id": current.task_id, "title": current.title},
        )

        # Prepare worker input
        if fmt._format == "json" or use_input_schema:
            args: dict[str, Any] = fmt._subtask_to_json(current)
        else:
            args = {"request": fmt.format_subtask(current)}

        # Run worker with retries
        retries = 0
        raw: Any = ""
        while retries < n_retries:
            raw = await worker.run_async(args=args, tool_context=tool_context)

            if isinstance(raw, str):
                if raw.strip():
                    break
            elif raw is not None:
                break

            retries += 1

        # ── Parse worker output ──────────────────────────────────────
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

        # Check for task_id mismatch
        raw_dump: Any = raw
        malformed_reason: Optional[str] = None
        if (
            validated
            and subtask_result is not None
            and subtask_result.task_id != current.task_id
        ):
            logger.warning(
                "Worker returned mismatched task_id",
                extra={
                    "expected": current.task_id,
                    "got": subtask_result.task_id,
                },
            )
            malformed_reason = (
                f"Worker returned result for task_id='{subtask_result.task_id}' "
                f"but expected '{current.task_id}'.\n\n"
                f"Original parsed output:\n{subtask_result.output}"
            )
            validated = False
            subtask_result = None

        # ── Apply malformed fallback ─────────────────────────────────
        if not validated:
            if malformed_reason is not None:
                raw_dump = malformed_reason
            with suppress(ValueError, TypeError):
                raw_dump = json.dumps(raw_dump, ensure_ascii=False)

            logger.warning(
                "Failed to parse worker output",
                extra={
                    "task_id": current.task_id,
                    "raw_type": type(raw).__name__,
                },
            )
            runtime_result = {
                "task_id": current.task_id,
                "status": "malformed",
                "output": str(raw_dump),
                "summary": SUBTASK_RESULT_MALFORMED,
            }
            success, error_msg = mgr.complete_current_subtask_from_runtime_result(
                runtime_result, tool_context
            )
            record: dict[str, Any] = {
                **current.model_dump(),
                **runtime_result,
            }
            response: dict[str, Any] = {
                "record": record,
                "error": SUBTASK_RESULT_MALFORMED,
                "action": SUBTASK_REQUIRES_RESOLUTION_MSG.format(
                    task_id=current.task_id,
                    status="malformed",
                ),
            }
            if not success and error_msg:
                response["error"] = error_msg
            return response

        # Defensive check instead of assert
        if subtask_result is None:
            logger.warning(
                "Validated execution path reached with no parsed subtask result",
                extra={"task_id": current.task_id},
            )
            runtime_result = {
                "task_id": current.task_id,
                "status": "malformed",
                "output": str(raw),
                "summary": SUBTASK_RESULT_MALFORMED,
            }
            success, error_msg = mgr.complete_current_subtask_from_runtime_result(
                runtime_result, tool_context
            )
            record: dict[str, Any] = {
                **current.model_dump(),
                **runtime_result,
            }
            response: dict[str, Any] = {
                "record": record,
                "error": SUBTASK_RESULT_MALFORMED,
                "action": SUBTASK_REQUIRES_RESOLUTION_MSG.format(
                    task_id=current.task_id,
                    status="malformed",
                ),
            }
            if not success and error_msg:
                response["error"] = error_msg
            return response

        # ── Apply validated result ───────────────────────────────────
        success, error_msg = mgr.complete_current_subtask(subtask_result, tool_context)

        record = fmt.format_task_record(current, subtask_result)
        response: dict[str, Any] = {"record": record}

        if not success and error_msg:
            response["error"] = error_msg

        # Inspect the post-update state directly instead of inferring it from idx math.
        next_current = mgr.get_current_subtask(tool_context)
        has_active_subtask = next_current is not None and next_current.status == "new"

        action_parts: list[str] = []
        if subtask_result.status == "incomplete":
            action_parts.append(
                SUBTASK_REQUIRES_DECOMPOSITION_MSG.format(
                    task_id=current.task_id,
                )
            )
        elif not has_active_subtask:
            action_parts.append(NO_ACTIVE_SUBTASKS_MSG)

        if action_parts:
            response["action"] = " ".join(action_parts)

        return response

    async def finish(
        status: Literal["done", "failed"],
        result: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Finalize the overall task and report the final outcome.

        Args:
            status: "done" if objective fully achieved, "failed" otherwise.
            result: Comprehensive, self-contained description of outcome.
                    MUST include:
                    - What was accomplished (specific deliverables, data, changes)
                    - What was NOT accomplished (if status is "failed")
                    - All information required by the original task description
                    - Follows the OUTPUT FORMAT specified in the task, if any
                    This result must be understandable WITHOUT access to
                    intermediate notes or execution records.

        When to use:
            - All subtasks done/skipped and goal achieved → "done"
            - Critical blocker prevents completion → "failed"
        """

        subtasks = mgr.get_subtasks(tool_context)
        if status == "done":
            has_new = any(t.status == "new" for t in subtasks)
            has_any = len(subtasks) > 0
            if has_new or not has_any:
                return {"error": DO_NOT_FINISH_WITH_NO_TASKS_DONE}

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
        tool_context._invocation_context.end_invocation = True
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
