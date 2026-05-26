from __future__ import annotations

import ast
import json
import re
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union
from xml.sax.saxutils import escape as xml_escape

import yaml
from pydantic import ValidationError

from contractor.tools.tasks.models import (
    Subtask,
    SubtaskExecutionResult,
    _MAX_LITERAL_EVAL_LEN,
)


def _stringify_formatted(value: Union[str, dict[str, Any]]) -> str:
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    return value


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

    @property
    def format(self) -> str:
        """Public read-only accessor for the active output format."""
        return self._format

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


def _is_empty_worker_response(raw: Any) -> bool:
    """True when the worker returned no usable content (None or blank string)."""
    if raw is None:
        return True
    return isinstance(raw, str) and not raw.strip()


def _parse_worker_output(
    raw: Any,
    fmt: "SubtaskFormatter",
    fallback_task_id: str,
) -> Optional["SubtaskExecutionResult"]:
    """Coerce a worker response into a SubtaskExecutionResult, or None.

    Accepts an already-typed result, a dict matching the schema, or a string
    in any of the formatter's supported output formats. Does NOT verify that
    the parsed task_id matches the requested one — caller's responsibility.
    """
    if isinstance(raw, SubtaskExecutionResult):
        return raw
    if isinstance(raw, dict):
        with suppress(ValidationError, TypeError):
            return SubtaskExecutionResult.model_validate(raw)
        return None
    if isinstance(raw, str):
        return fmt.parse_subtask_result(raw, fallback_task_id=fallback_task_id)
    return None
