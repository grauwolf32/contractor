from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from google.adk.tools.tool_context import ToolContext
from google.genai import types

from contractor.tools.result import aguard


class OpenApiLinterError(Exception):
    """Base exception for OpenApiLinter errors."""


class VacuumNotFoundError(OpenApiLinterError):
    """Raised when the vacuum binary is not found in PATH."""


class VacuumExecutionError(OpenApiLinterError):
    """Raised when vacuum execution fails."""

    def __init__(self, message: str, details: str = ""):
        self.details = details
        super().__init__(message)


class VacuumOutputError(OpenApiLinterError):
    """Raised when vacuum output cannot be parsed or has unexpected format."""

    def __init__(self, message: str, details: str = "", raw_output: Any = None):
        self.details = details
        self.raw_output = raw_output
        super().__init__(message)


def _format_linter_error(exc: OpenApiLinterError) -> dict[str, Any]:
    """Error envelope enriched with the exception's ``details`` / ``raw_output``
    when present (base errors carry neither)."""
    error_dict: dict[str, Any] = {"error": str(exc)}
    details = getattr(exc, "details", "")
    if details:
        error_dict["details"] = details
    raw_output = getattr(exc, "raw_output", None)
    if raw_output is not None:
        error_dict["raw_output"] = raw_output
    return error_dict


@dataclass
class OpenApiLinter:
    """
    OpenAPI linter that uses the Vacuum binary to run spectral reports.
    Raises on errors instead of returning error dicts.
    """

    name: str
    version: int | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._ensure_vacuum()

    @staticmethod
    def _ensure_vacuum() -> None:
        """
        Checks if the `vacuum` binary is available in PATH.

        Raises:
            VacuumNotFoundError: If vacuum binary is not found.
        """
        if shutil.which("vacuum") is None:
            raise VacuumNotFoundError("vacuum binary not found in PATH")

    def artifact_key(self) -> str:
        return f"user:oas-{self.name}"

    async def load_artifact(self, ctx: ToolContext) -> str:
        """
        Loads the OpenAPI artifact from the tool context.

        Args:
            ctx: The tool context to load the artifact from.

        Returns:
            The OpenAPI specification as a YAML string.

        Raises:
            OpenApiLinterError: If the artifact cannot be loaded.
        """
        async with self._lock:
            artifact = await ctx.load_artifact(filename=self.artifact_key())
            if artifact is None:
                raise OpenApiLinterError(
                    f"artifact '{self.artifact_key()}' not found in context"
                )
            if isinstance(artifact, types.Part):
                return artifact.text or ""
            if isinstance(artifact, list) and len(artifact) > 0:
                return artifact[0].text
            raise OpenApiLinterError(
                f"artifact '{self.artifact_key()}' has unexpected format"
            )

    @staticmethod
    def extract_snippet(
        source_text: str,
        start_line: int,
        start_character: int,
        end_line: int,
        end_character: int,
    ) -> str:
        """
        Extracts a snippet from the original source text using Spectral/Vacuum
        range coordinates.

        Args:
            source_text: Original OpenAPI source text.
            start_line: 1-based start line number.
            start_character: 0-based start character offset.
            end_line: 1-based end line number.
            end_character: 0-based end character offset.

        Returns:
            Extracted snippet, or an empty string if the coordinates are invalid.
        """
        lines = source_text.splitlines()
        if (
            start_line < 1
            or end_line < 1
            or start_line > len(lines)
            or end_line > len(lines)
            or start_line > end_line
        ):
            return ""

        start_idx = start_line - 1
        end_idx = end_line - 1

        if start_idx == end_idx:
            line = lines[start_idx]
            return line[start_character:end_character]

        parts = [lines[start_idx][start_character:]]
        for idx in range(start_idx + 1, end_idx):
            parts.append(lines[idx])
        parts.append(lines[end_idx][:end_character])
        return "\n".join(parts)

    def replace_range_with_snippet(
        self,
        issue: dict[str, Any],
        source_text: str,
    ) -> dict[str, Any]:
        """
        Returns a copy of the issue where `range` is replaced with `snippet`.
        All other fields are preserved unchanged.

        Args:
            issue: Original Vacuum/Spectral issue object.
            source_text: Original OpenAPI source text.

        Returns:
            A new issue dict with `snippet` instead of `range`.
        """
        result = dict(issue)
        issue_range = result.pop("range", None)
        snippet = ""

        if isinstance(issue_range, dict):
            start = issue_range.get("start")
            end = issue_range.get("end")
            if isinstance(start, dict) and isinstance(end, dict):
                start_line = start.get("line")
                start_character = start.get("character")
                end_line = end.get("line")
                end_character = end.get("character")
                if all(
                    isinstance(value, int)
                    for value in (
                        start_line,
                        start_character,
                        end_line,
                        end_character,
                    )
                ):
                    snippet = self.extract_snippet(
                        source_text=source_text,
                        start_line=start_line,
                        start_character=start_character,
                        end_line=end_line,
                        end_character=end_character,
                    )

        result["snippet"] = snippet
        return result

    def process_issues(
        self,
        issues: list[dict[str, Any]],
        source_text: str,
        include_severities: Iterable[int] = (1, 2),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filters issues by severity, sorts them in descending severity order,
        limits the result size, and replaces `range` with `snippet`.

        Args:
            issues: Raw Vacuum/Spectral issue list.
            source_text: Original OpenAPI source text.
            include_severities: Severity levels to include. Defaults to (1, 2).
            limit: Maximum number of issues to return.

        Returns:
            Processed issue list.
        """
        allowed = set(include_severities)
        filtered = [
            issue
            for issue in issues
            if isinstance(issue.get("severity"), int) and issue["severity"] in allowed
        ]
        filtered.sort(key=lambda item: item["severity"], reverse=True)

        if limit is not None:
            filtered = filtered[:limit]

        return [
            self.replace_range_with_snippet(issue=issue, source_text=source_text)
            for issue in filtered
        ]

    def lint(
        self,
        openapi_str: str,
        include_severities: Iterable[int] = (1, 2),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Runs Vacuum spectral-report on an OpenAPI string via stdin.

        Args:
            openapi_str: OpenAPI specification as YAML or JSON string.
            include_severities: Severity levels to include in output.
                Defaults to (1, 2), excluding severity 0.
            limit: Maximum number of issues to return.

        Returns:
            A list of processed issues.

        Raises:
            VacuumNotFoundError: If vacuum binary is not found.
            VacuumExecutionError: If vacuum process fails to run or returns
                an error.
            VacuumOutputError: If vacuum output cannot be parsed or has
                unexpected format.
        """
        self._ensure_vacuum()

        try:
            process = subprocess.run(
                ["vacuum", "spectral-report", "-i", "-o"],
                input=openapi_str.encode("utf-8"),
                capture_output=True,
            )
        except Exception as exc:
            raise VacuumExecutionError(f"failed to run vacuum: {exc}") from exc

        if process.returncode not in (0, 1):
            raise VacuumExecutionError(
                "vacuum execution failed",
                details=process.stderr.decode("utf-8", errors="ignore"),
            )

        try:
            parsed = json.loads(process.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise VacuumOutputError(
                "failed to parse vacuum output",
                details=str(exc),
                raw_output=process.stdout.decode("utf-8", errors="ignore"),
            ) from exc

        if not isinstance(parsed, list):
            raise VacuumOutputError(
                "unexpected vacuum output format",
                details=(f"expected a list of issues, got {type(parsed).__name__}"),
                raw_output=parsed,
            )

        return self.process_issues(
            issues=parsed,
            source_text=openapi_str,
            include_severities=include_severities,
            limit=limit,
        )


def openapi_linter_tools(name: str) -> list[Callable[..., Any]]:
    """
    Creates an instance of OpenApiLinter and returns tool functions
    that return dicts with `result` or `error` keys.

    Args:
        name: The name identifier for the OpenAPI artifact.

    Returns:
        A list of tool functions.
    """

    linter = OpenApiLinter(name=name)

    async def lint_openapi(
        ctx: ToolContext,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Runs Vacuum spectral-report on the current OpenAPI artifact.

        Returns only serious issues (severity 2). Lower-severity findings
        (warnings, hints) are intentionally suppressed so the caller can
        focus repair effort on schema-breaking problems.

        Args:
            ctx: The tool context for loading artifacts.
            limit: Maximum number of issues to return.

        Returns:
            A dict with processed issues under `result`, or an `error` dict.
        """

        async def _impl() -> dict[str, Any]:
            try:
                openapi_str = await linter.load_artifact(ctx)
                result = await asyncio.to_thread(
                    linter.lint,
                    openapi_str=openapi_str,
                    include_severities=(2,),
                    limit=limit,
                )
            except OpenApiLinterError as exc:
                return _format_linter_error(exc)
            return {"result": result}

        # aguard is the outer net for *unexpected* faults; the inner handler
        # keeps the rich per-error metadata (details / raw_output).
        return await aguard(_impl)

    return [lint_openapi]
