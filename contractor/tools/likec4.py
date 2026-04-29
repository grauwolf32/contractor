from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_RUNNER_FALLBACKS: tuple[str, ...] = ("bunx", "pnpx", "npx")
_DEFAULT_FILENAME = "main.c4"


class Likec4Error(Exception):
    """Base exception for LikeC4 linter errors."""


class Likec4NotFoundError(Likec4Error):
    """Raised when neither `likec4` nor a fallback runner (bunx/pnpx/npx) is available."""


class Likec4ExecutionError(Likec4Error):
    def __init__(self, message: str, details: str = "") -> None:
        self.details = details
        super().__init__(message)


class Likec4OutputError(Likec4Error):
    def __init__(self, message: str, details: str = "", raw_output: Any = None) -> None:
        self.details = details
        self.raw_output = raw_output
        super().__init__(message)


@dataclass
class Likec4Linter:
    """
    LikeC4 validator that wraps the `likec4` CLI (or a JS package runner like
    `bunx`/`pnpx`/`npx` as fallback).

    Validates a LikeC4 DSL string by writing it to a temporary project directory
    and running `likec4 validate --json --no-layout --file <tmp> <tmpdir>`. The
    agent owns the source string (typically kept in its memory store) and
    passes it in directly — no on-disk artifact is needed.
    """

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._resolve_command()

    @staticmethod
    def _resolve_command() -> list[str]:
        """
        Returns the command prefix used to invoke likec4 (without subcommand).

        Raises:
            Likec4NotFoundError: If no usable runner is found in PATH.
        """
        if shutil.which("likec4") is not None:
            return ["likec4"]
        for runner in _RUNNER_FALLBACKS:
            if shutil.which(runner) is not None:
                return [runner, "likec4"]
        raise Likec4NotFoundError(
            "likec4 not found in PATH and no fallback runner available "
            "(tried: likec4, bunx, pnpx, npx)"
        )

    def validate(self, content: str) -> list[dict[str, Any]]:
        """
        Runs `likec4 validate --json --no-layout` against a single in-memory
        LikeC4 DSL string.

        Args:
            content: Full source of a `.c4`/`.likec4` file.

        Returns:
            Parsed list of issue objects emitted by likec4. Empty list means
            no issues.

        Raises:
            Likec4NotFoundError: If likec4/runner binary is not available.
            Likec4ExecutionError: If the likec4 process fails to run.
            Likec4OutputError: If output cannot be parsed or has unexpected shape.
        """
        cmd_prefix = self._resolve_command()

        with tempfile.TemporaryDirectory(prefix="likec4-") as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / _DEFAULT_FILENAME
            source_path.write_text(content, encoding="utf-8")

            cmd = cmd_prefix + [
                "validate",
                "--json",
                "--no-layout",
                "--file",
                str(source_path),
                str(tmp_path),
            ]

            try:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:
                raise Likec4ExecutionError(f"failed to run likec4: {exc}") from exc

            stdout = process.stdout.decode("utf-8", errors="ignore")
            stderr = process.stderr.decode("utf-8", errors="ignore")

        # likec4 validate exits non-zero when the project has issues — that's
        # not an execution error, the JSON payload still contains the report.
        if not stdout.strip():
            raise Likec4ExecutionError(
                "likec4 produced no output",
                details=stderr or f"exit code {process.returncode}",
            )

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise Likec4OutputError(
                "failed to parse likec4 output",
                details=str(exc),
                raw_output=stdout,
            ) from exc

        if not isinstance(parsed, list):
            raise Likec4OutputError(
                "unexpected likec4 output format",
                details=f"expected a list of issues, got {type(parsed).__name__}",
                raw_output=parsed,
            )

        return parsed


def likec4_tools() -> list[Callable[..., Any]]:
    """
    Creates a Likec4Linter and returns tool functions.

    Returns:
        A list of tool functions returning `{"result": ...}` or `{"error": ...}`.
    """

    linter = Likec4Linter()

    async def validate_likec4(content: str) -> dict[str, Any]:
        """
        Validates a LikeC4 DSL string via `likec4 validate` and returns parsed issues.

        Args:
            content: Full source text of a `.c4`/`.likec4` file (typically held
                by the agent in its memory store).

        Returns:
            On success: {"result": [<issue>, ...]}. Empty list means no issues.
            On failure: {"error": "...", "details"?: "...", "raw_output"?: ...}.
        """
        try:
            issues = await asyncio.to_thread(linter.validate, content)
            return {"result": issues}
        except Likec4ExecutionError as exc:
            out: dict[str, Any] = {"error": str(exc)}
            if exc.details:
                out["details"] = exc.details
            return out
        except Likec4OutputError as exc:
            out = {"error": str(exc)}
            if exc.details:
                out["details"] = exc.details
            if exc.raw_output is not None:
                out["raw_output"] = exc.raw_output
            return out
        except Likec4Error as exc:
            return {"error": str(exc)}

    return [validate_likec4]
