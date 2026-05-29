from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fsspec import AbstractFileSystem

from contractor.tools.result import aguard
from contractor.utils.settings import get_settings

_RUNNER_FALLBACKS: tuple[str, ...] = ("bunx", "pnpx", "npx")
_DEFAULT_FILENAME = "main.c4"
DEFAULT_LIKEC4_PATH: str = "/architecture.c4"
_DEFAULT_TIMEOUT_S: float = get_settings().likec4_validate_timeout

# Auto-confirm flags for fallback package runners. Without these, a runner
# that needs to fetch `likec4` on first use will wait on stdin for a
# "Ok to proceed?" prompt and hang the agent.
_RUNNER_AUTOCONFIRM: dict[str, list[str]] = {
    "npx": ["--yes"],
    "pnpx": ["--yes"],
    "bunx": [],
}


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


class Likec4SourceNotFoundError(Likec4Error):
    """Raised when the LikeC4 source file does not exist on the configured fs."""


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
                return [runner, *_RUNNER_AUTOCONFIRM.get(runner, []), "likec4"]
        raise Likec4NotFoundError(
            "likec4 not found in PATH and no fallback runner available "
            "(tried: likec4, bunx, pnpx, npx)"
        )

    def validate(
        self, content: str, *, timeout: float = _DEFAULT_TIMEOUT_S
    ) -> list[dict[str, Any]]:
        """
        Runs `likec4 validate --json --no-layout` against a single in-memory
        LikeC4 DSL string.

        Args:
            content: Full source of a `.c4`/`.likec4` file.
            timeout: Seconds to wait for the likec4 process. The call hard-fails
                with `Likec4ExecutionError` past this — prevents an unattended
                agent from hanging on an interactive prompt or stuck install.

        Returns:
            Parsed list of issue objects emitted by likec4. Empty list means
            no issues.

        Raises:
            Likec4NotFoundError: If likec4/runner binary is not available.
            Likec4ExecutionError: If the likec4 process fails to run or times out.
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
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                stderr_tail = (
                    exc.stderr.decode("utf-8", errors="ignore")
                    if isinstance(exc.stderr, (bytes, bytearray))
                    else ""
                )
                raise Likec4ExecutionError(
                    f"likec4 timed out after {timeout:.0f}s",
                    details=stderr_tail or " ".join(cmd),
                ) from exc
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

        # likec4 normally emits a single JSON document. npx/bunx may prepend
        # banners (e.g. "Update available") before it, so if a direct parse
        # fails, retry from each '{'/'[' offset to skip the banner. A sentinel
        # (not None) tracks "never parsed" so a literal JSON `null` isn't
        # mistaken for a parse failure.
        unparsed = object()
        parsed: object = unparsed
        last_exc: json.JSONDecodeError | None = None
        try:
            parsed = json.loads(stdout.strip())
        except json.JSONDecodeError as exc:
            last_exc = exc
            for i, ch in enumerate(stdout):
                if ch in ('{', '['):
                    try:
                        parsed = json.loads(stdout[i:])
                        break
                    except json.JSONDecodeError as exc2:
                        last_exc = exc2
                        continue

        if parsed is unparsed:
            raise Likec4OutputError(
                "failed to parse likec4 output",
                details=str(last_exc) if last_exc else "no JSON found",
                raw_output=stdout,
            )

        # `likec4 validate --json` emits {"valid": bool, "errors": [...], "stats": {...}}.
        # Older/alternate builds emit a bare issue list. Accept both.
        if isinstance(parsed, dict):
            errors = parsed.get("errors")
            if not isinstance(errors, list):
                raise Likec4OutputError(
                    "unexpected likec4 output format",
                    details=(
                        "expected dict with list 'errors', "
                        f"got {type(errors).__name__} (keys: {sorted(parsed)})"
                    ),
                    raw_output=parsed,
                )
            return errors

        if isinstance(parsed, list):
            return parsed

        raise Likec4OutputError(
            "unexpected likec4 output format",
            details=f"expected list or dict, got {type(parsed).__name__}",
            raw_output=parsed,
        )

    def validate_path(
        self,
        fs: AbstractFileSystem,
        path: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> list[dict[str, Any]]:
        """
        Reads a LikeC4 source file from `fs` (overlay-aware) and validates it.

        Args:
            fs: fsspec filesystem; for the agent this is the overlay over the
                project root, so unsaved edits are visible.
            path: Path to the LikeC4 source file on `fs`.
            timeout: Forwarded to :meth:`validate`.

        Raises:
            Likec4SourceNotFoundError: If `path` does not exist on `fs`.
        """
        if not fs.exists(path):
            raise Likec4SourceNotFoundError(
                f"likec4 source file not found at {path!r}"
            )
        content = fs.read_text(path, encoding="utf-8")
        return self.validate(content, timeout=timeout)


def likec4_tools(
    fs: AbstractFileSystem,
    *,
    default_path: str = DEFAULT_LIKEC4_PATH,
) -> list[Callable[..., Any]]:
    """
    Creates a Likec4Linter wired to the given filesystem and returns tool
    functions.

    The agent maintains its LikeC4 source as a single file on `fs` (overlay-
    backed for the build agent), so `validate_likec4` re-reads from disk on
    every call and no separate in-memory copy of the source is needed.

    Args:
        fs: fsspec filesystem (typically an overlay over the project root).
        default_path: Path used when `validate_likec4` is called with no
            argument. Defaults to ``/architecture.c4``.
    """

    linter = Likec4Linter()

    async def validate_likec4(path: str = default_path) -> dict[str, Any]:
        """
        Validates a LikeC4 source file by running `likec4 validate --json` against it.

        The file is read from the agent's filesystem (overlay-aware), so any
        edits made via write_file / edit / replace_range are picked up
        immediately.

        Args:
            path: Path to the LikeC4 source file. Defaults to
                ``/architecture.c4``.

        Returns:
            On success: {"result": [<issue>, ...]}. Empty list means no issues.
            On failure: {"error": "...", "details"?: "...", "raw_output"?: ...}.
        """

        async def _impl() -> dict[str, Any]:
            try:
                issues = await asyncio.to_thread(linter.validate_path, fs, path)
                return {"result": issues}
            except Likec4SourceNotFoundError as exc:
                return {"error": str(exc)}
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

        # aguard is the outer net for *unexpected* faults; the inner handlers
        # keep the rich per-error metadata (details / raw_output).
        return await aguard(_impl)

    return [validate_likec4]
