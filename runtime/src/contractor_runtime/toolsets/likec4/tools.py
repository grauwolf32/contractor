"""Namespace-bound, CAS-backed tools for one single-file LikeC4 document."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import ProxySubprocessLauncher
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.probe import executable_responds
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _unconfigured_client,
    runtime_secrets,
)
from contractor_runtime.toolsets.common.document_session import (
    DocumentSession,
    validate_target_name,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.common.line_window import bounded_line_window
from contractor_runtime.toolsets.common.lines import split_lines
from contractor_runtime.toolsets.common.process import ProcessOutputLimitError, run_tool_command
from contractor_runtime.toolsets.common.tool_base import SessionArtifactTool
from contractor_runtime.workspace import AllocationWorkspace

MAX_DOCUMENT_UTF8_BYTES = 1024 * 1024
MAX_VISIBLE_UTF8_BYTES = 128 * 1024
MAX_READ_LINES = 400
DEFAULT_READ_LINES = 200
MAX_REPLACE_OCCURRENCES = 100
MAX_VALIDATOR_OUTPUT_BYTES = 1024 * 1024
# Banner-tolerant JSON extraction tries at most this many candidate starts.
MAX_JSON_FALLBACK_STARTS = 64
MAX_VALIDATION_DIAGNOSTICS = 100
MAX_DIAGNOSTIC_TEXT_BYTES = 4096
MAX_DIAGNOSTIC_ITEMS = 32
MAX_DIAGNOSTIC_DEPTH = 6
VALIDATE_TIMEOUT_SECONDS = 30
DEFAULT_TARGET_NAME = "architecture"
TARGET_MEDIA_TYPE = "text/vnd.likec4"
VALIDATOR_FILENAME = "main.c4"


class LikeC4ToolsetFactory:
    ref = "likec4@1"
    exported_tools = frozenset(
        {
            "load_likec4",
            "write_likec4",
            "read_likec4",
            "append_likec4",
            "replace_likec4",
            "validate_likec4",
        }
    )
    infrastructure_channels = MappingProxyType(
        {"validate_likec4": frozenset({"runtime-subprocess-launcher"})}
    )

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        available = set(self.exported_tools)
        if not await executable_responds("likec4", ("version",)):
            available.discard("validate_likec4")
        return frozenset(available)

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(state, "likec4@1")
        client = self._client_factory(allocation_id, runtime_settings)
        launcher = adapter_handles.tool_subprocess
        if launcher is not None and not isinstance(launcher, ProxySubprocessLauncher):
            raise TypeError("likec4@1 received an invalid subprocess handle")
        session = _LikeC4Session(client, namespace, workspace.path, launcher)
        secrets = runtime_secrets(runtime_settings)
        builders: dict[str, Callable[[], Any]] = {
            "load_likec4": lambda: LoadLikeC4Tool(session, client, metrics, secrets),
            "write_likec4": lambda: WriteLikeC4Tool(session, client, metrics, secrets),
            "read_likec4": lambda: ReadLikeC4Tool(session, client, metrics, secrets),
            "append_likec4": lambda: AppendLikeC4Tool(session, client, metrics, secrets),
            "replace_likec4": lambda: ReplaceLikeC4Tool(session, client, metrics, secrets),
            "validate_likec4": lambda: ValidateLikeC4Tool(session, client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


class _LikeC4Session(DocumentSession[str]):
    media_type = TARGET_MEDIA_TYPE
    seed_media_types = frozenset({TARGET_MEDIA_TYPE, "text/plain"})
    max_bytes = MAX_DOCUMENT_UTF8_BYTES
    reload_tool = "load_likec4"
    read_tool = "read_likec4"
    seed_revision_message = "a LikeC4 seed outside the Worker namespace requires an exact revision"
    seed_too_large_message = "LikeC4 document exceeds the 1 MiB tool limit"
    seed_media_type_message = f"LikeC4 seed media type must be {TARGET_MEDIA_TYPE} or text/plain"
    not_loaded_message = "load_likec4 or write_likec4 must be called first"
    not_loaded_code = "document_not_loaded"

    def __init__(
        self,
        client: ArtifactClient,
        namespace: str,
        workspace: Path,
        launcher: ProxySubprocessLauncher | None,
    ) -> None:
        super().__init__(client, namespace)
        self._workspace = workspace
        self._launcher = launcher

    def _decode_seed(self, data: bytes) -> str:
        return _decode_document(data)

    def _serialize(self, document: str) -> bytes:
        return _validate_document(document)

    def _document_state(
        self, artifact: ArtifactRef, target_name: str, document: str, *, changed: bool, copied: bool
    ) -> dict[str, Any]:
        return _document_state(artifact, target_name, document, changed=changed, copied=copied)

    async def write(
        self,
        *,
        content: str,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        validate_target_name(target_name)
        _validate_document(content)
        async with self._lock:
            effective_revision = expected_revision
            if (
                effective_revision is None
                and self._target_name == target_name
                and self._revision is not None
            ):
                effective_revision = self._revision
            written = await self._write(
                content,
                target_name=target_name,
                expected_revision=effective_revision,
            )
            self._remember(content, target_name, written.artifact)
            return _document_state(
                written.artifact,
                target_name,
                content,
                changed=True,
                copied=False,
            )

    async def read(self, *, start_line: int, max_lines: int) -> dict[str, Any]:
        _validate_line_window(start_line, max_lines)
        async with self._lock:
            content, artifact = self._require_document()
            lines = split_lines(content, keepends=True)
            total_lines = len(lines)
            if total_lines > 0 and start_line > total_lines:
                raise ToolInputError("start_line exceeds LikeC4 document line count")
            visible, end_line, partial_line = bounded_line_window(
                lines,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=MAX_VISIBLE_UTF8_BYTES,
            )
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "mediaType": TARGET_MEDIA_TYPE,
                "utf8Size": len(content.encode("utf-8")),
                "totalLines": total_lines,
                "startLine": start_line,
                "endLine": end_line,
                "text": visible,
                "truncated": partial_line or end_line < total_lines,
                "partialLine": partial_line,
            }

    async def append(self, content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content:
            raise ToolInputError("append content must be a non-empty string")
        async with self._lock:
            current, _ = self._require_document()
            candidate = current + content
            result = await self._commit_current(candidate)
            result["appendedUtf8Bytes"] = len(content.encode("utf-8"))
            return result

    async def replace(
        self,
        *,
        old: str,
        new: str,
        count: int | None,
    ) -> dict[str, Any]:
        if not isinstance(old, str) or not old:
            raise ToolInputError("old fragment must be a non-empty string")
        if not isinstance(new, str):
            raise ToolInputError("new fragment must be a string")
        if old == new:
            raise ToolInputError("old and new fragments must differ")
        if count is not None and (
            type(count) is not int or not 1 <= count <= MAX_REPLACE_OCCURRENCES
        ):
            raise ToolInputError("count must be an integer from 1 through 100")
        async with self._lock:
            current, _ = self._require_document()
            occurrences = current.count(old)
            if occurrences == 0:
                raise ToolInputError(
                    "old fragment is absent from the LikeC4 document", code="fragment_not_found"
                )
            if count is None:
                if occurrences != 1:
                    raise ToolInputError(
                        "old fragment is ambiguous; pass an explicit replacement count"
                    )
                replacement_count = 1
            else:
                if count > occurrences:
                    raise ToolInputError("replacement count exceeds matching occurrences")
                replacement_count = count
            candidate = current.replace(old, new, replacement_count)
            result = await self._commit_current(candidate)
            result.update(
                {
                    "replacementCount": replacement_count,
                    "matchingOccurrences": occurrences,
                }
            )
            return result

    async def validate(self) -> dict[str, Any]:
        async with self._lock:
            content, artifact = self._require_document()
            validation = await self._validating(
                _run_likec4(content, self._workspace, self._launcher)
            )
            return {
                "artifact": artifact.model_dump(by_alias=True),
                "valid": validation["valid"],
                "validator": "likec4",
                "validatorAvailable": validation["available"],
                "validatorExecutionError": validation["executionError"],
                "issues": validation["issues"],
                "issuesTruncated": validation["truncated"],
                "stats": validation["stats"],
            }

    async def _commit_current(self, candidate: str) -> dict[str, Any]:
        _validate_document(candidate)
        target_name, revision = self._require_target()
        written = await self._write(candidate, target_name=target_name, expected_revision=revision)
        self._remember(candidate, target_name, written.artifact)
        return _document_state(
            written.artifact,
            target_name,
            candidate,
            changed=True,
            copied=False,
        )


_BaseLikeC4Tool = SessionArtifactTool[_LikeC4Session]


def _artifact_metric(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "artifact": result.get("artifact"),
        "changed": result.get("changed"),
        "utf8Size": result.get("utf8Size"),
        "replacementCount": result.get("replacementCount"),
        "appendedUtf8Bytes": result.get("appendedUtf8Bytes"),
    }


class LoadLikeC4Tool(_BaseLikeC4Tool):
    name = "load_likec4"
    description = """Load a LikeC4 artifact into the Worker's current document session.

    Copies the source into the Worker's namespace or resumes the same binding.
    Call this or write_likec4 before reading, editing or validating the document.

    Args:
        namespace: Source artifact namespace.
        name: Source artifact binding name.
        revision: Source revision; required for a source outside the Worker's namespace.
        target_name: Destination binding in the Worker's namespace; default "architecture".
        expected_revision: Current destination revision when copying over an existing
            binding; omit to create it. Resuming the same binding uses its revision.

    Returns:
        Exact current artifact reference, document size and changed/copied flags.
    """

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        target_name: str = DEFAULT_TARGET_NAME,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "target_name": target_name,
            "expected_revision": expected_revision,
        }
        return await self._call(
            arguments,
            self._session.load(
                namespace=namespace,
                name=name,
                revision=revision,
                target_name=target_name,
                expected_revision=expected_revision,
            ),
            _artifact_metric,
        )


class WriteLikeC4Tool(_BaseLikeC4Tool):
    name = "write_likec4"
    description = """Create or replace the current UTF-8 LikeC4 document.

    Writes in the Worker's namespace with revision checks. Use validate_likec4
    after editing to check the document syntax.

    Args:
        content: Complete LikeC4 source text.
        target_name: Destination binding; defaults to "architecture".
        expected_revision: Current destination revision. Omit to reuse the loaded
            revision of this target, or to create a new binding when none is loaded.

    Returns:
        Exact saved artifact reference, document size and changed/copied flags.
    """

    async def __call__(
        self,
        content: str,
        target_name: str = DEFAULT_TARGET_NAME,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "content": content,
            "target_name": target_name,
            "expected_revision": expected_revision,
        }
        return await self._call(
            arguments,
            self._session.write(
                content=content,
                target_name=target_name,
                expected_revision=expected_revision,
            ),
            _artifact_metric,
        )


class ReadLikeC4Tool(_BaseLikeC4Tool):
    name = "read_likec4"
    description = """Read a line window from the current LikeC4 document.

    Load or write the document first. Output is limited to 128 KiB. Continue at
    endLine + 1; partialLine means one line alone exceeds the limit and only its
    prefix is returned.

    Args:
        start_line: First line to read, 1-based and inclusive; defaults to 1.
        max_lines: Maximum lines to return, from 1 to 400; defaults to 200.

    Returns:
        Exact artifact metadata, text, startLine, endLine, totalLines, truncated
        and partialLine.
    """

    async def __call__(
        self,
        start_line: int = 1,
        max_lines: int = DEFAULT_READ_LINES,
    ) -> dict[str, Any]:
        return await self._call(
            {"start_line": start_line, "max_lines": max_lines},
            self._session.read(start_line=start_line, max_lines=max_lines),
            lambda result: {
                "artifact": result["artifact"],
                "utf8Size": result["utf8Size"],
                "totalLines": result["totalLines"],
                "startLine": result["startLine"],
                "endLine": result["endLine"],
                "visibleUtf8Bytes": len(result["text"].encode("utf-8")),
                "truncated": result["truncated"],
            },
        )


class AppendLikeC4Tool(_BaseLikeC4Tool):
    name = "append_likec4"
    description = """Append exact text to the current LikeC4 document and save its revision.

    Load or write the document first. Saving checks the current revision.

    Args:
        content: Non-empty text to append; include any required separating newline.

    Returns:
        Exact saved artifact metadata, changed and appendedUtf8Bytes.
    """

    async def __call__(self, content: str) -> dict[str, Any]:
        return await self._call(
            {"content": content},
            self._session.append(content),
            _artifact_metric,
        )


class ReplaceLikeC4Tool(_BaseLikeC4Tool):
    name = "replace_likec4"
    description = """Replace an exact fragment in the current LikeC4 document and save its revision.

    Read the document first and copy the exact fragment, including whitespace.
    Saving checks the current revision.

    Args:
        old: Non-empty literal fragment to find.
        new: Replacement text; empty deletes the matched fragment.
        count: Number of occurrences to replace from the start, from 1 to 100.
            Omit to require exactly one match; cannot exceed matching occurrences.

    Returns:
        Exact saved artifact metadata, changed and replacementCount.
    """

    async def __call__(
        self,
        old: str,
        new: str,
        count: int | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            {"content": {"old": old, "new": new}, "count": count},
            self._session.replace(old=old, new=new, count=count),
            _artifact_metric,
        )


class ValidateLikeC4Tool(_BaseLikeC4Tool):
    name = "validate_likec4"
    description = """Validate the current LikeC4 document with the LikeC4 CLI.

    Load or write the document first. Validation checks the current session's
    exact artifact revision and does not modify it.

    Returns:
        Exact artifact reference, valid, bounded issues, issuesTruncated,
        validatorAvailable and validatorExecutionError. Unavailable or failed
        validation is not a successful check.
    """

    async def __call__(self) -> dict[str, Any]:
        return await self._call(
            {},
            self._session.validate(),
            lambda result: {
                "artifact": result["artifact"],
                "valid": result["valid"],
                "validatorAvailable": result["validatorAvailable"],
                "validatorExecutionError": result["validatorExecutionError"] is not None,
                "issues": len(result["issues"]),
                "issuesTruncated": result["issuesTruncated"],
            },
        )


async def _run_likec4(
    content: str,
    workspace: Path,
    launcher: ProxySubprocessLauncher | None = None,
) -> dict[str, Any]:
    executable = shutil.which("likec4")
    if executable is None:
        return _validation_failure(False, "LikeC4 executable is unavailable")
    temporary: tempfile.TemporaryDirectory | None = None

    def prepare() -> None:
        nonlocal temporary
        temporary = tempfile.TemporaryDirectory(prefix=".likec4-validate-", dir=workspace)
        (Path(temporary.name) / VALIDATOR_FILENAME).write_text(content, encoding="utf-8")

    try:
        try:
            await to_thread_until_done(prepare, name="likec4-filesystem")
            assert temporary is not None
            project = Path(temporary.name)
            source = project / VALIDATOR_FILENAME
            command = [
                executable,
                "validate",
                "--json",
                "--no-layout",
                "--file",
                str(source),
                str(project),
            ]
            environment = {
                "PATH": os.environ.get("PATH", os.defpath),
                "LANG": "C.UTF-8",
                "CI": "1",
                "NO_COLOR": "1",
                "NO_UPDATE_NOTIFIER": "1",
            }
            process = await run_tool_command(
                command,
                launcher=launcher,
                cwd=project,
                env=environment,
                timeout=VALIDATE_TIMEOUT_SECONDS,
                max_output_bytes=2 * MAX_VALIDATOR_OUTPUT_BYTES,
            )
        finally:
            if temporary is not None:
                await to_thread_until_done(temporary.cleanup, name="likec4-filesystem")
    except subprocess.TimeoutExpired:
        return _validation_failure(True, "LikeC4 validation timed out")
    except ProcessOutputLimitError:
        return _validation_failure(True, "LikeC4 returned oversized output")
    except OSError:
        return _validation_failure(True, "LikeC4 could not be executed")

    if process.returncode not in {0, 1}:
        return _validation_failure(True, "LikeC4 returned an execution failure")
    if not process.stdout or len(process.stdout) > MAX_VALIDATOR_OUTPUT_BYTES:
        return _validation_failure(True, "LikeC4 returned empty or oversized output")
    try:
        text = process.stdout.decode("utf-8", errors="strict")
        parsed = _extract_json(text)
    except (UnicodeDecodeError, ValueError):
        return _validation_failure(True, "LikeC4 returned invalid JSON")

    if not isinstance(parsed, dict) or not isinstance(parsed.get("errors"), list):
        return _validation_failure(True, "LikeC4 returned an unexpected JSON shape")
    errors = parsed["errors"]
    reported_valid: bool | None = None
    if "valid" in parsed:
        if not isinstance(parsed["valid"], bool):
            return _validation_failure(True, "LikeC4 returned an unexpected JSON shape")
        reported_valid = parsed["valid"]
    stats = _sanitize_stats(parsed.get("stats"))

    try:
        issues, truncated = _sanitize_diagnostics(errors)
    except ValueError:
        return _validation_failure(True, "LikeC4 returned invalid diagnostics")
    valid = not issues and not truncated
    if reported_valid is not None:
        valid = valid and reported_valid
    return {
        "available": True,
        "executionError": None,
        "valid": valid,
        "issues": issues,
        "truncated": truncated,
        "stats": stats,
    }


def _extract_json(text: str) -> Any:
    """Decode CLI output that is JSON, possibly after a short text banner.

    Each fallback start may scan the rest of the output, so only a bounded
    number of ``[``/``{`` positions are tried. Nesting deeper than the decoder
    supports is invalid output, not a Runtime failure.
    """

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    except RecursionError:
        raise ValueError("LikeC4 output JSON is nested too deeply") from None
    decoder = json.JSONDecoder()
    attempts = 0
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        if attempts >= MAX_JSON_FALLBACK_STARTS:
            break
        attempts += 1
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        except RecursionError:
            raise ValueError("LikeC4 output JSON is nested too deeply") from None
        if not text[end:].strip():
            return value
    raise ValueError("LikeC4 output contains no complete JSON value")


def _sanitize_diagnostics(values: list[Any]) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    truncated = len(values) > MAX_VALIDATION_DIAGNOSTICS
    for value in values[:MAX_VALIDATION_DIAGNOSTICS]:
        if not isinstance(value, dict):
            raise ValueError("LikeC4 diagnostic must be an object")
        sanitized = _sanitize_cli_value(value)
        if not isinstance(sanitized, dict):
            raise ValueError("LikeC4 diagnostic must remain an object")
        if "file" in sanitized:
            sanitized["file"] = VALIDATOR_FILENAME
        result.append(sanitized)
    return result, truncated


def _sanitize_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in list(value.items())[:MAX_DIAGNOSTIC_ITEMS]:
        if isinstance(key, str) and isinstance(item, bool | int | float | str | type(None)):
            result[_bounded_cli_text(key)] = (
                _bounded_cli_text(item) if isinstance(item, str) else item
            )
    return result


def _sanitize_cli_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_DIAGNOSTIC_DEPTH:
        return "[TRUNCATED]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _bounded_cli_text(value)
    if isinstance(value, list):
        return [_sanitize_cli_value(item, depth=depth + 1) for item in value[:MAX_DIAGNOSTIC_ITEMS]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_DIAGNOSTIC_ITEMS]:
            if not isinstance(key, str):
                raise ValueError("LikeC4 diagnostic keys must be strings")
            result[_bounded_cli_text(key)] = _sanitize_cli_value(item, depth=depth + 1)
        return result
    raise ValueError("LikeC4 diagnostic contains a non-JSON value")


def _bounded_cli_text(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_DIAGNOSTIC_TEXT_BYTES:
        return value
    # Parser messages can list hundreds of expected alternatives before the
    # actual offending token. Keep that actionable suffix as well as the title.
    marker = "\n[TRUNCATED]\n"
    available = MAX_DIAGNOSTIC_TEXT_BYTES - len(marker.encode("utf-8"))
    head_bytes = available // 2
    tail_bytes = available - head_bytes
    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore")
    return head + marker + tail


def _validation_failure(available: bool, message: str) -> dict[str, Any]:
    return {
        "available": available,
        "executionError": message,
        "valid": False,
        "issues": [],
        "truncated": False,
        "stats": {},
    }


def _decode_document(data: bytes) -> str:
    if len(data) > MAX_DOCUMENT_UTF8_BYTES:
        raise ToolInputError("LikeC4 document exceeds the 1 MiB tool limit")
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ToolInputError("LikeC4 document must be valid UTF-8") from error


def _validate_document(content: Any) -> bytes:
    if not isinstance(content, str):
        raise ToolInputError("LikeC4 content must be a string")
    try:
        data = content.encode("utf-8")
    except UnicodeError:
        raise ToolInputError("LikeC4 content must be valid UTF-8") from None
    if len(data) > MAX_DOCUMENT_UTF8_BYTES:
        raise ToolInputError("LikeC4 document exceeds the 1 MiB tool limit")
    return data


def _validate_line_window(start_line: int, max_lines: int) -> None:
    if type(start_line) is not int or start_line < 1:
        raise ToolInputError("start_line must be a positive integer")
    if type(max_lines) is not int or not 1 <= max_lines <= MAX_READ_LINES:
        raise ToolInputError("max_lines must be an integer from 1 through 400")


def _document_state(
    artifact: ArtifactRef,
    target_name: str,
    content: str,
    *,
    changed: bool,
    copied: bool,
) -> dict[str, Any]:
    return {
        "artifact": artifact.require_exact().model_dump(by_alias=True),
        "mediaType": TARGET_MEDIA_TYPE,
        "targetName": target_name,
        "utf8Size": len(content.encode("utf-8")),
        "changed": changed,
        "copied": copied,
    }
