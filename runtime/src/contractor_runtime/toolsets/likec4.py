"""Namespace-bound, CAS-backed tools for one single-file LikeC4 document."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.probe import executable_responds
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_DOCUMENT_UTF8_BYTES = 1024 * 1024
MAX_VISIBLE_UTF8_BYTES = 128 * 1024
MAX_READ_LINES = 400
DEFAULT_READ_LINES = 200
MAX_REPLACE_OCCURRENCES = 100
MAX_VALIDATOR_OUTPUT_BYTES = 1024 * 1024
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
    ) -> Mapping[str, Any]:
        del run_id
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("likec4@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        session = _LikeC4Session(client, namespace, workspace.path)
        secrets = (runtime_settings.llm_gateway_token.get_secret_value(),)
        builders: dict[str, Callable[[], Any]] = {
            "load_likec4": lambda: LoadLikeC4Tool(session, client, metrics, secrets),
            "write_likec4": lambda: WriteLikeC4Tool(session, client, metrics, secrets),
            "read_likec4": lambda: ReadLikeC4Tool(session, client, metrics, secrets),
            "append_likec4": lambda: AppendLikeC4Tool(session, client, metrics, secrets),
            "replace_likec4": lambda: ReplaceLikeC4Tool(session, client, metrics, secrets),
            "validate_likec4": lambda: ValidateLikeC4Tool(session, client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


class _LikeC4Session:
    def __init__(self, client: ArtifactClient, namespace: str, workspace: Path) -> None:
        self._client = client
        self._namespace = namespace
        self._workspace = workspace
        self._lock = asyncio.Lock()
        self._content: str | None = None
        self._target_name: str | None = None
        self._revision: str | None = None

    async def load(
        self,
        *,
        namespace: str,
        name: str,
        revision: str | None,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        _validate_target_name(target_name)
        if namespace != self._namespace and revision is None:
            raise ValueError(
                "a LikeC4 seed outside the Worker namespace requires an exact revision"
            )
        source_ref = ArtifactRef(namespace=namespace, name=name, revision=revision)
        async with self._lock:
            value = await self._client.read_artifact(source_ref)
            if revision is not None and value.artifact.revision != revision:
                raise ValueError("Artifact API did not preserve the requested exact revision")
            if value.media_type not in {TARGET_MEDIA_TYPE, "text/plain"}:
                raise ValueError(
                    f"LikeC4 seed media type must be {TARGET_MEDIA_TYPE} or text/plain"
                )
            content = _decode_document(value.data)
            same_binding = namespace == self._namespace and name == target_name
            if same_binding and value.media_type == TARGET_MEDIA_TYPE:
                self._content = content
                self._target_name = target_name
                self._revision = value.artifact.require_exact().revision
                return _document_state(
                    value.artifact,
                    target_name,
                    content,
                    changed=False,
                    copied=False,
                )
            written = await self._write_content(
                content,
                target_name=target_name,
                expected_revision=(value.artifact.revision if same_binding else expected_revision),
            )
            self._content = content
            self._target_name = target_name
            self._revision = written.artifact.require_exact().revision
            return _document_state(
                written.artifact,
                target_name,
                content,
                changed=True,
                copied=True,
            )

    async def write(
        self,
        *,
        content: str,
        target_name: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        _validate_target_name(target_name)
        _validate_document(content)
        async with self._lock:
            effective_revision = expected_revision
            if (
                effective_revision is None
                and self._target_name == target_name
                and self._revision is not None
            ):
                effective_revision = self._revision
            written = await self._write_content(
                content,
                target_name=target_name,
                expected_revision=effective_revision,
            )
            self._content = content
            self._target_name = target_name
            self._revision = written.artifact.require_exact().revision
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
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            if total_lines > 0 and start_line > total_lines:
                raise ValueError("start_line exceeds LikeC4 document line count")
            visible, end_line, partial_line = _bounded_lines(
                lines, start_line=start_line, max_lines=max_lines
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
            raise ValueError("append content must be a non-empty string")
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
            raise ValueError("old fragment must be a non-empty string")
        if not isinstance(new, str):
            raise TypeError("new fragment must be a string")
        if old == new:
            raise ValueError("old and new fragments must differ")
        if count is not None and (
            type(count) is not int or not 1 <= count <= MAX_REPLACE_OCCURRENCES
        ):
            raise ValueError("count must be an integer from 1 through 100")
        async with self._lock:
            current, _ = self._require_document()
            occurrences = current.count(old)
            if occurrences == 0:
                raise ValueError("old fragment is absent from the LikeC4 document")
            if count is None:
                if occurrences != 1:
                    raise ValueError(
                        "old fragment is ambiguous; pass an explicit replacement count"
                    )
                replacement_count = 1
            else:
                if count > occurrences:
                    raise ValueError("replacement count exceeds matching occurrences")
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
            validation = await asyncio.to_thread(
                _run_likec4,
                content,
                self._workspace,
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

    async def close(self) -> None:
        async with self._lock:
            self._content = None
            self._target_name = None
            self._revision = None

    async def _commit_current(self, candidate: str) -> dict[str, Any]:
        _validate_document(candidate)
        assert self._target_name is not None and self._revision is not None
        written = await self._write_content(
            candidate,
            target_name=self._target_name,
            expected_revision=self._revision,
        )
        self._content = candidate
        self._revision = written.artifact.require_exact().revision
        return _document_state(
            written.artifact,
            self._target_name,
            candidate,
            changed=True,
            copied=False,
        )

    async def _write_content(
        self,
        content: str,
        *,
        target_name: str,
        expected_revision: str | None,
    ) -> Any:
        data = _validate_document(content)
        return await self._client.write_artifact(
            ArtifactRef(namespace=self._namespace, name=target_name),
            data=data,
            media_type=TARGET_MEDIA_TYPE,
            expected_revision=expected_revision,
        )

    def _require_document(self) -> tuple[str, ArtifactRef]:
        if self._content is None or self._target_name is None or self._revision is None:
            raise ValueError("load_likec4 or write_likec4 must be called first")
        return self._content, ArtifactRef(
            namespace=self._namespace,
            name=self._target_name,
            revision=self._revision,
        )


class _BaseLikeC4Tool:
    name: str
    description: str

    def __init__(
        self,
        session: _LikeC4Session,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._session = session
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(getattr(self._client, "known_exact_refs", ()))

    async def close(self) -> None:
        await self._session.close()
        self._secrets = ()

    async def _call(
        self,
        arguments: Mapping[str, Any],
        operation: Any,
        metric_result: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await operation
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                result=metric_result(result),
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            raise


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
    description = "Load an exact LikeC4 artifact and copy or resume it in this Agent Namespace."

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
    description = "Create or CAS-replace the bounded UTF-8 LikeC4 document in this Agent Namespace."

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
    description = "Read a bounded line window from the current exact LikeC4 document."

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
    description = "Append exact text to the current document and CAS-save one new revision."

    async def __call__(self, content: str) -> dict[str, Any]:
        return await self._call(
            {"content": content},
            self._session.append(content),
            _artifact_metric,
        )


class ReplaceLikeC4Tool(_BaseLikeC4Tool):
    name = "replace_likec4"
    description = (
        "Replace one exact fragment, or an explicit bounded number of matches, then CAS-save."
    )

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
    description = "Validate the current exact document with a fixed direct LikeC4 CLI invocation."

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


def _run_likec4(content: str, workspace: Path) -> dict[str, Any]:
    executable = shutil.which("likec4")
    if executable is None:
        return _validation_failure(False, "LikeC4 executable is unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix=".likec4-validate-", dir=workspace) as temporary:
            project = Path(temporary)
            source = project / VALIDATOR_FILENAME
            source.write_text(content, encoding="utf-8")
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
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                cwd=project,
                env=environment,
                timeout=VALIDATE_TIMEOUT_SECONDS,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return _validation_failure(True, "LikeC4 validation timed out")
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

    reported_valid: bool | None = None
    stats: dict[str, Any] = {}
    if isinstance(parsed, dict):
        errors = parsed.get("errors")
        if not isinstance(errors, list):
            return _validation_failure(True, "LikeC4 returned an unexpected JSON shape")
        if "valid" in parsed:
            if not isinstance(parsed["valid"], bool):
                return _validation_failure(True, "LikeC4 returned an unexpected JSON shape")
            reported_valid = parsed["valid"]
        stats = _sanitize_stats(parsed.get("stats"))
    elif isinstance(parsed, list):
        errors = parsed
    else:
        return _validation_failure(True, "LikeC4 returned an unexpected JSON shape")

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
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
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
    visible = encoded[:MAX_DIAGNOSTIC_TEXT_BYTES].decode("utf-8", errors="ignore")
    return visible + "[TRUNCATED]"


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
        raise ValueError("LikeC4 document exceeds the 1 MiB tool limit")
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("LikeC4 document must be valid UTF-8") from error


def _validate_document(content: Any) -> bytes:
    if not isinstance(content, str):
        raise TypeError("LikeC4 content must be a string")
    data = content.encode("utf-8")
    if len(data) > MAX_DOCUMENT_UTF8_BYTES:
        raise ValueError("LikeC4 document exceeds the 1 MiB tool limit")
    return data


def _validate_target_name(value: str) -> None:
    ArtifactRef(namespace="likec4", name=value)


def _validate_line_window(start_line: int, max_lines: int) -> None:
    if type(start_line) is not int or start_line < 1:
        raise ValueError("start_line must be a positive integer")
    if type(max_lines) is not int or not 1 <= max_lines <= MAX_READ_LINES:
        raise ValueError("max_lines must be an integer from 1 through 400")


def _bounded_lines(lines: list[str], *, start_line: int, max_lines: int) -> tuple[str, int, bool]:
    selected: list[str] = []
    visible_bytes = 0
    partial_line = False
    end_line = start_line - 1
    for line_number, line in enumerate(
        lines[start_line - 1 : start_line - 1 + max_lines], start=start_line
    ):
        encoded = line.encode("utf-8")
        remaining = MAX_VISIBLE_UTF8_BYTES - visible_bytes
        if len(encoded) <= remaining:
            selected.append(line)
            visible_bytes += len(encoded)
            end_line = line_number
            continue
        if remaining > 0:
            selected.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            end_line = line_number
        partial_line = True
        break
    return "".join(selected), end_line, partial_line


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


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
