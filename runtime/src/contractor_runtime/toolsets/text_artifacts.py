"""Bounded UTF-8 tools over one allocation-bound RunScope ArtifactClient."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_TEXT_WRITE_BYTES = 1024 * 1024
MAX_TEXT_READ_LINES = 400
DEFAULT_TEXT_READ_LINES = 200
MAX_VISIBLE_TEXT_BYTES = 128 * 1024


class TextArtifactsToolsetFactory:
    """Construct only the explicitly selected ``text-artifacts@1`` tools."""

    ref = "text-artifacts@1"
    exported_tools = frozenset({"read_text_artifact", "write_text_artifact"})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

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
        del run_id, workspace
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("text-artifacts@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = (runtime_settings.llm_gateway_token.get_secret_value(),)
        builders: dict[str, Callable[[], Any]] = {
            "read_text_artifact": lambda: ReadTextArtifactTool(client, metrics, secrets),
            "write_text_artifact": lambda: WriteTextArtifactTool(
                client, metrics, secrets, namespace
            ),
        }
        return {name: builders[name]() for name in selected}


class _BaseTextTool:
    name: str
    description: str

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(getattr(self._client, "known_exact_refs", ()))

    async def close(self) -> None:
        self._secrets = ()

    def _success(
        self, arguments: Mapping[str, Any], result: Mapping[str, Any], started_ns: int
    ) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            result=result,
            secrets=self._secrets,
            duration_ms=_elapsed_ms(started_ns),
        )

    def _failure(self, arguments: Mapping[str, Any], error: Exception, started_ns: int) -> None:
        self._metrics.record_tool_call(
            self.name,
            arguments=arguments,
            error=error,
            secrets=self._secrets,
            duration_ms=_elapsed_ms(started_ns),
        )


class ReadTextArtifactTool(_BaseTextTool):
    name = "read_text_artifact"
    description = (
        "Read a bounded line window from a UTF-8 artifact in this Workflow Run. "
        "Use the exact revision supplied in Stage context when one is available."
    )

    async def __call__(
        self,
        namespace: str,
        name: str,
        revision: str | None = None,
        start_line: int = 1,
        max_lines: int = DEFAULT_TEXT_READ_LINES,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        arguments = {
            "namespace": namespace,
            "name": name,
            "revision": revision,
            "start_line": start_line,
            "max_lines": max_lines,
        }
        try:
            _validate_line_window(start_line, max_lines)
            value = await self._client.read_artifact(
                ArtifactRef(namespace=namespace, name=name, revision=revision)
            )
            try:
                content = value.data.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError("artifact is not valid UTF-8") from error
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            if total_lines > 0 and start_line > total_lines:
                raise ValueError("start_line exceeds artifact line count")
            selected, end_line, partial_line = _bounded_lines(
                lines, start_line=start_line, max_lines=max_lines
            )
            truncated = partial_line or end_line < total_lines
            result = {
                "artifact": value.artifact.model_dump(by_alias=True),
                "mediaType": value.media_type,
                "size": len(value.data),
                "totalLines": total_lines,
                "startLine": start_line,
                "endLine": end_line,
                "text": selected,
                "truncated": truncated,
                "partialLine": partial_line,
            }
            self._success(
                arguments,
                {
                    "artifact": result["artifact"],
                    "mediaType": value.media_type,
                    "size": len(value.data),
                    "totalLines": total_lines,
                    "startLine": start_line,
                    "endLine": end_line,
                    "visibleUtf8Bytes": len(selected.encode("utf-8")),
                    "truncated": truncated,
                    "partialLine": partial_line,
                },
                started_ns,
            )
            return result
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


class WriteTextArtifactTool(_BaseTextTool):
    name = "write_text_artifact"
    description = (
        "Create or CAS-update a UTF-8 artifact in this Worker's fixed Namespace. "
        "Omit expected_revision only when creating a new binding."
    )

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        namespace: str,
    ) -> None:
        super().__init__(client, metrics, secrets)
        self._namespace = namespace

    async def __call__(
        self,
        name: str,
        text: str,
        media_type: str,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        if not isinstance(text, str):
            error = TypeError("text must be a string")
            self._failure(
                {
                    "name": name,
                    "media_type": media_type,
                    "expected_revision": expected_revision,
                    "content": text,
                },
                error,
                started_ns,
            )
            raise error
        data = text.encode("utf-8")
        arguments = {
            "name": name,
            "media_type": media_type,
            "expected_revision": expected_revision,
            "content": text,
            "utf8_size": len(data),
        }
        try:
            if len(data) > MAX_TEXT_WRITE_BYTES:
                raise ValueError("UTF-8 artifact text exceeds the 1 MiB tool limit")
            result = await self._client.write_artifact(
                ArtifactRef(namespace=self._namespace, name=name),
                data=data,
                media_type=media_type,
                expected_revision=expected_revision,
            )
            serialized = result.model_dump(by_alias=True)
            self._success(
                arguments,
                {
                    "artifact": serialized["artifact"],
                    "mediaType": result.media_type,
                    "size": result.size,
                },
                started_ns,
            )
            return serialized
        except Exception as error:
            self._failure(arguments, error, started_ns)
            raise


def _validate_line_window(start_line: int, max_lines: int) -> None:
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError("start_line must be a positive integer")
    if (
        isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or max_lines < 1
        or max_lines > MAX_TEXT_READ_LINES
    ):
        raise ValueError(f"max_lines must be between 1 and {MAX_TEXT_READ_LINES}")


def _bounded_lines(lines: list[str], *, start_line: int, max_lines: int) -> tuple[str, int, bool]:
    if not lines:
        return "", 0, False
    window = lines[start_line - 1 : start_line - 1 + max_lines]
    result: list[str] = []
    size = 0
    partial_line = False
    for line in window:
        encoded = line.encode("utf-8")
        remaining = MAX_VISIBLE_TEXT_BYTES - size
        if len(encoded) <= remaining:
            result.append(line)
            size += len(encoded)
            continue
        if not result and remaining > 0:
            result.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            partial_line = True
        break
    complete_lines = len(result) - (1 if partial_line else 0)
    end_line = start_line + complete_lines - 1
    if partial_line:
        end_line = start_line
    return "".join(result), end_line, partial_line


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del allocation_id, runtime_settings
    raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
