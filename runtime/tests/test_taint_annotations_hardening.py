from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs
from test_taint_annotations import MemoryWriter, make_tools

import contractor_runtime.toolsets.taint_annotations.tools as taint_annotations
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs import MemoryWorkspaceProvider, hydrate_workspace
from contractor_runtime.projectfs.storage import WorkspaceStorageError
from contractor_runtime.toolsets.edit_files.tools import EditFilesToolsetFactory
from contractor_runtime.toolsets.taint_annotations.tools import (
    MAX_ANNOTATION_BYTES,
    MAX_DEFINITION_LINE,
    MAX_LIST_ENTRIES,
    MAX_SOURCE_FILE_BYTES,
    MAX_TOKEN_CHARS,
    TaintAnnotationError,
)
from contractor_runtime.workspace import AllocationWorkspace

SOURCE_CANARY = "taint-source-retention-canary"
PATH_CANARY = "private-taint-path-canary.py"
TARGET_CANARY = "taint-target-value-canary"
CREDENTIAL_CANARY = "taint-runtime-credential-canary"


def test_raw_argument_shape_is_rejected_before_binding(tmp_path: Path) -> None:
    async def scenario() -> None:
        tools, metrics = await make_tools(
            tmp_path, MemoryWriter({"app.py": "def handler(): pass\n"})
        )
        cases = (
            ("annotate_trace", {}),
            (
                "annotate_trace",
                {"path": "app.py", "symbol": "handler", "unknown": "value"},
            ),
            (
                "annotate_validate",
                {"path": "app.py", "symbol": "handler", "arg": "value"},
            ),
            (
                "annotate_sink",
                {
                    "path": "app.py",
                    "symbol": "handler",
                    "kind": "db.query",
                    "definition_line": True,
                },
            ),
        )
        for name, raw in cases:
            failure = tools[name].contractor_raw_argument_error(raw)
            assert failure is not None
            assert failure.code == "taint_annotation_input_invalid"
        assert metrics.counters["tool_errors"] == len(cases)

    asyncio.run(scenario())


def test_real_workspace_cross_toolset_race_preserves_the_edit_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [
                (
                    "source",
                    "",
                    archive(
                        {
                            "app.py": (
                                f"def handler(req):\n    return '{SOURCE_CANARY}' + req\n"
                            ).encode()
                        }
                    ),
                )
            ]
        )
        spec.mode = "overlay"  # type: ignore[assignment]
        provider = MemoryWorkspaceProvider(settings("memory"))
        workspace = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="taint-race",
            timeout_seconds=5,
        )
        original_parse = taint_annotations._parse_target_file
        entered = threading.Event()
        resume = threading.Event()

        def delayed_parse(source: bytes, language: Any) -> Any:
            entered.set()
            assert resume.wait(timeout=5)
            return original_parse(source, language)

        monkeypatch.setattr(taint_annotations, "_parse_target_file", delayed_parse)
        annotation_tools, _ = await make_tools(tmp_path, workspace.writer_view())
        edit_tools = await EditFilesToolsetFactory().create_selected(
            selected=("append_file",),
            allocation_id="taint-race",
            run_id="run-race",
            namespace="analysis",
            runtime_settings=runtime_settings(),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
            state=WorkerState(),
            project_workspace=workspace.writer_view(),
        )
        annotation = asyncio.create_task(annotation_tools["annotate_trace"]("app.py", "handler"))
        assert await asyncio.to_thread(entered.wait, 5)
        await edit_tools["append_file"]("app.py", "# competing-writer-won")
        resume.set()
        with pytest.raises(TaintAnnotationError) as changed:
            await annotation
        assert changed.value.code == "taint_annotation_workspace_changed"
        text = await workspace.read_text("app.py")
        assert "# competing-writer-won" in text
        assert "@trace" not in text
        await annotation_tools["annotate_trace"].close()
        await edit_tools["append_file"].close()
        await workspace.close()
        await provider.cleanup(workspace.storage)

    asyncio.run(scenario())


def test_cancelled_parse_is_joined_and_close_prevents_queued_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        writer = MemoryWriter({"app.py": "def handler(req):\n    return req\n"})
        tools, metrics = await make_tools(tmp_path, writer)
        original_parse = taint_annotations._parse_target_file
        entered = threading.Event()
        resume = threading.Event()

        def delayed_parse(source: bytes, language: Any) -> Any:
            entered.set()
            assert resume.wait(timeout=5)
            return original_parse(source, language)

        monkeypatch.setattr(taint_annotations, "_parse_target_file", delayed_parse)
        active = asyncio.create_task(tools["annotate_trace"]("app.py", "handler"))
        assert await asyncio.to_thread(entered.wait, 5)
        active.cancel()
        await asyncio.sleep(0)
        closing = asyncio.create_task(tools["annotate_trace"].close())
        queued = asyncio.create_task(tools["annotate_sink"]("app.py", "handler", "db.query"))
        await asyncio.sleep(0.01)
        assert not active.done()
        assert not closing.done()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        await closing
        with pytest.raises(TaintAnnotationError) as stopped:
            await queued
        assert stopped.value.code == "taint_annotation_closing"
        await tools["annotate_sink"].close()
        await tools["annotate_sink"].close()
        assert await writer.read_text("app.py") == "def handler(req):\n    return req\n"
        rendered = json.dumps(metrics.snapshot(), sort_keys=True)
        assert "taint_annotation_cancelled" in rendered
        assert SOURCE_CANARY not in rendered

    asyncio.run(scenario())


def test_raw_argument_validation_and_unexpected_failures_are_closed_and_redacted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        writer = ExplodingWriter()
        tools, metrics = await make_tools(tmp_path, writer)
        trace = tools["annotate_trace"]
        for raw in (
            {},
            {"path": PATH_CANARY, "symbol": "handler", "unknown": SOURCE_CANARY},
            {"path": PATH_CANARY, "symbol": "handler", "definition_line": True},
            {"path": 7, "symbol": "handler"},
        ):
            failure = trace.contractor_raw_argument_error(raw)
            assert failure is not None
            assert failure.code == "taint_annotation_input_invalid"

        with pytest.raises(TaintAnnotationError) as read_failure:
            await trace(PATH_CANARY, "handler", target=TARGET_CANARY)
        assert read_failure.value.code == "taint_annotation_unavailable"
        assert read_failure.value.retryable

        update_writer = UpdateExplodingWriter(
            {PATH_CANARY: f"def handler():\n    return '{SOURCE_CANARY}'\n"}
        )
        update_tools, update_metrics = await make_tools(tmp_path, update_writer)
        with pytest.raises(TaintAnnotationError) as update_failure:
            await update_tools["annotate_sink"](PATH_CANARY, "handler", TARGET_CANARY)
        assert update_failure.value.code == "taint_annotation_unavailable"
        assert update_failure.value.retryable

        parser_writer = MemoryWriter(
            {PATH_CANARY: f"def handler():\n    return '{SOURCE_CANARY}'\n"}
        )
        parser_tools, parser_metrics = await make_tools(tmp_path, parser_writer)

        def broken_parser(*_: object) -> object:
            raise RuntimeError(f"{SOURCE_CANARY} /physical/{PATH_CANARY}")

        monkeypatch.setattr(taint_annotations, "_parse_target_file", broken_parser)
        with pytest.raises(TaintAnnotationError) as parser_failure:
            await parser_tools["annotate_trace"](PATH_CANARY, "handler", target=TARGET_CANARY)
        assert parser_failure.value.code == "taint_annotation_unavailable"
        assert parser_failure.value.retryable

        retained = json.dumps(
            {
                "snapshots": [
                    metrics.snapshot(),
                    update_metrics.snapshot(),
                    parser_metrics.snapshot(),
                ],
                "reports": [
                    state.build_report(report_id=f"taint-report-{index}", duration_ms=1).model_dump(
                        mode="json", by_alias=True
                    )
                    for index, state in enumerate(
                        (metrics, update_metrics, parser_metrics), start=1
                    )
                ],
                "errors": [
                    str(read_failure.value),
                    str(update_failure.value),
                    str(parser_failure.value),
                ],
            },
            sort_keys=True,
        )
        sessions = [
            trace._session,
            update_tools["annotate_sink"]._session,
            parser_tools["annotate_trace"]._session,
        ]
        await trace.close()
        await update_tools["annotate_sink"].close()
        await parser_tools["annotate_trace"].close()
        retained += "".join(repr(session.__dict__) for session in sessions)
        for canary in (SOURCE_CANARY, PATH_CANARY, TARGET_CANARY, CREDENTIAL_CANARY):
            assert canary not in retained

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    rendered_logs = "\n".join(record.getMessage() for record in caplog.records)
    for canary in (SOURCE_CANARY, PATH_CANARY, TARGET_CANARY, CREDENTIAL_CANARY):
        assert canary not in rendered_logs


def test_manual_malformed_annotation_blocks_conflict_without_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        fixtures = {
            "trace.py": "# @trace malformed\ndef handler(): pass\n",
            "validate.py": "# @validate malformed\ndef handler(): pass\n",
            "sink.py": "# @sink malformed\ndef handler(): pass\n",
        }
        writer = MemoryWriter(fixtures)
        tools, _ = await make_tools(tmp_path, writer)
        calls = (
            ("trace.py", lambda: tools["annotate_trace"]("trace.py", "handler")),
            (
                "validate.py",
                lambda: tools["annotate_validate"]("validate.py", "handler", "value", "schema"),
            ),
            (
                "sink.py",
                lambda: tools["annotate_sink"]("sink.py", "handler", "db.query", "value"),
            ),
        )
        for path, invoke in calls:
            with pytest.raises(TaintAnnotationError) as conflict:
                await invoke()
            assert conflict.value.code == "taint_annotation_conflict"
            assert await writer.read_text(path) == fixtures[path]

    asyncio.run(scenario())


def test_exact_scalar_list_line_and_resulting_file_limits(tmp_path: Path) -> None:
    async def bounded_scenario() -> None:
        writer = MemoryWriter({"app.py": "def handler(): pass\n"})
        tools, _ = await make_tools(tmp_path, writer)
        token = "t" * MAX_TOKEN_CHARS
        result = await tools["annotate_trace"](
            "app.py",
            "handler",
            target=token,
            calls=",".join(f"call{index}" for index in range(MAX_LIST_ENTRIES)),
        )
        assert result["changed"]
        failures = (
            (token + "x", "taint_annotation_input_invalid"),
            (
                ",".join(f"call{index}" for index in range(MAX_LIST_ENTRIES + 1)),
                "taint_annotation_capacity_exceeded",
            ),
        )
        for calls, code in failures:
            with pytest.raises(TaintAnnotationError) as failure:
                await tools["annotate_trace"]("app.py", "handler", target="other", calls=calls)
            assert failure.value.code == code
        with pytest.raises(TaintAnnotationError) as line:
            await tools["annotate_trace"](
                "app.py", "handler", definition_line=MAX_DEFINITION_LINE + 1
            )
        assert line.value.code == "taint_annotation_input_invalid"
        with pytest.raises(TaintAnnotationError) as accepted_line_bound:
            await tools["annotate_trace"]("app.py", "handler", definition_line=MAX_DEFINITION_LINE)
        assert accepted_line_bound.value.code == "taint_annotation_target_not_found"

        wide_calls = ",".join(
            f"c{index}" + "x" * (MAX_TOKEN_CHARS - len(f"c{index}"))
            for index in range(MAX_LIST_ENTRIES)
        )
        with pytest.raises(TaintAnnotationError) as annotation:
            await tools["annotate_trace"]("app.py", "handler", target="wide", calls=wide_calls)
        assert annotation.value.code == "taint_annotation_capacity_exceeded"
        assert len(wide_calls.encode()) > MAX_ANNOTATION_BYTES

        line = "# @trace target=fit\n"
        definition = "def handler(): pass\n"
        filler_size = MAX_SOURCE_FILE_BYTES - len(line.encode()) - len(definition.encode())
        fitting_source = "#" + "x" * (filler_size - 2) + "\n" + definition
        fitting = MemoryWriter({"fit.py": fitting_source})
        fitting_tools, _ = await make_tools(tmp_path, fitting)
        await fitting_tools["annotate_trace"]("fit.py", "handler", target="fit")
        assert len((await fitting.read_text("fit.py")).encode()) == MAX_SOURCE_FILE_BYTES

        too_large = MemoryWriter({"over.py": fitting_source + "x"})
        too_large_tools, _ = await make_tools(tmp_path, too_large)
        with pytest.raises(TaintAnnotationError) as capacity:
            await too_large_tools["annotate_trace"]("over.py", "handler", target="fit")
        assert capacity.value.code == "taint_annotation_capacity_exceeded"

    asyncio.run(bounded_scenario())


def test_simultaneous_annotations_have_one_stable_serial_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        writer = MemoryWriter({"app.py": "def handler(req):\n    return req\n"})
        tools, _ = await make_tools(tmp_path, writer)
        ready = asyncio.Event()

        async def insert(target: str) -> dict[str, Any]:
            await ready.wait()
            return await tools["annotate_trace"]("app.py", "handler", target=target)

        first = asyncio.create_task(insert("first"))
        second = asyncio.create_task(insert("second"))
        ready.set()
        results = await asyncio.gather(first, second)
        assert all(result["changed"] for result in results)
        assert await writer.read_text("app.py") == (
            "# @trace target=first\n# @trace target=second\ndef handler(req):\n    return req\n"
        )

    asyncio.run(scenario())


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken=CREDENTIAL_CANARY,
        artifactApiUrl="https://server.example/private/v1",
        requestTimeoutSeconds=30,
    )


class ExplodingWriter:
    async def read_text(self, path: str) -> str:
        raise WorkspaceStorageError(f"unexpected {SOURCE_CANARY} {path} {CREDENTIAL_CANARY}")

    async def update_text(self, path: str, transform: Any) -> None:
        raise AssertionError((path, transform))


class UpdateExplodingWriter(MemoryWriter):
    async def update_text(self, path: str, transform: Any) -> None:
        del transform
        raise WorkspaceStorageError(
            f"unexpected {SOURCE_CANARY} {path} {TARGET_CANARY} {CREDENTIAL_CANARY}"
        )
