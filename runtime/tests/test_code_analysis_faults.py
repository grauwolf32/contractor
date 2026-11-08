from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.clock import run_with_advanceable_clock
from fakes.model import scripted_model, tool_call
from fakes.worker_state import WorkerState
from test_adk_runtime import create_runtime, stage_request, terminal_text
from test_code_analysis_child_lifecycle import (
    CHILD_BUDGET_SECONDS,
    term_marker,
    through_term_grace,
)

import contractor_runtime.toolsets.code_analysis.tools as code_analysis
import contractor_runtime.toolsets.common.source_languages as source_languages
import contractor_runtime.toolsets.common.source_names as source_names
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs.storage import (
    ManagedWorkspaceTree,
    WorkspaceSnapshot,
    WorkspaceTextFile,
)
from contractor_runtime.toolsets.code_analysis.tools import (
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    TrailmarkChildHost,
    TrailmarkHostError,
)
from contractor_runtime.workspace import AllocationWorkspace

FAULT_CHILD = Path(__file__).parent / "fixtures" / "trailmark_fault_child.py"
DEFECT_CANARY = "recognizable-parser-defect-canary"


def test_definitely_unprocessed_read_only_build_gets_exactly_one_safe_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        marker = tmp_path / "first-child.marker"
        host = TrailmarkChildHost(
            tmp_path / "scratch",
            child_command=(
                sys.executable,
                "-I",
                str(FAULT_CHILD),
                "exit-once-before-read",
                str(marker),
            ),
            build_timeout_seconds=CHILD_BUDGET_SECONDS,
            stop_timeout_seconds=CHILD_BUDGET_SECONDS,
        )
        starts = 0
        original_start = host._start_locked

        async def observe_first_exit(mirror: Path) -> None:
            nonlocal starts
            starts += 1
            await original_start(mirror)
            if starts == 1:
                assert host._process is not None
                await host._process.wait()

        monkeypatch.setattr(host, "_start_locked", observe_first_exit)
        result = await host.build(_snapshot())
        assert result.node_count == 1
        assert starts == 2
        # The known-dead child received no bytes; only the one replay has a
        # request ID. There is no retry loop hidden behind this boundary.
        assert host._request_number == 1
        await host.close()
        assert list((tmp_path / "scratch").iterdir()) == []

    asyncio.run(scenario())


def test_dead_replacement_child_during_query_replay_is_a_stable_engine_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        host = TrailmarkChildHost(
            scratch,
            child_command=(
                sys.executable,
                "-I",
                str(FAULT_CHILD),
                "exit-after-first-build",
                str(tmp_path / "first-child.marker"),
            ),
            build_timeout_seconds=CHILD_BUDGET_SECONDS,
            query_timeout_seconds=CHILD_BUDGET_SECONDS,
            stop_timeout_seconds=CHILD_BUDGET_SECONDS,
        )
        await host.build(_snapshot())
        assert host._process is not None
        await host._process.wait()
        original_start = host._start_locked

        async def start_and_observe_exit(mirror: Path) -> None:
            await original_start(mirror)
            assert host._process is not None
            await host._process.wait()

        monkeypatch.setattr(host, "_start_locked", start_and_observe_exit)
        # The query is definitely unprocessed, so the host restarts once and
        # rebuilds; the replacement is dead before that rebuild is handed off.
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.summary()
        assert rejected.value.code == "code_analysis_engine_failed"
        assert rejected.value.retryable
        assert host.pid is None and not host.mirror_exists
        await host.close()
        assert list(scratch.iterdir()) == []

    asyncio.run(scenario())


def test_crash_after_request_acceptance_is_never_replayed(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "crash")
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.build(_snapshot())
        assert rejected.value.code == "code_analysis_engine_failed"
        assert rejected.value.retryable
        assert host._request_number == 1
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_crash_after_query_acceptance_reaps_child_and_erases_mirror(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "crash-query")
        await host.build(_snapshot())
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.summary()
        assert rejected.value.code == "code_analysis_engine_failed"
        assert rejected.value.retryable
        assert host._request_number == 2
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_stderr_flood_is_drained_but_never_retained_as_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "stderr-flood")
        result = await host.build(_snapshot())
        assert result.node_count == 1
        assert host.stderr_observed is True
        assert not hasattr(host, "stderr")
        assert "xxxx" not in repr(host.__dict__)
        await host.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "code", "retryable"),
    [
        ("crash", "code_analysis_engine_failed", True),
        ("oom", "code_analysis_capacity_exceeded", False),
        ("oom-exit", "code_analysis_capacity_exceeded", False),
        ("killed", "code_analysis_engine_failed", True),
        ("malformed", "code_analysis_engine_failed", True),
        ("partial", "code_analysis_engine_failed", True),
        ("wrong-id", "code_analysis_engine_failed", True),
        ("bad-result", "code_analysis_engine_failed", True),
        ("oversized", "code_analysis_capacity_exceeded", False),
        ("hang", "code_analysis_build_timeout", True),
    ],
)
def test_fault_classification_is_stable_and_scratch_is_immediately_reusable(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    mode: str,
    code: str,
    retryable: bool,
) -> None:
    marker = term_marker(tmp_path_factory)

    async def scenario() -> None:
        # Only the hanging child meets a deadline short enough to expire.
        host = _fault_host(
            tmp_path,
            mode,
            build_timeout=0.08 if mode == "hang" else CHILD_BUDGET_SECONDS,
            marker=marker if mode == "hang" else None,
        )
        with pytest.raises(TrailmarkHostError) as rejected:
            await through_term_grace(host.build(_snapshot()), marker)
        assert rejected.value.code == code
        assert rejected.value.retryable is retryable
        assert str(rejected.value) == f"Code analysis child failed ({code})"
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

        clean = TrailmarkChildHost(tmp_path)
        assert (await clean.build(_snapshot("7"))).node_count >= 1
        await clean.close()
        assert list(tmp_path.iterdir()) == []

    run_with_advanceable_clock(scenario())


def test_memory_exhaustion_reaches_the_model_as_non_retryable_capacity(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = code_analysis._CodeAnalysisSession(object(), graph_scratch=tmp_path)
        session._graph_host = _fault_host(tmp_path, "oom")
        with pytest.raises(CodeAnalysisError) as rejected:
            await session._ensure_graph(_snapshot())
        assert rejected.value.code == "code_analysis_capacity_exceeded"
        assert rejected.value.retryable is False
        await session.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def _fail_first_parser_acquisition(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    original = source_languages.load_parser

    def load_parser(language: source_languages.Language) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            raise TypeError(f"parser factory defect {DEFECT_CANARY}")
        return original(language)

    monkeypatch.setattr(source_languages, "load_parser", load_parser)
    return calls


def _fail_first_name_extraction(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    original = source_names.extract_node_name

    def extract_name(node: Any, source: bytes, preferred_field: str = "") -> str | None:
        calls[0] += 1
        if calls[0] == 1:
            raise TypeError(f"extraction defect {DEFECT_CANARY}")
        return original(node, source, preferred_field)

    monkeypatch.setattr(code_analysis.language_support, "extract_node_name", extract_name)
    return calls


@pytest.mark.parametrize(
    "inject",
    [_fail_first_parser_acquisition, _fail_first_name_extraction],
    ids=["parser-acquisition", "extraction"],
)
def test_unexpected_parser_path_failure_is_a_tool_error_not_parse_error_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject: Callable[[pytest.MonkeyPatch], list[int]],
) -> None:
    calls = inject(monkeypatch)
    source = "class Socket {\npublic:\n  int target() const { return 1; }\n};\n"

    async def scenario() -> None:
        state = WorkerState()
        reader = _TreeReader({"include/socket.hh": source})
        tools = await CodeAnalysisToolsetFactory().create_selected(
            selected=("list_symbols", "search_def"),
            allocation_id="allocation-parser-defect",
            run_id="run-parser-defect",
            namespace="analysis",
            runtime_settings=RuntimeSettings(
                llm_gateway_url="https://llm.example/v1",
                llm_gateway_token="temporary-token",
                artifact_api_url="https://server.example/private/v1/artifacts",
                request_timeout_seconds=10,
            ),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
            resources=state.resources,
            project_workspace=reader,
        )
        model = scripted_model(
            [
                tool_call("list_symbols", {}, call_id="list-defect"),
                tool_call("list_symbols", {}, call_id="list-after-defect"),
                tool_call("search_def", {"symbol": "target"}, call_id="search-after-defect"),
                terminal_text("Recovered from a Runtime parser defect"),
            ]
        )
        runtime = await create_runtime(tmp_path / "runtime", state, dict(tools), model)

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        session_id = runtime._session_lifecycle.shared_session_id
        assert session_id is not None
        adk_session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=session_id,
        )
        assert adk_session is not None
        responses = [
            part.function_response.response
            for event in adk_session.events
            for part in (event.content.parts if event.content is not None else [])
            if part.function_response is not None
        ]
        assert len(responses) == 3
        # The existing Worker tool-error boundary bounds the defect: a stable
        # non-retryable code and the exception class, never its message.
        assert responses[0] == {
            "ok": False,
            "error": {
                "code": "tool_call_failed",
                "message": "list_symbols failed (TypeError)",
                "retryable": False,
            },
        }
        # Nothing about the failed call was cached as source coverage.
        listed, found = responses[1:]
        assert [(item["path"], item["name"]) for item in listed["items"]] == [
            ("include/socket.hh", "Socket"),
            ("include/socket.hh", "target"),
        ]
        assert [(item["path"], item["name"]) for item in found["items"]] == [
            ("include/socket.hh", "target")
        ]
        for response in (listed, found):
            assert response["coverage"]["parseErrors"] == 0
            assert response["coverage"]["incomplete"] is False
        assert calls[0] > 1

        recorded = state.metrics.tool_calls
        assert [call.tool for call in recorded] == ["list_symbols", "list_symbols", "search_def"]
        assert recorded[0].error is not None
        assert recorded[0].error.code == "tool_call_failed"
        assert recorded[0].error.retryable is False
        assert [call.error for call in recorded[1:]] == [None, None]
        assert DEFECT_CANARY not in repr(adk_session.events) + repr(state.metrics.snapshot())
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


class _TreeReader:
    def __init__(self, files: dict[str, str]) -> None:
        directories = {
            "/".join(path.split("/")[:depth])
            for path in files
            for depth in range(1, path.count("/") + 1)
        }
        self._tree = ManagedWorkspaceTree(directories=directories, text_files=dict(files))

    async def snapshot(self) -> WorkspaceSnapshot:
        return self._tree.snapshot()


def _fault_host(
    root: Path,
    mode: str,
    *,
    build_timeout: float = CHILD_BUDGET_SECONDS,
    marker: Path | None = None,
) -> TrailmarkChildHost:
    marker_argument = () if marker is None else (str(marker),)
    return TrailmarkChildHost(
        root,
        child_command=(sys.executable, "-I", str(FAULT_CHILD), mode, *marker_argument),
        build_timeout_seconds=build_timeout,
        query_timeout_seconds=CHILD_BUDGET_SECONDS,
        stop_timeout_seconds=CHILD_BUDGET_SECONDS,
    )


def _snapshot(tag: str = "6") -> WorkspaceSnapshot:
    source = "def main():\n    return 1\n"
    return WorkspaceSnapshot(
        directories=(),
        files=(WorkspaceTextFile("app.py", source, len(source.encode("utf-8"))),),
        binary_paths=(),
        digest="sha256:" + tag * 64,
    )
