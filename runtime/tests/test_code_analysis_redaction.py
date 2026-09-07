from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs.storage import ManagedWorkspaceTree, WorkspaceSnapshot
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.code_analysis.tools import (
    GRAPH_TOOLS,
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace


def test_code_analysis_retained_state_metrics_errors_and_scratch_are_content_free(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_canary = 'SOURCE_CANARY_"雪"/private'
    symbol_canary = "SensitiveSymbolCanary"
    path_canary = "private/PATH_CANARY_file.py"
    query_canary = 'QUERY_CANARY_"雪"/private'
    credential_canary = "CREDENTIAL_CANARY_never-retain"
    source = f"def {symbol_canary}():\n    value = {source_canary!r}\n    return value\n"

    async def scenario() -> None:
        root = tmp_path / "runtime-private"
        root.mkdir()
        factory = CodeAnalysisToolsetFactory(
            workspace_storage="local",
            graph_probe_root=root / "probe",
        )
        assert await factory.probe() >= GRAPH_TOOLS
        metrics = MetricsState()
        reader = MutableReader({path_canary: source})
        scratch = root / "allocation"
        scratch.mkdir()
        tools = dict(
            await factory.create_selected(
                selected=("search_def", "find_symbol", "find_callers"),
                allocation_id="allocation-redaction",
                run_id="run-redaction",
                namespace="analysis",
                runtime_settings=RuntimeSettings(
                    llm_gateway_url="https://llm.example/v1",
                    llm_gateway_token=credential_canary,
                    artifact_api_url="https://server.example/private/v1/artifacts",
                    request_timeout_seconds=10,
                ),
                workspace=AllocationWorkspace(root=root, path=scratch),
                state=SimpleNamespace(metrics=metrics),
                project_workspace=reader,
            )
        )
        session = tools["search_def"]._session

        shallow = await tools["search_def"](symbol_canary)
        assert source_canary in shallow["items"][0]["preview"]
        graph = await tools["find_symbol"](symbol_canary)
        symbol_id = graph["items"][0]["symbolId"]
        assert path_canary == graph["items"][0]["path"]
        assert symbol_canary not in symbol_id
        assert (await tools["find_symbol"](query_canary))["items"] == []
        with pytest.raises(CodeAnalysisError) as rejected:
            await tools["search_def"](query_canary, path=f"../{path_canary}")
        assert str(rejected.value) == (
            "Code analysis operation failed (code_analysis_input_invalid)"
        )

        retained_before_close = json.dumps(
            {
                "metrics": metrics.snapshot(),
                "report": metrics.build_report(
                    report_id="redaction-report",
                    duration_ms=1,
                ).model_dump(mode="json", by_alias=True),
                "logs": [record.getMessage() for record in caplog.records],
                "error": str(rejected.value),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        _assert_absent(
            retained_before_close,
            source_canary,
            symbol_canary,
            path_canary,
            query_canary,
            credential_canary,
            symbol_id,
        )

        await _close(tools)
        assert session._reader is None
        assert session._graph_host is None
        assert session._file_cache == {}
        assert session._parsers == {}
        assert session._graph_result is None
        assert session._digest is None
        assert set(session._cursor_key) == {0}
        assert list(scratch.iterdir()) == []
        retained_after_close = repr(session.__dict__) + json.dumps(
            metrics.snapshot(), ensure_ascii=False, sort_keys=True
        )
        _assert_absent(
            retained_after_close,
            source_canary,
            symbol_canary,
            path_canary,
            query_canary,
            credential_canary,
            symbol_id,
        )

    asyncio.run(scenario())


def _assert_absent(rendered: str, *values: str) -> None:
    for value in values:
        variants = {
            value,
            json.dumps(value, ensure_ascii=False)[1:-1],
            json.dumps(value, ensure_ascii=True)[1:-1],
            quote(value, safe=""),
        }
        assert all(candidate not in rendered for candidate in variants)


class MutableReader:
    def __init__(self, files: dict[str, str]) -> None:
        self._tree = ManagedWorkspaceTree(
            directories=_directories(files),
            text_files=dict(files),
        )

    async def snapshot(self) -> WorkspaceSnapshot:
        return self._tree.snapshot()

    async def read_text(self, path: str) -> str:
        return self._tree.text_files[path]


async def _close(tools: dict[str, object]) -> None:
    for tool in reversed(tuple(tools.values())):
        await tool.close()  # type: ignore[attr-defined]


def _directories(files: dict[str, str]) -> set[str]:
    result: set[str] = set()
    for path in files:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            result.add(str(parent))
            parent = parent.parent
    return result
