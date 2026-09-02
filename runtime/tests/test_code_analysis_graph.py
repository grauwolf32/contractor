from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.metrics import MetricsState
from contractor_runtime.projectfs.storage import ManagedWorkspaceTree, WorkspaceSnapshot
from contractor_runtime.toolsets.code_analysis import (
    CORE_GRAPH_TOOLS,
    GRAPH_TOOLS,
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.toolsets.code_analysis_ids import (
    decode_symbol_id,
    encode_symbol_id,
    symbol_id_matches_upstream,
)
from contractor_runtime.workspace import AllocationWorkspace


def test_symbol_id_is_allocation_local_and_bound_to_complete_upstream_id() -> None:
    key = bytes(range(32))
    digest = "sha256:" + "a" * 64
    symbol_id = encode_symbol_id(key, digest, 7, "module:Class.method")
    decoded = decode_symbol_id(key, symbol_id)

    assert decoded.snapshot_digest == digest
    assert decoded.index == 7
    assert symbol_id_matches_upstream(key, decoded, "module:Class.method") is True
    assert symbol_id_matches_upstream(key, decoded, "module:Other.method") is False
    with pytest.raises(ValueError):
        decode_symbol_id(bytes(reversed(key)), symbol_id)


def test_core_graph_tools_expose_duplicate_identity_and_exact_relationships(
    tmp_path: Path,
) -> None:
    files = {
        "a.py": (
            "class A:\n"
            "    def target(self):\n"
            "        return 1\n"
            "    def caller_a(self):\n"
            "        return self.target()\n"
        ),
        "b.py": (
            "class B:\n"
            "    def target(self):\n"
            "        return 2\n"
            "    def caller_b(self):\n"
            "        return self.target()\n"
        ),
    }

    async def scenario() -> None:
        tools, _, scratch = await _tools(tmp_path, MutableReader(files))
        summary = await tools["graph_summary"]()
        assert set(summary) == {
            "nodeCount",
            "callEdgeCount",
            "entrypointCount",
            "languages",
            "coverage",
        }
        assert summary["nodeCount"] == 8
        assert summary["callEdgeCount"] == 2
        assert summary["languages"] == ["python"]

        found = await tools["find_symbol"]("TARGET")
        assert found["observedTotal"] == 2
        assert found["truncated"] is False
        assert [item["path"] for item in found["items"]] == ["a.py", "b.py"]
        ids = [item["symbolId"] for item in found["items"]]
        assert len(set(ids)) == 2
        assert all(value.startswith("cas1.") for value in ids)
        assert all("target" not in value.casefold() for value in ids)

        qualified = await tools["find_symbol"]("A.target")
        assert [(item["name"], item["path"]) for item in qualified["items"]] == [("target", "a.py")]

        first_callers = await tools["find_callers"](ids[0])
        second_callers = await tools["find_callers"](ids[1])
        assert [(item["name"], item["confidence"]) for item in first_callers["items"]] == [
            ("caller_a", "certain")
        ]
        assert [(item["name"], item["confidence"]) for item in second_callers["items"]] == [
            ("caller_b", "certain")
        ]

        caller = (await tools["find_symbol"]("caller_a"))["items"][0]
        callees = await tools["find_callees"](caller["symbolId"])
        assert [(item["name"], item["path"]) for item in callees["items"]] == [("target", "a.py")]

        with pytest.raises(CodeAnalysisError) as bare_name:
            await tools["find_callers"]("target")
        assert bare_name.value.code == "code_analysis_symbol_not_found"
        prefix, body, signature = ids[0].split(".")
        tampered_id = ".".join(
            (prefix, body, ("A" if signature[0] != "A" else "B") + signature[1:])
        )
        with pytest.raises(CodeAnalysisError) as tampered:
            await tools["find_callers"](tampered_id)
        assert tampered.value.code == "code_analysis_symbol_not_found"
        assert len(tuple(scratch.glob("code-analysis-mirror-*"))) == 1
        await _close(tools)
        assert list(scratch.iterdir()) == []

    asyncio.run(scenario())


def test_callees_include_bounded_unresolved_proxy_projection(tmp_path: Path) -> None:
    files = {
        "service.py": (
            "class Service:\n    def handle(self):\n        return self.dependency.do_thing()\n"
        )
    }

    async def scenario() -> None:
        tools, _, _ = await _tools(tmp_path, MutableReader(files))
        handle = (await tools["find_symbol"]("Service.handle"))["items"][0]
        result = await tools["find_callees"](handle["symbolId"])
        assert result["observedTotal"] == 1
        assert result["items"][0]["kind"] == "proxy"
        assert result["items"][0]["path"] == "service.py"
        assert result["items"][0]["confidence"] in {
            "certain",
            "inferred",
            "uncertain",
        }
        assert set(result["items"][0]) == {
            "symbolId",
            "name",
            "kind",
            "path",
            "line",
            "endLine",
            "column",
            "confidence",
        }
        await _close(tools)

    asyncio.run(scenario())


def test_find_symbol_pagination_is_deterministic_and_integrity_protected(
    tmp_path: Path,
) -> None:
    files = {f"module_{index:03d}.py": "def target():\n    return 1\n" for index in range(60)}

    async def scenario() -> None:
        tools, _, _ = await _tools(tmp_path, MutableReader(files))
        first = await tools["find_symbol"]("target", limit=20)
        assert len(first["items"]) == 20
        assert first["observedTotal"] == 60
        assert first["truncated"] is True
        assert first["nextCursor"]
        second = await tools["find_symbol"]("TARGET", cursor=first["nextCursor"], limit=20)
        assert len(second["items"]) == 20
        assert first["items"][-1]["path"] < second["items"][0]["path"]
        assert {item["symbolId"] for item in first["items"]}.isdisjoint(
            item["symbolId"] for item in second["items"]
        )
        encoded_body, encoded_signature = first["nextCursor"].split(".", 1)
        tampered = (
            encoded_body
            + "."
            + (("A" if encoded_signature[0] != "A" else "B") + encoded_signature[1:])
        )
        with pytest.raises(CodeAnalysisError) as rejected:
            await tools["find_symbol"]("target", cursor=tampered, limit=20)
        assert rejected.value.code == "code_analysis_cursor_invalid"
        await _close(tools)

    asyncio.run(scenario())


def test_edit_invalidates_child_cursor_and_symbol_id_before_rebuild(tmp_path: Path) -> None:
    reader = MutableReader(
        {
            "a.py": "def target():\n    return 1\n",
            "b.py": "def target():\n    return 2\n",
        }
    )

    async def scenario() -> None:
        tools, _, scratch = await _tools(tmp_path, reader)
        first = await tools["find_symbol"]("target", limit=1)
        old_id = first["items"][0]["symbolId"]
        old_cursor = first["nextCursor"]
        assert old_cursor
        assert len(tuple(scratch.glob("code-analysis-mirror-*"))) == 1

        reader.write("a.py", "def replacement():\n    return 3\n")
        with pytest.raises(CodeAnalysisError) as stale:
            await tools["find_callers"](old_id)
        assert stale.value.code == "code_analysis_stale_symbol"
        assert list(scratch.iterdir()) == []

        with pytest.raises(CodeAnalysisError) as stale_cursor:
            await tools["find_symbol"]("target", cursor=old_cursor, limit=1)
        assert stale_cursor.value.code == "code_analysis_workspace_changed"
        assert list(scratch.iterdir()) == []

        fresh = await tools["find_symbol"]("replacement")
        assert fresh["observedTotal"] == 1
        assert fresh["items"][0]["symbolId"] != old_id
        assert len(tuple(scratch.glob("code-analysis-mirror-*"))) == 1
        await _close(tools)

    asyncio.run(scenario())


def test_symbol_ids_cannot_cross_allocation_sessions(tmp_path: Path) -> None:
    files = {"app.py": "def target():\n    return 1\n"}

    async def scenario() -> None:
        factory = CodeAnalysisToolsetFactory(
            workspace_storage="local", graph_probe_root=tmp_path / "probe"
        )
        assert await factory.probe() >= CORE_GRAPH_TOOLS
        first, _, first_scratch = await _tools(
            tmp_path / "first",
            MutableReader(files),
            factory=factory,
        )
        second, _, second_scratch = await _tools(
            tmp_path / "second",
            MutableReader(files),
            factory=factory,
        )
        first_id = (await first["find_symbol"]("target"))["items"][0]["symbolId"]
        await second["graph_summary"]()
        with pytest.raises(CodeAnalysisError) as rejected:
            await second["find_callers"](first_id)
        assert rejected.value.code == "code_analysis_symbol_not_found"
        assert len(tuple(first_scratch.glob("code-analysis-mirror-*"))) == 1
        assert len(tuple(second_scratch.glob("code-analysis-mirror-*"))) == 1
        await _close(first)
        await _close(second)

    asyncio.run(scenario())


def test_graph_results_are_bounded_relative_and_metrics_are_content_free(
    tmp_path: Path,
) -> None:
    query_canary = "SensitiveGraphNameCanary"
    files = {
        "nested/app.py": (
            f"def {query_canary}():\n    return 1\n\ndef caller():\n    return {query_canary}()\n"
        )
    }

    async def scenario() -> None:
        tools, metrics, scratch = await _tools(tmp_path, MutableReader(files))
        found = await tools["find_symbol"](query_canary)
        callers = await tools["find_callers"](found["items"][0]["symbolId"])
        serialized = json.dumps({"found": found, "callers": callers}, sort_keys=True)
        assert str(tmp_path) not in serialized
        assert len(serialized.encode("utf-8")) < 256 * 1024
        assert all(not item["path"].startswith("/") for item in found["items"])

        retained = json.dumps(metrics.snapshot(), sort_keys=True)
        assert query_canary not in retained
        assert str(tmp_path) not in retained
        assert "cas1." not in retained
        assert metrics.tool_calls[-1].arguments == {}
        await _close(tools)
        assert list(scratch.iterdir()) == []

    asyncio.run(scenario())


def test_graph_collection_stops_at_encoded_result_ceiling_with_resumable_cursor(
    tmp_path: Path,
) -> None:
    component = "long_segment_" + "x" * 35
    prefix = "/".join(component + f"_{index:02d}" for index in range(31))
    files = {
        f"{prefix}/module_{index:03d}.py": "def target():\n    return 1\n" for index in range(180)
    }

    async def scenario() -> None:
        tools, _, _ = await _tools(tmp_path, MutableReader(files))
        first = await tools["find_symbol"]("target", limit=200)
        assert 0 < len(first["items"]) < 180
        assert first["observedTotal"] == 180
        assert first["truncated"] is True
        assert first["nextCursor"]
        assert len(json.dumps(first, separators=(",", ":")).encode("utf-8")) <= 256 * 1024

        second = await tools["find_symbol"]("target", cursor=first["nextCursor"], limit=200)
        assert second["items"]
        assert first["items"][-1]["path"] < second["items"][0]["path"]
        await _close(tools)

    asyncio.run(scenario())


def test_path_tools_use_exact_ids_depth_and_limit_plus_one(tmp_path: Path) -> None:
    files = {
        "app.py": (
            "def target():\n"
            "    return 1\n\n"
            "def left():\n"
            "    return target()\n\n"
            "def right():\n"
            "    return target()\n\n"
            "def entry():\n"
            "    left()\n"
            "    return right()\n\n"
            "def main():\n"
            "    return entry()\n"
        )
    }

    async def scenario() -> None:
        tools, metrics, _ = await _tools(tmp_path, MutableReader(files))
        symbols = {
            name: (await tools["find_symbol"](name))["items"][0]["symbolId"]
            for name in ("main", "target")
        }
        first = await tools["paths_between"](
            symbols["main"],
            symbols["target"],
            max_depth=4,
            limit=1,
        )
        assert set(first) == {"items", "truncated", "coverage"}
        assert first["truncated"] is True
        assert len(first["items"]) == 1
        assert [item["name"] for item in first["items"][0]] in (
            ["main", "entry", "left", "target"],
            ["main", "entry", "right", "target"],
        )
        assert all("confidence" not in item for item in first["items"][0])

        too_shallow = await tools["paths_between"](
            symbols["main"],
            symbols["target"],
            max_depth=3,
        )
        assert too_shallow["items"] == []
        assert too_shallow["truncated"] is False

        from_entrypoints = await tools["entrypoint_paths_to"](
            symbols["target"],
            max_depth=4,
            limit=1,
        )
        assert from_entrypoints["items"]
        assert from_entrypoints["truncated"] is True
        assert from_entrypoints["items"][0][-1]["symbolId"] == symbols["target"]

        with pytest.raises(CodeAnalysisError) as bare_name:
            await tools["paths_between"]("main", symbols["target"])
        assert bare_name.value.code == "code_analysis_symbol_not_found"
        with pytest.raises(CodeAnalysisError) as bad_depth:
            await tools["entrypoint_paths_to"](symbols["target"], max_depth=0)
        assert bad_depth.value.code == "code_analysis_input_invalid"
        retained = json.dumps(metrics.snapshot(), sort_keys=True)
        assert symbols["target"] not in retained
        await _close(tools)

    asyncio.run(scenario())


def test_entrypoint_paths_keep_duplicate_targets_exact(tmp_path: Path) -> None:
    files = {
        "a.py": "def target():\n    return 'a'\n\ndef main():\n    return target()\n",
        "b.py": "def target():\n    return 'b'\n\ndef main():\n    return target()\n",
    }

    async def scenario() -> None:
        tools, _, _ = await _tools(tmp_path, MutableReader(files))
        targets = (await tools["find_symbol"]("target"))["items"]
        assert [item["path"] for item in targets] == ["a.py", "b.py"]

        selected = targets[0]
        result = await tools["entrypoint_paths_to"](
            selected["symbolId"],
            max_depth=2,
            limit=10,
        )
        assert result["items"]
        assert all(path[-1]["symbolId"] == selected["symbolId"] for path in result["items"])
        assert all(path[-1]["path"] == "a.py" for path in result["items"])
        assert all(
            len({item["symbolId"] for item in path}) == len(path) for path in result["items"]
        )
        await _close(tools)

    asyncio.run(scenario())


def test_attack_surface_complexity_and_exception_tools_are_slim(tmp_path: Path) -> None:
    files = {
        "app.py": (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            "@app.get('/items')\n"
            "def route(value):\n"
            "    if value:\n"
            "        for item in value:\n"
            "            if item:\n"
            "                raise ValueError('bad')\n"
            "    return 1\n\n"
            "def simple():\n"
            "    raise TypeError('other')\n"
        )
    }

    async def scenario() -> None:
        tools, _, _ = await _tools(tmp_path, MutableReader(files))
        surface = await tools["attack_surface"]()
        route = next(item for item in surface["items"] if item["name"] == "route")
        assert route["entrypointKind"] == "api"
        assert route["trustLevel"] == "untrusted_external"
        assert route["assetValue"] == "high"
        assert "attributes" not in route
        assert not route["path"].startswith("/")

        hotspots = await tools["complexity_hotspots"](threshold=2)
        assert hotspots["items"]
        assert hotspots["items"][0]["name"] == "route"
        assert hotspots["items"][0]["complexity"] >= 2
        assert [item["complexity"] for item in hotspots["items"]] == sorted(
            (item["complexity"] for item in hotspots["items"]),
            reverse=True,
        )

        raises = await tools["functions_that_raise"]("ValueError")
        assert [(item["name"], item["path"]) for item in raises["items"]] == [("route", "app.py")]
        assert (await tools["functions_that_raise"]("valueerror"))["items"] == []
        await _close(tools)

    asyncio.run(scenario())


def test_advanced_collection_cursor_is_argument_bound_and_resumable(tmp_path: Path) -> None:
    functions = "\n\n".join(
        f"def raises_{index:03d}():\n    raise ValueError('{index}')" for index in range(35)
    )

    async def scenario() -> None:
        tools, _, _ = await _tools(
            tmp_path,
            MutableReader({"errors.py": functions + "\n"}),
        )
        first = await tools["functions_that_raise"]("ValueError", limit=10)
        assert len(first["items"]) == 10
        assert first["observedTotal"] == 35
        assert first["truncated"] is True
        second = await tools["functions_that_raise"](
            "ValueError",
            cursor=first["nextCursor"],
            limit=10,
        )
        assert first["items"][-1]["name"] < second["items"][0]["name"]
        with pytest.raises(CodeAnalysisError) as wrong_argument:
            await tools["functions_that_raise"](
                "TypeError",
                cursor=first["nextCursor"],
                limit=10,
            )
        assert wrong_argument.value.code == "code_analysis_cursor_invalid"
        await _close(tools)

    asyncio.run(scenario())


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

    def write(self, path: str, text: str) -> None:
        self._tree.directories.update(_directories({path: text}))
        self._tree.text_files[path] = text


async def _tools(
    root: Path,
    reader: MutableReader,
    *,
    factory: CodeAnalysisToolsetFactory | None = None,
) -> tuple[dict[str, Any], MetricsState, Path]:
    root.mkdir(parents=True, exist_ok=True)
    selected_factory = factory or CodeAnalysisToolsetFactory(
        workspace_storage="local", graph_probe_root=root / "probe"
    )
    if factory is None:
        assert await selected_factory.probe() >= GRAPH_TOOLS
    scratch = root / "allocation"
    scratch.mkdir(mode=0o700)
    metrics = MetricsState()
    tools = await selected_factory.create_selected(
        selected=tuple(sorted(GRAPH_TOOLS)),
        allocation_id=f"allocation-{root.name}",
        run_id="run-code-graph",
        namespace="analysis",
        runtime_settings=RuntimeSettings(
            llm_gateway_url="https://llm.example/v1",
            llm_gateway_token="temporary-token",
            artifact_api_url="https://server.example/private/v1/artifacts",
            request_timeout_seconds=10,
        ),
        workspace=AllocationWorkspace(root=root, path=scratch),
        state=SimpleNamespace(metrics=metrics),
        project_workspace=reader,
    )
    return dict(tools), metrics, scratch


async def _close(tools: dict[str, Any]) -> None:
    for tool in reversed(tuple(tools.values())):
        await tool.close()


def _directories(files: dict[str, str]) -> set[str]:
    result: set[str] = set()
    for path in files:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            result.add(str(parent))
            parent = parent.parent
    return result
