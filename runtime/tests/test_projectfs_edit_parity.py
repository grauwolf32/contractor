from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from test_edit_files_toolset import hydrated_workspace, make_tools

from contractor_runtime.allocation import WorkerState


def test_complete_edit_matrix_matches_all_direct_overlay_local_memory_backends(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        results: list[Any] = []
        for storage in ("local", "memory"):
            for mode in ("direct", "overlay"):
                allocation_id = f"{storage}-{mode}"
                session, provider = await hydrated_workspace(tmp_path, storage, mode, allocation_id)
                state = WorkerState()
                tools = await make_tools(
                    tmp_path,
                    session.writer_view(),
                    state,
                    [
                        "append_file",
                        "cp",
                        "edit",
                        "insert_line",
                        "mkdir",
                        "mv",
                        "replace_range",
                        "rm",
                        "write_file",
                    ],
                )
                physical_root = f"{session.storage.root}/run_workdir"
                if storage != "local" or mode != "direct":
                    assert not Path(physical_root).exists()

                await tools["append_file"]("crlf.txt", "four\nfive")
                await tools["insert_line"]("crlf.txt", 2, "inserted")
                await tools["edit"]("crlf.txt", "two\n", "TWO\n")
                await tools["replace_range"]("crlf.txt", 3, 4, "middle")
                await tools["mkdir"]("generated/deep", True)
                await tools["write_file"]("generated/new.txt", "new\n")
                await tools["cp"]("generated/new.txt", "generated/deep/copy.txt")
                await tools["mv"]("generated/deep/copy.txt", "generated/deep/moved.txt")
                await tools["rm"]("generated/new.txt")
                await tools["cp"]("tree", "tree-copy", True)
                await tools["rm"]("tree", True)

                assert await session.read_text("crlf.txt") == (
                    "one\r\ninserted\r\nmiddle\r\nfour\r\nfive"
                )
                snapshot = await session.snapshot()
                results.append(snapshot)
                assert all(call.arguments == {} for call in state.metrics.tool_calls)

                if storage == "local" and mode == "direct":
                    physical_crlf = Path(f"{physical_root}/crlf.txt").read_bytes()
                    assert physical_crlf == (await session.read_text("crlf.txt")).encode()
                    assert (
                        Path(f"{physical_root}/generated/deep/moved.txt").read_bytes() == b"new\n"
                    )
                    assert not Path(f"{physical_root}/tree").exists()
                else:
                    assert not Path(physical_root).exists()
                await provider.cleanup(session.storage)

        assert all(snapshot == results[0] for snapshot in results[1:])

    asyncio.run(scenario())
