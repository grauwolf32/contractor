"""Numbered display must preserve source coordinates and bounded pagination."""

import asyncio

import pytest
from google.adk.tools import FunctionTool
from test_filesystem_toolset import create_tools, workspace

import contractor_runtime.toolsets.filesystem.tools as filesystem
from contractor_runtime.allocation import WorkerState


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_numbered_reads_preserve_windows_newlines_and_byte_limits(tmp_path, monkeypatch, mode):
    async def scenario():
        session = await workspace(mode, "numbered")
        tools = await create_tools(
            filesystem.FilesystemToolsetFactory(),
            session.reader_view(),
            WorkerState(),
            tmp_path,
            ["read_file"],
        )
        read = tools["read_file"]
        declaration = FunctionTool(read)._get_declaration()
        assert "with_line_numbers" not in declaration.parameters_json_schema.get("required", [])
        plain = await read("docs/readme.txt", start_line=2, max_lines=2)
        numbered = await read("docs/readme.txt", start_line=2, max_lines=2, with_line_numbers=True)
        assert [line["text"] for line in plain["lines"]] == ["second", "last"]
        assert [line["text"] for line in numbered["lines"]] == ["2 | second", "3 | last"]
        assert [line["newline"] for line in numbered["lines"]] == ["lf", "none"]
        assert numbered["returnedBytes"] == plain["returnedBytes"] + len("2 | 3 | ")
        assert not numbered["truncated"]
        with pytest.raises(filesystem.FilesystemToolError):
            await read("docs/readme.txt", with_line_numbers="yes")
        monkeypatch.setattr(filesystem, "MAX_READ_BYTES", len("1 | ") - 1)
        bounded = await read("docs/readme.txt", with_line_numbers=True)
        assert bounded["lines"] == []
        assert bounded["truncated"] and bounded["nextLine"] == 1
        await read.close()
        await session.close()

    asyncio.run(scenario())


def test_numbered_reads_multibyte_empty_lines_and_eof(tmp_path, monkeypatch):
    async def scenario():
        session = await workspace("direct", "numbered-unicode")
        await session.write_text("unicode.txt", "é猫\r\n\nlast")
        await session.write_text("empty.txt", "")
        tools = await create_tools(
            filesystem.FilesystemToolsetFactory(),
            session.reader_view(),
            WorkerState(),
            tmp_path,
            ["read_file"],
        )
        read = tools["read_file"]
        result = await read("unicode.txt", with_line_numbers=True)
        assert [row["text"] for row in result["lines"]] == ["1 | é猫", "2 | ", "3 | last"]
        assert result["returnedBytes"] == len("1 | é猫2 | 3 | last".encode())
        assert (await read("empty.txt", with_line_numbers=True))["lines"] == []
        assert (await read("unicode.txt", start_line=4, with_line_numbers=True))["lines"] == []
        monkeypatch.setattr(filesystem, "MAX_READ_BYTES", len("1 | é".encode()) + 1)
        bounded = await read("unicode.txt", with_line_numbers=True)
        assert bounded["lines"][0]["text"] == "1 | é"
        assert bounded["lines"][0]["truncated"]
        assert bounded["returnedBytes"] <= filesystem.MAX_READ_BYTES
        monkeypatch.setattr(filesystem, "MAX_READ_BYTES", len("2 | "))
        empty_line = await read("unicode.txt", start_line=2, max_lines=1, with_line_numbers=True)
        assert empty_line["lines"][0]["text"] == "2 | "
        assert empty_line["nextLine"] == 3
        await read.close()
        await session.close()

    asyncio.run(scenario())
