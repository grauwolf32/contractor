import asyncio
import json
import subprocess

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from contractor.tools.likec4 import (
    DEFAULT_LIKEC4_PATH,
    Likec4ExecutionError,
    Likec4Linter,
    Likec4NotFoundError,
    Likec4OutputError,
    Likec4SourceNotFoundError,
    likec4_tools,
)


@pytest.fixture
def mock_likec4_in_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend `likec4` is the resolved binary so the linter can be constructed."""
    monkeypatch.setattr(
        "contractor.tools.likec4.shutil.which",
        lambda name: "/usr/local/bin/likec4" if name == "likec4" else None,
    )


@pytest.fixture
def linter(mock_likec4_in_path: None) -> Likec4Linter:
    return Likec4Linter()


@pytest.fixture
def fs() -> MemoryFileSystem:
    """Fresh MemoryFileSystem per test (it's a singleton, so wipe its store)."""
    fs = MemoryFileSystem()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    return fs


def _proc(stdout: str, *, returncode: int = 0, stderr: bytes = b"") -> object:
    class CompletedProcessMock:
        pass

    proc = CompletedProcessMock()
    proc.returncode = returncode
    proc.stdout = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# _resolve_command
# ---------------------------------------------------------------------------

def test_resolve_command_prefers_likec4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    assert Likec4Linter._resolve_command() == ["likec4"]


def test_resolve_command_falls_back_to_npx_with_autoconfirm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.shutil.which",
        lambda name: "/usr/bin/npx" if name == "npx" else None,
    )
    assert Likec4Linter._resolve_command() == ["npx", "--yes", "likec4"]


def test_resolve_command_falls_back_to_bunx_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.shutil.which",
        lambda name: "/usr/bin/bunx" if name == "bunx" else None,
    )
    assert Likec4Linter._resolve_command() == ["bunx", "likec4"]


def test_resolve_command_raises_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.shutil.which",
        lambda _name: None,
    )
    with pytest.raises(Likec4NotFoundError):
        Likec4Linter._resolve_command()


# ---------------------------------------------------------------------------
# validate (content-based core)
# ---------------------------------------------------------------------------

def test_validate_returns_empty_list_for_clean_dict_output(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps({"valid": True, "errors": [], "stats": {"elements": 4}})
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(payload),
    )

    assert linter.validate("specification { } model { } views { }") == []


def test_validate_returns_errors_list_from_dict_output(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    issues = [{"message": "bad", "range": {}}]
    payload = json.dumps({"valid": False, "errors": issues, "stats": {}})
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(payload, returncode=1),
    )

    assert linter.validate("specification { }") == issues


def test_validate_accepts_legacy_bare_list_output(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    issues = [{"message": "x"}]
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps(issues)),
    )

    assert linter.validate("specification { }") == issues


def test_validate_raises_execution_error_on_empty_output(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc("", returncode=2, stderr=b"boom"),
    )

    with pytest.raises(Likec4ExecutionError, match="produced no output"):
        linter.validate("specification { }")


def test_validate_raises_output_error_on_invalid_json(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc("not valid json"),
    )

    with pytest.raises(Likec4OutputError, match="failed to parse"):
        linter.validate("specification { }")


def test_validate_raises_output_error_when_dict_missing_errors_key(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps({"valid": True})),
    )

    with pytest.raises(Likec4OutputError, match="unexpected likec4 output format"):
        linter.validate("specification { }")


def test_validate_raises_output_error_for_scalar_json(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps("hello")),
    )

    with pytest.raises(Likec4OutputError, match="unexpected likec4 output format") as excinfo:
        linter.validate("specification { }")
    assert "expected list or dict" in excinfo.value.details


def test_validate_raises_execution_error_on_timeout(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["likec4"], timeout=1.0)

    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        _raise_timeout,
    )

    with pytest.raises(Likec4ExecutionError, match="timed out"):
        linter.validate("specification { }", timeout=1.0)


def test_validate_subprocess_call_uses_devnull_stdin_and_timeout(
    linter: Likec4Linter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: hardening against interactive prompts and runaway processes."""
    captured: dict = {}

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _proc(json.dumps({"valid": True, "errors": [], "stats": {}}))

    monkeypatch.setattr("contractor.tools.likec4.subprocess.run", _capture)

    linter.validate("specification { }", timeout=42.0)

    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["timeout"] == 42.0
    assert captured["cmd"][:4] == ["likec4", "validate", "--json", "--no-layout"]


# ---------------------------------------------------------------------------
# validate_path (fs-backed)
# ---------------------------------------------------------------------------

def test_validate_path_raises_when_file_missing(
    linter: Likec4Linter, fs: MemoryFileSystem
) -> None:
    with pytest.raises(Likec4SourceNotFoundError, match="/architecture.c4"):
        linter.validate_path(fs, "/architecture.c4")


def test_validate_path_reads_file_and_validates(
    linter: Likec4Linter, fs: MemoryFileSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs.pipe_file("/architecture.c4", b"specification { } model { } views { }")
    seen: dict = {}

    def _fake_run(cmd, **kwargs):
        # The temp file written by the linter should mirror the fs content.
        for token in cmd:
            if token.endswith(".c4"):
                with open(token, "rb") as fh:
                    seen["src"] = fh.read()
        return _proc(json.dumps({"valid": True, "errors": [], "stats": {}}))

    monkeypatch.setattr("contractor.tools.likec4.subprocess.run", _fake_run)

    assert linter.validate_path(fs, "/architecture.c4") == []
    assert seen["src"] == b"specification { } model { } views { }"


# ---------------------------------------------------------------------------
# likec4_tools async tool wrapper
# ---------------------------------------------------------------------------

def test_likec4_tools_factory_exposes_validate_likec4(
    mock_likec4_in_path: None, fs: MemoryFileSystem
) -> None:
    tools = likec4_tools(fs=fs)
    assert len(tools) == 1
    assert tools[0].__name__ == "validate_likec4"


def test_likec4_tools_default_path_constant() -> None:
    assert DEFAULT_LIKEC4_PATH == "/architecture.c4"


def test_validate_likec4_tool_returns_error_for_missing_file(
    mock_likec4_in_path: None, fs: MemoryFileSystem
) -> None:
    tool = likec4_tools(fs=fs)[0]
    result = asyncio.run(tool())
    assert result == {"error": "likec4 source file not found at '/architecture.c4'"}


def test_validate_likec4_tool_returns_clean_result(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/architecture.c4", b"specification { } model { } views { }")
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps({"valid": True, "errors": [], "stats": {}})),
    )

    tool = likec4_tools(fs=fs)[0]
    result = asyncio.run(tool())
    assert result == {"result": []}


def test_validate_likec4_tool_returns_issues_on_error_payload(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/architecture.c4", b"specification { }")
    issues = [{"message": "bad", "range": {}}]
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(
            json.dumps({"valid": False, "errors": issues, "stats": {}}),
            returncode=1,
        ),
    )

    tool = likec4_tools(fs=fs)[0]
    result = asyncio.run(tool())
    assert result == {"result": issues}


def test_validate_likec4_tool_accepts_custom_path(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/sub/model.c4", b"specification { }")
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps({"valid": True, "errors": [], "stats": {}})),
    )

    tool = likec4_tools(fs=fs)[0]
    assert asyncio.run(tool(path="/sub/model.c4")) == {"result": []}


def test_validate_likec4_tool_wraps_execution_error_with_details(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/architecture.c4", b"specification { }")
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc("", returncode=2, stderr=b"boom"),
    )

    tool = likec4_tools(fs=fs)[0]
    result = asyncio.run(tool())
    assert result["error"] == "likec4 produced no output"
    assert result["details"] == "boom"


def test_validate_likec4_tool_wraps_output_error_with_raw_output(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/architecture.c4", b"specification { }")
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc("not valid json"),
    )

    tool = likec4_tools(fs=fs)[0]
    result = asyncio.run(tool())
    assert result["error"] == "failed to parse likec4 output"
    assert result["raw_output"] == "not valid json"
    assert "details" in result


def test_validate_likec4_tool_factory_uses_custom_default_path(
    mock_likec4_in_path: None,
    fs: MemoryFileSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs.pipe_file("/foo.c4", b"specification { }")
    monkeypatch.setattr(
        "contractor.tools.likec4.subprocess.run",
        lambda *_a, **_kw: _proc(json.dumps({"valid": True, "errors": [], "stats": {}})),
    )

    tool = likec4_tools(fs=fs, default_path="/foo.c4")[0]
    assert asyncio.run(tool()) == {"result": []}
