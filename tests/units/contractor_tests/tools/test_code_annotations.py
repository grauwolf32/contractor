"""Unit tests for ``contractor.tools.code.annotations``.

Drives the three annotation tools against tiny on-disk fixtures wrapped
in a ``MemoryOverlayFileSystem`` (the same setup the trace pipelines
use), and asserts both the on-disk text and the structured tool
response.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.fs import RootedLocalFileSystem
from contractor.tools.code import annotation_tools
from contractor.tools.fs import MemoryOverlayFileSystem


def _setup(tmp_path: Path, files: dict[str, str]):
    for rel, content in files.items():
        target = tmp_path / rel.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    base = RootedLocalFileSystem(str(tmp_path))
    overlay = MemoryOverlayFileSystem(base)
    by_name = {t.__name__: t for t in annotation_tools(overlay)}
    return overlay, by_name


def test_annotate_trace_inserts_above_function(tmp_path: Path) -> None:
    overlay, tools = _setup(
        tmp_path,
        {
            "/app.py": (
                "def login(token: str) -> bool:\n"
                "    return token == 'ok'\n"
            )
        },
    )
    r = tools["annotate_trace"](
        "/app.py", "login", target="login-flow", args="token:tainted"
    )
    assert r["kind"] == "trace"
    assert r["annotation_line"] == 1
    text = overlay.read_text("/app.py", encoding="utf-8")
    assert "# @trace target=login-flow args=token:tainted\n" in text
    assert "def login" in text


def test_annotate_preserves_indentation_for_methods(tmp_path: Path) -> None:
    overlay, tools = _setup(
        tmp_path,
        {
            "/svc.py": (
                "class Auth:\n"
                "    def verify(self, token):\n"
                "        return token == 'ok'\n"
            )
        },
    )
    r = tools["annotate_trace"]("/svc.py", "verify", target="auth")
    assert "error" not in r, r
    text = overlay.read_text("/svc.py", encoding="utf-8")
    # Annotation indented to match the method def.
    assert "    # @trace target=auth\n    def verify" in text


def test_annotate_validate_requires_arg_and_kind(tmp_path: Path) -> None:
    _, tools = _setup(
        tmp_path, {"/app.py": "def x():\n    return 1\n"}
    )
    assert tools["annotate_validate"]("/app.py", "x", arg="", kind="regex").get(
        "error"
    )
    assert tools["annotate_validate"]("/app.py", "x", arg="t", kind="").get(
        "error"
    )


def test_annotate_sink_writes_kind_and_arg(tmp_path: Path) -> None:
    overlay, tools = _setup(
        tmp_path, {"/q.py": "def raw_query(q):\n    return q\n"}
    )
    r = tools["annotate_sink"]("/q.py", "raw_query", kind="sql", arg="q")
    assert "error" not in r, r
    text = overlay.read_text("/q.py", encoding="utf-8")
    assert "# @sink kind=sql arg=q\n" in text


def test_annotate_refuses_undefined_function(tmp_path: Path) -> None:
    """Annotation must not land on a call site or an import line — the
    locator requires a real tree-sitter definition in the named file.
    """
    _, tools = _setup(
        tmp_path,
        {
            "/app.py": (
                "from auth import authenticate\n"
                "\n"
                "def login(token):\n"
                "    return authenticate(token)\n"
            )
        },
    )
    r = tools["annotate_sink"]("/app.py", "authenticate", kind="auth")
    assert "error" in r, r
    assert "not defined" in r["error"]


def test_annotate_refuses_duplicate(tmp_path: Path) -> None:
    overlay, tools = _setup(
        tmp_path, {"/app.py": "def login(t):\n    return t\n"}
    )
    tools["annotate_trace"]("/app.py", "login", target="x")
    r2 = tools["annotate_trace"]("/app.py", "login", target="x")
    assert "error" in r2
    assert "already has a @trace" in r2["error"]


def test_annotate_bad_arg_state_rejected(tmp_path: Path) -> None:
    _, tools = _setup(
        tmp_path, {"/app.py": "def f(x):\n    return x\n"}
    )
    r = tools["annotate_trace"](
        "/app.py", "f", target="t", args="x:fancy"
    )
    assert "error" in r
    assert "fancy" in r["error"]


def test_annotate_uses_double_slash_comment_for_java(tmp_path: Path) -> None:
    overlay, tools = _setup(
        tmp_path,
        {
            "/Main.java": (
                "public class Main {\n"
                "    public void run() {\n"
                "        return;\n"
                "    }\n"
                "}\n"
            )
        },
    )
    r = tools["annotate_trace"]("/Main.java", "run", target="java-entry")
    assert "error" not in r, r
    text = overlay.read_text("/Main.java", encoding="utf-8")
    assert "    // @trace target=java-entry\n    public void run()" in text
