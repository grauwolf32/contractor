"""Unit tests for ``contractor.tools.code.graph``.

Builds a real trailmark graph over a tiny on-disk fixture (cheap: <100 ms
for a handful of files) and confirms each tool returns the expected
shape. The UTF-8 safety patch is also exercised by including a file with
a non-UTF8 byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contractor.tools.code.graph import code_graph_tools, strip_prefix_resolver


@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "from auth import authenticate\n"
        "from fastapi import FastAPI\n"
        "\n"
        "app = FastAPI()\n"
        "\n"
        "@app.get('/login')\n"
        "def login(token: str):\n"
        "    return authenticate(token)\n",
        encoding="utf-8",
    )
    (tmp_path / "auth.py").write_text(
        "def authenticate(token: str) -> bool:\n"
        "    return _verify(token)\n"
        "\n"
        "def _verify(token: str) -> bool:\n"
        "    return token == 'secret'\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def project_with_bad_utf8(tmp_path: Path) -> Path:
    (tmp_path / "good.py").write_text(
        "def hello():\n    return 'world'\n", encoding="utf-8"
    )
    (tmp_path / "bad.py").write_bytes(
        b"def broken():\n    return '\xf0\x28\x8c\xbc'\n"
    )
    return tmp_path


def _by_name(tools):
    return {t.__name__: t for t in tools}


def test_graph_summary_reports_nodes_and_edges(tiny_project: Path) -> None:
    tools = _by_name(code_graph_tools(tiny_project))
    summary = tools["graph_summary"]()["result"]
    assert summary["total_nodes"] >= 3  # app, login, authenticate, _verify
    assert summary["call_edges"] >= 2


def test_find_symbol_resolves_bare_name(tiny_project: Path) -> None:
    tools = _by_name(code_graph_tools(tiny_project))
    result = tools["find_symbol"]("authenticate")
    assert result["total_items"] >= 1
    ids = {row["id"] for row in result["result"]}
    assert any(rid.endswith(":authenticate") for rid in ids)


def test_find_callees_includes_inferred_edges(tiny_project: Path) -> None:
    tools = _by_name(code_graph_tools(tiny_project))
    callees = tools["find_callees"]("login")
    targets = {row["id"] for row in callees["result"]}
    # `authenticate(token)` is a direct call — must surface (resolved or
    # symbolic). Trailmark resolves cross-file calls when unambiguous.
    assert any("authenticate" in t for t in targets), targets


def test_find_callers_finds_login(tiny_project: Path) -> None:
    tools = _by_name(code_graph_tools(tiny_project))
    callers = tools["find_callers"]("authenticate")
    caller_names = {row["name"] for row in callers["result"]}
    assert "login" in caller_names


def test_attack_surface_picks_up_fastapi_route(tiny_project: Path) -> None:
    tools = _by_name(code_graph_tools(tiny_project))
    asurf = tools["attack_surface"]()
    assert asurf["total_items"] >= 1
    node_ids = {row["node_id"] for row in asurf["result"]}
    assert any(nid.endswith(":login") for nid in node_ids)


def test_utf8_safety_skips_bad_file(project_with_bad_utf8: Path) -> None:
    # Must not raise UnicodeDecodeError; bad file is silently skipped and
    # the good file is still parsed.
    tools = _by_name(code_graph_tools(project_with_bad_utf8))
    summary = tools["graph_summary"]()["result"]
    assert summary["total_nodes"] >= 1
    result = tools["find_symbol"]("hello")
    assert result["total_items"] >= 1


def test_paths_default_to_host_absolute(tiny_project: Path) -> None:
    """Without a ``path_resolver``, tools surface trailmark's host-FS
    paths untouched. ``code/graph`` does not bake in any specific FS
    convention; callers opt into translation explicitly.
    """
    tools = _by_name(code_graph_tools(tiny_project))
    found = tools["find_symbol"]("login")
    assert found["result"], found
    file = found["result"][0]["file"]
    assert file is not None
    assert str(tiny_project) in file, file


def test_path_resolver_rewrites_to_virtual_root(tiny_project: Path) -> None:
    """With ``strip_prefix_resolver`` injected, tool results expose
    ``/relative/path.py`` form so they compose with overlay-FS file
    tools rooted at the same project dir. Regression: this path used
    to live inside ``code/graph`` itself, coupling the module to the
    ``RootedLocalFileSystem`` convention.
    """
    tools = _by_name(
        code_graph_tools(
            tiny_project,
            path_resolver=strip_prefix_resolver(str(tiny_project)),
        )
    )
    found = tools["find_symbol"]("login")
    assert found["result"], found
    file = found["result"][0]["file"]
    assert file == "/app.py", file
    # Same translator must apply to caller / callee rows too.
    callers = tools["find_callers"]("authenticate")
    assert callers["result"]
    assert callers["result"][0]["file"] == "/app.py", callers


def test_path_resolver_returning_none_keeps_original(tiny_project: Path) -> None:
    """If the resolver returns None for a path, the original is kept —
    lets callers do partial mapping (e.g. only translate paths inside
    a sandboxed root and pass external library paths through).
    """
    def _never_match(_: str):
        return None

    tools = _by_name(
        code_graph_tools(tiny_project, path_resolver=_never_match)
    )
    found = tools["find_symbol"]("login")
    file = found["result"][0]["file"]
    assert str(tiny_project) in file, file


def test_unresolved_callee_returns_symbolic_row(tmp_path: Path) -> None:
    # A call to an attribute on a runtime object (`self.svc.do()`) is
    # left INFERRED with a symbolic target; the tool must still surface
    # it with kind="unresolved" so the agent can chase the name.
    (tmp_path / "svc.py").write_text(
        "class Service:\n"
        "    def __init__(self, dep):\n"
        "        self.dep = dep\n"
        "    def handle(self):\n"
        "        return self.dep.do_thing()\n",
        encoding="utf-8",
    )
    tools = _by_name(code_graph_tools(tmp_path))
    callees = tools["find_callees"]("Service.handle")
    rows = callees["result"]
    assert rows, callees
    assert any(row.get("kind") == "unresolved" for row in rows)
