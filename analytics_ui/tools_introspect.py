"""Static extraction of each agent's tool surface.

Agents wire their tools imperatively in ``build_<agent>`` factories: they call
tool-registry builders (``memory_tools``, ``ro_file_tools``, …), splat the
returned lists into ``tools = [default_tool, *fs_tools, …]``, and sometimes gate
or filter them (``with_graph_tools``, ``enable_vuln_reporting``, a
``[t for t in fs_tools if t.__name__ not in DISALLOWED]`` comprehension). There
is no declarative manifest, so — exactly like :mod:`analytics_ui.graph` for
pipelines — we recover the surface with :mod:`ast`, never importing the agent
runtime or instantiating a model.

The result per agent is an ordered list of tool *groups* (one per builder),
each carrying the resolved tool names + one-line docstrings, with conditional
tools flagged by the parameter that gates them.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_CONTRACTOR = Path(__file__).resolve().parents[1] / "contractor"
TOOLS_DIR = _CONTRACTOR / "tools"
AGENTS_DIR = _CONTRACTOR / "agents"
CALLBACKS_FILE = _CONTRACTOR / "callbacks" / "__init__.py"

# Friendly labels for the tool-registry builders.
GROUP_LABELS = {
    "memory_tools": "Memory",
    "ro_file_tools": "Filesystem · read-only",
    "rw_file_tools": "Filesystem · read/write",
    "code_tools": "Code search",
    "annotation_tools": "Trace annotations",
    "attach_graph_tools_if_local": "Call graph",
    "code_graph_tools": "Call graph",
    "http_tools": "HTTP",
    "vulnerability_report_tools": "Vulnerability reports",
    "verification_tools": "Verification",
    "openapi_tools": "OpenAPI editor",
    "openapi_linter_tools": "OpenAPI linter",
    "likec4_tools": "LikeC4",
    "task_tools": "Planner · subtasks",
}


def _doc1(node: ast.AST) -> str:
    d = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    if not d:
        return ""
    for line in d.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _flag_str(test: ast.AST) -> str:
    """A short readable label for an ``if`` test gating some tools."""
    if isinstance(test, ast.Name):
        return test.id
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return "not " + _flag_str(test.operand)
    if isinstance(test, ast.Attribute):
        return test.attr
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
        return test.left.id
    return "conditional"


def _kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _set_from_node(node: ast.AST, file_sets: dict[str, set[str]]) -> Optional[set[str]]:
    """Resolve a name set used in a filter: a Name (→ file_sets), an inline set,
    or ``frozenset({...})``."""
    if isinstance(node, ast.Name):
        return file_sets.get(node.id)
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set"):
        return _set_from_node(node.args[0], file_sets) if node.args else set()
    return None


def _collect_file_sets(tree: ast.Module) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for stmt in tree.body:
        target = value = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target, value = stmt.target.id, stmt.value
        if target:
            s = _set_from_node(value, out)
            if s is not None:
                out[target] = s
    return out


# ───────────────────────── builder index ─────────────────────────


@lru_cache(maxsize=1)
def _builder_index() -> dict[str, tuple[str, ast.AST]]:
    """Top-level function name → (file, FunctionDef) across tools + callbacks.

    Cached: agent/tool wiring rarely changes, and re-parsing the whole tool tree
    on every request would be wasteful. Prompts/tasks (the live-edit surface)
    are read uncached elsewhere.
    """
    out: dict[str, tuple[str, ast.AST]] = {}
    files = list(TOOLS_DIR.rglob("*.py"))
    if CALLBACKS_FILE.is_file():
        files.append(CALLBACKS_FILE)
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, (str(f), node))
    return out


def _own_stmts(stmts: list[ast.stmt]):
    """Yield statements in ``fn``'s own scope, recursing through control flow
    (if/try/for/while/with) but NOT into nested function/class bodies — so an
    inner tool closure's ``return x`` never masquerades as the builder's."""
    for s in stmts:
        yield s
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(s, attr, None)
            if isinstance(sub, list):
                yield from _own_stmts(sub)
        if isinstance(s, ast.Try):
            for h in s.handlers:
                yield from _own_stmts(h.body)


def _inner_defs(fn: ast.AST) -> dict[str, str]:
    """name → first docstring line for every tool closure defined in ``fn``'s
    own scope (the candidate tools), excluding nested helper defs."""
    out: dict[str, str] = {}
    for n in _own_stmts(getattr(fn, "body", [])):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, _doc1(n))
    return out


def _builder_exports(fn: ast.AST) -> dict[str, Any]:
    """Resolve which inner tools a registry builder returns.

    Returns ``{"alias": name}`` when the builder just forwards another builder
    (e.g. ``attach_graph_tools_if_local`` → ``code_graph_tools``), else
    ``{"tools": [{name, doc, gate}]}`` where ``gate`` is the builder-internal
    flag (``with_interaction_tools``/``use_skip``) when a tool is conditional.
    """
    inner = _inner_defs(fn)
    returns = [n for n in _own_stmts(getattr(fn, "body", []))
               if isinstance(n, ast.Return) and n.value is not None]

    members: list[tuple[str, Optional[str]]] = []
    target_var: Optional[str] = None
    ret_list: Optional[ast.List] = None
    for r in returns:
        if isinstance(r.value, ast.Name):
            target_var = r.value.id
        elif isinstance(r.value, (ast.List, ast.Tuple)):
            ret_list = r.value  # type: ignore[assignment]

    if target_var:
        def visit(stmts: list[ast.stmt], gate: Optional[str]) -> None:
            for s in stmts:
                assign_targets = (
                    s.targets if isinstance(s, ast.Assign)
                    else [s.target] if isinstance(s, ast.AnnAssign) else []
                )
                assign_value = getattr(s, "value", None) if assign_targets else None
                if (assign_value is not None
                        and any(isinstance(t, ast.Name) and t.id == target_var for t in assign_targets)
                        and isinstance(assign_value, (ast.List, ast.Tuple))):
                    for el in assign_value.elts:
                        if isinstance(el, ast.Name):
                            members.append((el.id, gate))
                elif (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                      and isinstance(s.value.func, ast.Attribute)
                      and isinstance(s.value.func.value, ast.Name)
                      and s.value.func.value.id == target_var):
                    meth, args = s.value.func.attr, s.value.args
                    if meth == "append" and args and isinstance(args[0], ast.Name):
                        members.append((args[0].id, gate))
                    elif meth == "extend" and args and isinstance(args[0], (ast.List, ast.Tuple)):
                        for el in args[0].elts:
                            if isinstance(el, ast.Name):
                                members.append((el.id, gate))
                elif isinstance(s, ast.If):
                    g = _flag_str(s.test)
                    visit(s.body, gate or g)
                    visit(s.orelse, gate)
        visit(fn.body, None)  # type: ignore[arg-type]
    elif ret_list is not None:
        for el in ret_list.elts:
            if isinstance(el, ast.Name):
                members.append((el.id, None))

    tools = [
        {"name": n, "doc": inner.get(n, ""), "gate": g}
        for n, g in members if n in inner
    ]
    if tools:
        return {"tools": tools}

    # No inner tools → maybe an alias that forwards another builder.
    idx = _builder_index()
    for s in _own_stmts(getattr(fn, "body", [])):
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in idx and n.func.id != getattr(fn, "name", None):
                return {"alias": n.func.id}
    return {"tools": []}


@lru_cache(maxsize=128)
def _resolve_builder(name: str) -> list[dict[str, Any]]:
    """Builder name → [{name, doc, gate}] (alias-resolved, depth-guarded)."""
    idx = _builder_index()
    seen: set[str] = set()
    while name in idx and name not in seen:
        seen.add(name)
        exports = _builder_exports(idx[name][1])
        if "alias" in exports:
            name = exports["alias"]
            continue
        return exports["tools"]
    return []


# ───────────────────────── per-agent resolution ─────────────────────────


def _find_build_fn(tree: ast.Module, agent_name: str) -> Optional[ast.AST]:
    candidates = [n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("build_")]
    exact = f"build_{agent_name}"
    for c in candidates:
        if c.name == exact:
            return c
    return candidates[0] if candidates else None


def _value_source(
    value: ast.AST, gate: Optional[str], varmap: dict[str, dict], file_sets: dict[str, set[str]]
) -> Optional[dict[str, Any]]:
    """Describe what a variable holds: a builder call, a gated builder, a
    filtered view of another var, or an alias of one."""
    idx = _builder_index()
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in idx:
        return {"kind": "builder", "builder": value.func.id, "gate": gate, "filters": []}
    if isinstance(value, ast.IfExp):  # `builder(...) if flag else []`
        body = _value_source(value.body, _flag_str(value.test), varmap, file_sets)
        if body:
            return body
    if isinstance(value, ast.ListComp) and len(value.generators) == 1:
        gen = value.generators[0]
        if isinstance(gen.iter, ast.Name) and gen.iter.id in varmap:
            base = dict(varmap[gen.iter.id])
            base["filters"] = list(base.get("filters", []))
            for cond in gen.ifs:
                if isinstance(cond, ast.Compare) and cond.ops:
                    op = cond.ops[0]
                    names = _set_from_node(cond.comparators[0], file_sets) if cond.comparators else None
                    if names is not None:
                        base["filters"].append({"op": "in" if isinstance(op, ast.In) else "notin", "names": names})
            if gate:
                base["gate"] = gate
            return base
    if isinstance(value, ast.Name) and value.id in varmap:
        src = dict(varmap[value.id])
        if gate:
            src["gate"] = gate
        return src
    return None


def _resolve_source(src: dict[str, Any]) -> list[dict[str, Any]]:
    """A var descriptor → concrete [{name, doc, conditional, flag}]."""
    if src.get("kind") != "builder":
        return []
    tools = _resolve_builder(src["builder"])
    for filt in src.get("filters", []):
        if filt["op"] == "in":
            tools = [t for t in tools if t["name"] in filt["names"]]
        else:
            tools = [t for t in tools if t["name"] not in filt["names"]]
    gate = src.get("gate")
    out = []
    for t in tools:
        flag = gate or t.get("gate")
        out.append({"name": t["name"], "doc": t["doc"], "conditional": bool(flag), "flag": flag})
    return out


def agent_tools(agent_name: str) -> Optional[dict[str, Any]]:
    """Resolve the tool surface of one agent, or None if it can't be parsed."""
    path = AGENTS_DIR / agent_name / "agent.py"
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    build = _find_build_fn(tree, agent_name)
    if build is None:
        return None

    file_sets = _collect_file_sets(tree)
    varmap: dict[str, dict] = {}
    assembly: list[dict[str, Any]] = []  # ordered items feeding `tools`

    def walk(stmts: list[ast.stmt], gate: Optional[str]) -> None:
        for s in stmts:
            if isinstance(s, ast.If):
                g = _flag_str(s.test)
                walk(s.body, gate or g)
                walk(s.orelse, gate)
                continue
            tgt_node = (
                s.targets[0] if isinstance(s, ast.Assign) and len(s.targets) == 1
                else s.target if isinstance(s, ast.AnnAssign) else None
            )
            if isinstance(tgt_node, ast.Name) and getattr(s, "value", None) is not None:
                tgt = tgt_node.id
                if tgt == "tools" and isinstance(s.value, (ast.List, ast.Tuple)):
                    for el in s.value.elts:
                        if isinstance(el, ast.Starred) and isinstance(el.value, ast.Name):
                            assembly.append({"var": el.value.id, "gate": gate})
                        elif isinstance(el, ast.Name):
                            assembly.append({"bare": el.id, "gate": gate})
                    continue
                src = _value_source(s.value, gate, varmap, file_sets)
                if src is not None:
                    varmap[tgt] = src
            elif (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                  and isinstance(s.value.func, ast.Attribute)
                  and isinstance(s.value.func.value, ast.Name)
                  and s.value.func.value.id == "tools"
                  and s.value.func.attr in ("extend", "append") and s.value.args):
                arg = s.value.args[0]
                if isinstance(arg, ast.Name):
                    assembly.append({"var": arg.id, "gate": gate})

    walk(build.body, None)  # type: ignore[arg-type]

    # default_tool's docstring (it lives in callbacks, indexed as a builder fn).
    idx = _builder_index()
    default_doc = _doc1(idx["default_tool"][1]) if "default_tool" in idx else ""

    groups: list[dict[str, Any]] = []
    total = 0
    for item in assembly:
        gate = item.get("gate")
        if "bare" in item:
            name = item["bare"]
            doc = default_doc if name == "default_tool" else ""
            groups.append({
                "label": "Core" if name == "default_tool" else name,
                "gate": gate,
                "tools": [{"name": name, "doc": doc, "conditional": bool(gate), "flag": gate}],
            })
            total += 1
            continue
        src = varmap.get(item["var"])
        if not src:
            continue
        if gate and not src.get("gate"):
            src = {**src, "gate": gate}
        tools = _resolve_source(src)
        if not tools:
            continue
        groups.append({
            "label": GROUP_LABELS.get(src["builder"], src["builder"].replace("_tools", "").replace("_", " ").title()),
            "builder": src["builder"],
            "gate": src.get("gate"),
            "tools": tools,
        })
        total += len(tools)

    return {"count": total, "groups": groups}
