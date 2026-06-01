"""Static extraction of a workflow's task pipeline.

Workflows assemble their pipelines imperatively in ``workflow.py`` (build a
``TaskRunner``, queue ``add_task`` calls, sometimes invoke sub-workflows), so
there is no declarative DAG to read. This module recovers an approximate graph
by parsing the module with :mod:`ast`:

* ``partial(build_<x>_agent, ...)`` assignments map a worker-builder variable
  to its agent (``build_swe_agent`` → agent dir ``swe_agent``).
* ``runner.add_task(name=..., worker_builder=..., artifacts=[...], skills=[...])``
  calls become nodes.
* Edges are inferred from ``artifacts=["<task>/result", ...]``: the producer is
  the node whose task name matches the artifact's first path segment.
* ``<Name>Workflow(...)`` constructor calls become sub-workflow nodes.

Because this is static analysis, conditional branches and the exact ordering of
sub-workflow stages are approximate — flagged in ``warnings`` and per-node so
the UI can show the caveat honestly rather than imply false precision.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional


def _const(node: ast.AST) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def _str_list(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out = []
    for el in node.elts:
        v = _const(el)
        if isinstance(v, str):
            out.append(v)
    return out


def _kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _agent_from_builder(func: ast.AST) -> Optional[str]:
    """`build_swe_agent` / `build_oas_linter_agent` → `swe_agent` / `oas_linter_agent`."""
    name = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    if name and name.startswith("build_"):
        return name[len("build_") :]
    return None


def _resolve_partial(call: ast.Call) -> Optional[str]:
    """Agent name for a ``partial(build_x_agent, ...)`` call, else None."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "partial"):
        return None
    if not call.args:
        return None
    return _agent_from_builder(call.args[0])


class _Extractor(ast.NodeVisitor):
    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.partial_agents: dict[str, str] = {}  # var name -> agent
        self.class_consts: dict[str, Any] = {}  # class attr -> constant
        self.tasks: list[dict[str, Any]] = []
        self.sub_workflows: list[str] = []
        self._seen_sub: set[str] = set()

    # -- pass 1: collect partial() assignments and class-level constants --

    def collect_bindings(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                agent = _resolve_partial(node.value)
                if agent:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.partial_agents[tgt.id] = agent
            # class-body `namespace: str = "vuln-scan"` / `namespace = "x"`
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                v = _const(node.value) if node.value is not None else None
                if isinstance(v, str):
                    self.class_consts[node.target.id] = v
            if isinstance(node, ast.Assign):
                v = _const(node.value)
                if isinstance(v, str):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.class_consts.setdefault(tgt.id, v)

    # -- pass 2: walk for add_task / sub-workflow calls, tracking branches --

    def walk(self, node: ast.AST, in_cond: bool) -> None:
        if isinstance(node, ast.If):
            for n in node.body:
                self.walk(n, True)
            for n in node.orelse:
                self.walk(n, True)
            return
        if isinstance(node, (ast.Try,)):
            for n in node.body:
                self.walk(n, in_cond)
            for h in node.handlers:
                for n in h.body:
                    self.walk(n, True)
            for n in node.orelse + node.finalbody:
                self.walk(n, in_cond)
            return
        if isinstance(node, ast.Call):
            self._handle_call(node, in_cond)
        for child in ast.iter_child_nodes(node):
            self.walk(child, in_cond)

    def _resolve_namespace(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        v = _const(node)
        if isinstance(v, str):
            return v
        # self.namespace -> class constant
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return self.class_consts.get(node.attr)
        return None

    def _resolve_agent(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.partial_agents.get(node.id)
        if isinstance(node, ast.Call):  # inline partial(...)
            return _resolve_partial(node)
        return None

    def _handle_call(self, call: ast.Call, in_cond: bool) -> None:
        func = call.func
        # add_task
        if isinstance(func, ast.Attribute) and func.attr == "add_task":
            kw = _kwargs(call)
            name = _const(kw.get("name"))
            if name is None and call.args:
                name = _const(call.args[0])
            if not isinstance(name, str):
                return
            ref = _const(kw.get("ref"))
            self.tasks.append(
                {
                    "id": ref if isinstance(ref, str) else f"{name}#{len(self.tasks)}",
                    "task": name,
                    "ref": ref if isinstance(ref, str) else None,
                    "agent": self._resolve_agent(kw.get("worker_builder")),
                    "artifacts": _str_list(kw.get("artifacts"))
                    if "artifacts" in kw
                    else [],
                    "skills": _str_list(kw.get("skills")) if "skills" in kw else [],
                    "namespace": self._resolve_namespace(kw.get("namespace")),
                    "conditional": in_cond,
                }
            )
            return
        # sub-workflow constructor: SomethingWorkflow(...)
        if isinstance(func, ast.Name) and func.id.endswith("Workflow"):
            if func.id != self.class_name and func.id not in self._seen_sub:
                self._seen_sub.add(func.id)
                self.sub_workflows.append(func.id)


def _class_node(tree: ast.AST, class_name: Optional[str]) -> Optional[ast.ClassDef]:
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if class_name:
        for c in classes:
            if c.name == class_name:
                return c
    # fall back to the first *Workflow subclass
    for c in classes:
        if c.name.endswith("Workflow"):
            return c
    return classes[0] if classes else None


def extract_graph(
    module_file: str,
    class_name: Optional[str] = None,
    subworkflow_keys: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build a node/edge graph for one workflow module.

    ``subworkflow_keys`` maps a workflow class name to its registry key so
    sub-workflow nodes can deep-link; unknown classes still render, just
    without a link.
    """
    text = Path(module_file).read_text(encoding="utf-8")
    tree = ast.parse(text)
    cls = _class_node(tree, class_name)
    cname = cls.name if cls else (class_name or "")

    ex = _Extractor(cname)
    ex.collect_bindings(tree)
    ex.walk(cls if cls else tree, False)

    keys = subworkflow_keys or {}

    nodes: list[dict[str, Any]] = []
    for i, t in enumerate(ex.tasks):
        nodes.append(
            {
                "id": t["id"],
                "kind": "task",
                "task": t["task"],
                "agent": t["agent"],
                "skills": t["skills"],
                "namespace": t["namespace"],
                "artifacts": t["artifacts"],
                "conditional": t["conditional"],
                "order": i,
            }
        )

    # Map an artifact's producing task by name *or* namespace.
    by_task: dict[str, str] = {}
    by_ns: dict[str, str] = {}
    for n in nodes:
        by_task.setdefault(n["task"], n["id"])
        if n["namespace"]:
            by_ns.setdefault(n["namespace"], n["id"])

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for n in nodes:
        for art in n["artifacts"]:
            producer_key = art.split("/", 1)[0]
            src = by_task.get(producer_key) or by_ns.get(producer_key)
            if src and src != n["id"]:
                key = (src, n["id"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({"source": src, "target": n["id"], "label": art})
            else:
                n.setdefault("external_inputs", []).append(art)

    # Sub-workflow nodes appended as a downstream stage strip.
    sub_nodes = []
    for cls_name in ex.sub_workflows:
        sub_nodes.append(
            {
                "id": f"wf:{cls_name}",
                "kind": "subworkflow",
                "class_name": cls_name,
                "workflow_key": keys.get(cls_name),
                "order": len(nodes) + len(sub_nodes),
            }
        )

    warnings: list[str] = []
    if any(n["conditional"] for n in nodes):
        warnings.append(
            "Some tasks run inside conditional branches (skip-if-artifact-exists, "
            "env gates); they may not execute on every run."
        )
    if sub_nodes:
        warnings.append(
            "This workflow delegates to sub-workflows; their internal stages and "
            "exact ordering are shown as a single node each."
        )
    if not nodes and not sub_nodes:
        warnings.append(
            "No add_task calls found by static analysis — this workflow may drive "
            "agents through a different runner (e.g. the router's single-agent path)."
        )

    return {
        "class_name": cname,
        "module": str(module_file),
        "doc": ast.get_docstring(tree) or (ast.get_docstring(cls) if cls else "") or "",
        "nodes": nodes + sub_nodes,
        "edges": edges,
        "sub_workflows": [n["class_name"] for n in sub_nodes],
        "warnings": warnings,
    }
