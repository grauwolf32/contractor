"""Workflow registry inspection for the UI.

Bridges the imperative workflow registry (``get_workflows()``) to the static
graph extractor: resolves each registry key to its module file + class name,
builds the class→key reverse map for sub-workflow deep-links, and merges the
sibling ``config.yaml`` budgets onto graph nodes.

Importing the registry pulls in the agent runtime, which is heavier than the
pure file readers — so this is the one module that touches it, and it degrades
gracefully (returning a best-effort list) if an import fails.
"""
from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import Any

from analytics_ui import graph, reader

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    """{key: WorkflowClass}. Cached; harmless to call repeatedly."""
    from contractor.workflows import get_workflows

    return dict(get_workflows())


def _class_to_key() -> dict[str, str]:
    return {cls.__name__: key for key, cls in _registry().items()}


def _module_file(cls: Any) -> str | None:
    try:
        return inspect.getsourcefile(cls)
    except (TypeError, OSError):
        return None


def list_workflows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, cls in _registry().items():
        doc = (inspect.getdoc(cls) or "").strip()
        summary = doc.splitlines()[0] if doc else ""
        out.append(
            {
                "key": key,
                "class_name": cls.__name__,
                "summary": summary,
            }
        )
    out.sort(key=lambda w: w["key"])
    return out


def get_workflow(key: str) -> dict[str, Any] | None:
    cls = _registry().get(key)
    if cls is None:
        return None
    module_file = _module_file(cls)
    if module_file is None:
        return {
            "key": key,
            "class_name": cls.__name__,
            "doc": inspect.getdoc(cls) or "",
            "nodes": [],
            "edges": [],
            "sub_workflows": [],
            "warnings": ["Could not locate the workflow source file."],
            "budgets": {},
        }

    g = graph.extract_graph(
        module_file, class_name=cls.__name__, subworkflow_keys=_class_to_key()
    )

    cfg = reader.workflow_config(module_file)
    task_budgets = cfg.get("tasks") or {}
    for node in g["nodes"]:
        if node.get("kind") == "task":
            node["budget"] = task_budgets.get(node["task"])

    g["key"] = key
    g["budgets"] = cfg.get("budgets") or {}
    g["agents_config"] = cfg.get("agents") or {}
    return g


# ───────────────────── cross-references ─────────────────────


def _all_graphs() -> dict[str, dict[str, Any]]:
    # Not cached: AST-parsing ~14 modules is cheap and keeps cross-refs live
    # against edits to workflow.py without a server restart.
    out: dict[str, dict[str, Any]] = {}
    for key in _registry():
        try:
            g = get_workflow(key)
        except Exception:  # pragma: no cover - never let one bad module break the UI
            logger.exception("failed to extract graph for workflow %s", key)
            continue
        if g:
            out[key] = g
    return out


def crossrefs() -> dict[str, Any]:
    """Reverse indexes: which workflows use each agent / each skill, and which
    tasks declare each skill. Recomputed lazily; cached for the process."""
    agent_to_workflows: dict[str, set[str]] = {}
    skill_to_workflows: dict[str, set[str]] = {}
    task_to_workflows: dict[str, set[str]] = {}

    for key, g in _all_graphs().items():
        for node in g.get("nodes", []):
            if node.get("agent"):
                agent_to_workflows.setdefault(node["agent"], set()).add(key)
            if node.get("kind") == "task":
                task_to_workflows.setdefault(node["task"], set()).add(key)
            for s in node.get("skills", []) or []:
                skill_to_workflows.setdefault(s, set()).add(key)

    skill_to_tasks: dict[str, set[str]] = {}
    for t in reader.list_tasks():
        for s in t.skills:
            skill_to_tasks.setdefault(s, set()).add(t.name)

    def _sorted(d: dict[str, set[str]]) -> dict[str, list[str]]:
        return {k: sorted(v) for k, v in d.items()}

    return {
        "agent_to_workflows": _sorted(agent_to_workflows),
        "skill_to_workflows": _sorted(skill_to_workflows),
        "task_to_workflows": _sorted(task_to_workflows),
        "skill_to_tasks": _sorted(skill_to_tasks),
    }


def invalidate_caches() -> None:
    """Drop the memoized registry so the next request re-imports it."""
    _registry.cache_clear()
