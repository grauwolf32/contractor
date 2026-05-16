"""Call-graph tools backed by trailmark.

Exposes the trailmark ``QueryEngine`` as a small set of ADK tool functions
that an LLM agent can use to navigate a code base by structure instead of
by grep:

- ``graph_summary``           — node/edge/entrypoint counts
- ``find_symbol``             — resolve a name to graph nodes
- ``find_callers``            — direct callers of a symbol
- ``find_callees``            — direct callees of a symbol
- ``paths_between``           — simple call paths src→dst
- ``entrypoint_paths_to``     — paths from any entrypoint to a sink
- ``attack_surface``          — list of detected entrypoints
- ``complexity_hotspots``     — high cyclomatic-complexity nodes
- ``functions_that_raise``    — functions whose parser-detected exception
                                  list contains ``exc``

The engine is built lazily on the first tool call and cached for the
lifetime of the tool factory. Trailmark crashes on files with non-UTF8
bytes (real-world C/C++ codebases hit this); we monkey-patch the inner
parser dispatcher to skip such files gracefully and log them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "auto"
DEFAULT_MAX_RESULTS = 200
DEFAULT_MAX_PATHS = 25
_MAX_PATH_DEPTH = 30


def _install_utf8_safety() -> None:
    """Patch ``trailmark.parsers._common.parse_directory`` to tolerate
    non-UTF8 source files.

    Idempotent — safe to call multiple times.
    """
    from trailmark.models.graph import CodeGraph
    from trailmark.parsers import _common as tm_common

    if getattr(tm_common.parse_directory, "_utf8_safe", False):
        return

    _orig = tm_common.parse_directory

    def _safe(parse_file_fn, language, dir_path, extensions):  # type: ignore[no-untyped-def]
        skipped: list[str] = []

        def _wrapped(p: str):
            try:
                return parse_file_fn(p)
            except UnicodeDecodeError as exc:
                logger.warning(
                    "trailmark: skipping non-UTF8 file %s (%s)", p, exc
                )
                skipped.append(str(p))
                return CodeGraph(language=language, root_path=str(p))

        graph = _orig(_wrapped, language, dir_path, extensions)
        if skipped:
            logger.info(
                "trailmark: skipped %d non-UTF8 file(s) under %s",
                len(skipped),
                dir_path,
            )
        return graph

    _safe._utf8_safe = True  # type: ignore[attr-defined]
    tm_common.parse_directory = _safe


def _to_virtual_path(host_path: Optional[str], host_root: str) -> Optional[str]:
    """Rewrite a host-FS absolute path into the virtual form the agent
    sees through a ``RootedLocalFileSystem``-sandboxed overlay (i.e.
    ``/relative/segment/file.py``).

    Returns the input unchanged for paths outside the configured root
    (so external library symbols stay unambiguous) or for ``None``.
    """
    if host_path is None:
        return None
    import os

    norm = os.path.normpath(host_path)
    root = os.path.normpath(host_root)
    if norm == root:
        return "/"
    sep = os.sep
    if norm.startswith(root + sep):
        rel = norm[len(root) + 1 :]
        return "/" + rel.replace(os.sep, "/")
    # Fallback for paths that don't share the sandbox prefix — keep them
    # untouched but log once so we notice if trailmark resolves something
    # outside the project root.
    return host_path


def _slim_unit(
    unit: dict[str, Any],
    host_root: Optional[str] = None,
) -> dict[str, Any]:
    """Strip a trailmark CodeUnit dict down to fields useful to an agent.

    Drops parameters/branches/exception_types to keep token cost low; the
    agent can read the file directly with ``read_file`` if it needs the
    body. When ``host_root`` is provided, the ``file`` field is rewritten
    to the virtual form (``/relative/path.py``) so it composes with the
    overlay-FS file-mutation tools.
    """
    loc = unit.get("location") or {}
    file_path = loc.get("file_path")
    if host_root is not None:
        file_path = _to_virtual_path(file_path, host_root)
    return {
        "id": unit.get("id"),
        "name": unit.get("name"),
        "kind": unit.get("kind"),
        "file": file_path,
        "start_line": loc.get("start_line"),
        "end_line": loc.get("end_line"),
    }


@dataclass
class GraphEngineHolder:
    """Lazily builds and caches a single trailmark QueryEngine."""

    root: Path
    language: str = DEFAULT_LANGUAGE
    _engine: Optional[Any] = field(default=None, init=False, repr=False)

    def engine(self):  # type: ignore[no-untyped-def]
        if self._engine is None:
            _install_utf8_safety()
            from trailmark.query.api import QueryEngine

            logger.info(
                "trailmark: building code graph (root=%s, language=%s)",
                self.root,
                self.language,
            )
            self._engine = QueryEngine.from_directory(
                str(self.root), language=self.language
            )
            logger.info(
                "trailmark: graph ready (%s)",
                self._engine.summary(),
            )
        return self._engine

    def graph(self):  # type: ignore[no-untyped-def]
        # noqa: SLF001 — accessing private attr is the only path right now.
        return self.engine()._store._graph


def code_graph_tools(
    root: str | Path,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> list:
    """Build the code-graph tool set for an LLM agent.

    ``root`` must be a host-filesystem path (trailmark uses tree-sitter
    against real files; in-memory overlays are not supported).
    """
    holder = GraphEngineHolder(root=Path(root), language=language)
    host_root = str(holder.root)

    def graph_summary() -> dict[str, Any]:
        """Return node/edge/entrypoint counts + the list of imported deps.

        Cheap (microseconds) — call this once at the start of a trace to
        gauge graph size and confirm the engine built successfully.
        """
        return holder.engine().summary()

    def find_symbol(symbol: str) -> dict[str, Any]:
        """Resolve a symbol name to one or more graph nodes.

        Matches by ``module:Class.method`` id, bare name (last segment),
        or case-insensitive equality. Returns up to 50 candidates with
        location info; use the ``id`` value with other graph tools for an
        unambiguous lookup.
        """
        graph = holder.graph()
        sym = str(symbol)
        bare = sym.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
        matches: list[dict[str, Any]] = []
        for node_id, unit in graph.nodes.items():
            if (
                node_id == sym
                or unit.name == sym
                or unit.name == bare
                or unit.name.lower() == bare.lower()
            ):
                from dataclasses import asdict

                d = asdict(unit)
                d["kind"] = unit.kind.value
                matches.append(_slim_unit(d, host_root=host_root))
            if len(matches) >= 50:
                break
        return {
            "result": matches,
            "total_items": len(matches),
            "kind": "find_symbol",
        }

    def _resolve_id(symbol: str) -> Optional[str]:
        graph = holder.graph()
        sym = str(symbol)
        if sym in graph.nodes:
            return sym
        bare = sym.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
        for node_id, unit in graph.nodes.items():
            if unit.name == bare or unit.name == sym:
                return node_id
        return None

    def find_callers(
        symbol: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Direct callers of ``symbol``.

        Returns every node that has a call edge whose target matches
        ``symbol``. Trailmark stores both ``certain`` and ``inferred``
        edges; both are included so the agent can decide whether to
        follow uncertain ones. Use ``find_symbol`` first to disambiguate.
        """
        node_id = _resolve_id(symbol)
        graph = holder.graph()
        if node_id is None:
            return {"result": [], "total_items": 0, "kind": "callers"}
        unit = graph.nodes[node_id]
        bare = unit.name
        rows: list[dict[str, Any]] = []
        from dataclasses import asdict

        for edge in graph.edges:
            if edge.kind.value != "calls":
                continue
            tgt = edge.target_id
            if tgt == node_id or tgt.endswith("." + bare) or tgt.endswith(":" + bare):
                src_unit = graph.nodes.get(edge.source_id)
                if src_unit is None:
                    continue
                d = asdict(src_unit)
                d["kind"] = src_unit.kind.value
                slim = _slim_unit(d, host_root=host_root)
                slim["edge_confidence"] = edge.confidence.value
                slim["edge_target"] = tgt
                rows.append(slim)
                if len(rows) >= max_results:
                    break
        return {
            "result": rows,
            "total_items": len(rows),
            "kind": "callers",
        }

    def find_callees(
        symbol: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        """Direct callees of ``symbol`` (call edges leaving its body).

        Returns the raw call targets, including ``inferred`` and
        ``uncertain`` edges whose target may be a virtual name (e.g.
        ``svc.authenticate``) rather than a resolved graph node. When the
        target resolves to a known node, location info is attached; when
        not, only the symbolic name is returned. Use ``find_symbol`` to
        chase virtual names.
        """
        node_id = _resolve_id(symbol)
        graph = holder.graph()
        if node_id is None:
            return {"result": [], "total_items": 0, "kind": "callees"}
        rows: list[dict[str, Any]] = []
        from dataclasses import asdict

        for edge in graph.edges:
            if edge.source_id != node_id or edge.kind.value != "calls":
                continue
            target_unit = graph.nodes.get(edge.target_id)
            if target_unit is not None:
                d = asdict(target_unit)
                d["kind"] = target_unit.kind.value
                row = _slim_unit(d, host_root=host_root)
            else:
                row = {
                    "id": edge.target_id,
                    "name": edge.target_id.rsplit(":", 1)[-1],
                    "kind": "unresolved",
                    "file": None,
                    "start_line": None,
                    "end_line": None,
                }
            row["edge_confidence"] = edge.confidence.value
            rows.append(row)
            if len(rows) >= max_results:
                break
        return {
            "result": rows,
            "total_items": len(rows),
            "kind": "callees",
        }

    def paths_between(
        src: str,
        dst: str,
        max_paths: int = DEFAULT_MAX_PATHS,
    ) -> dict[str, Any]:
        """All simple call paths from ``src`` to ``dst``.

        Each path is a list of node ids ordered from caller to callee.
        Empty result means no static call path was found.
        """
        engine = holder.engine()
        paths = engine.paths_between(str(src), str(dst)) or []
        return {
            "result": [list(p) for p in paths[:max_paths]],
            "total_items": len(paths),
            "kind": "paths_between",
        }

    def entrypoint_paths_to(
        symbol: str,
        max_paths: int = DEFAULT_MAX_PATHS,
        max_depth: int = _MAX_PATH_DEPTH,
    ) -> dict[str, Any]:
        """Paths from any detected entrypoint to ``symbol``.

        Canonical attack-surface query: given a sensitive sink, what
        externally-triggered call paths can reach it? Empty result means
        the sink is unreachable from a known entrypoint (or trailmark
        could not detect one for this codebase / language).
        """
        engine = holder.engine()
        paths = (
            engine.entrypoint_paths_to(str(symbol), max_depth=max_depth)
            or []
        )
        return {
            "result": [list(p) for p in paths[:max_paths]],
            "total_items": len(paths),
            "kind": "entrypoint_paths_to",
        }

    def attack_surface() -> dict[str, Any]:
        """List detected entrypoints with trust/asset metadata.

        Backed by trailmark's framework-aware entrypoint scanner
        (FastAPI/Flask/NestJS/Spring/ASP.NET/actix/etc.). Returns
        ``node_id``, ``kind`` (user_input/api/database/file_system/
        third_party), ``trust_level``, ``asset_value``, and ``description``.
        """
        rows = holder.engine().attack_surface() or []
        return {
            "result": rows,
            "total_items": len(rows),
            "kind": "attack_surface",
        }

    def complexity_hotspots(threshold: int = 10) -> dict[str, Any]:
        """Functions/methods with cyclomatic complexity ≥ ``threshold``.

        Useful for prioritising review effort; high complexity is
        correlated with bug density but is not a finding by itself.
        """
        rows = holder.engine().complexity_hotspots(int(threshold)) or []
        return {
            "result": [_slim_unit(r, host_root=host_root) for r in rows],
            "total_items": len(rows),
            "kind": "complexity_hotspots",
        }

    def functions_that_raise(exception: str) -> dict[str, Any]:
        """Functions whose parser-detected exception list includes
        ``exception`` (case-sensitive bare name, e.g. ``PermissionError``).
        """
        rows = (
            holder.engine().functions_that_raise(str(exception)) or []
        )
        return {
            "result": [_slim_unit(r, host_root=host_root) for r in rows],
            "total_items": len(rows),
            "kind": "functions_that_raise",
        }

    return [
        graph_summary,
        find_symbol,
        find_callers,
        find_callees,
        paths_between,
        entrypoint_paths_to,
        attack_surface,
        complexity_hotspots,
        functions_that_raise,
    ]


__all__ = [
    "GraphEngineHolder",
    "code_graph_tools",
]
