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

Path resolution: trailmark returns host-FS absolute paths in
``CodeUnit.location.file_path``. By default the tools surface those
paths as-is — callers are responsible for translating them to whatever
their agent's filesystem view expects (e.g. an overlay rooted at the
same project dir surfaces ``/relative/segment.py``). Pass
``path_resolver`` to inject that translation here rather than wrapping
every result on the caller side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "auto"
DEFAULT_MAX_RESULTS = 200
DEFAULT_MAX_PATHS = 25
_MAX_PATH_DEPTH = 30

# A path resolver maps a host-FS absolute path (as trailmark returns it)
# to whatever string the consuming agent should see. Returning ``None``
# means "keep the original host path". The signature is deliberately
# minimal so callers can wire any FS convention (rooted sandboxes,
# repo-relative, URI-prefixed, …) without leaking into this module.
PathResolver = Callable[[str], Optional[str]]


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


def strip_prefix_resolver(host_root: str, virtual_root: str = "/") -> PathResolver:
    """Build a ``PathResolver`` that strips ``host_root`` and re-roots
    matches under ``virtual_root``.

    Useful when the consuming agent operates on a sandboxed view of the
    same project tree trailmark parsed. Paths that fall outside
    ``host_root`` are returned unchanged so external-library / generated
    symbols stay unambiguous.

    Example:
        # Agent sees the project as ``/foo.py``; trailmark sees it as
        # ``/abs/path/foo.py``.
        resolver = strip_prefix_resolver("/abs/path", virtual_root="/")
    """
    import os

    root = os.path.normpath(host_root)
    vroot = virtual_root.rstrip("/") or ""

    def _resolve(host_path: str) -> Optional[str]:
        if host_path is None:
            return None
        norm = os.path.normpath(host_path)
        if norm == root:
            return vroot or "/"
        if norm.startswith(root + os.sep):
            rel = norm[len(root) + 1 :].replace(os.sep, "/")
            return f"{vroot}/{rel}" if vroot else f"/{rel}"
        return host_path

    return _resolve


def _slim_unit(
    unit: dict[str, Any],
    path_resolver: Optional[PathResolver] = None,
) -> dict[str, Any]:
    """Strip a trailmark CodeUnit dict down to fields useful to an agent.

    Drops parameters/branches/exception_types to keep token cost low; the
    agent can read the file directly with ``read_file`` if it needs the
    body. ``path_resolver``, when supplied, rewrites the ``file`` field
    so it composes with whatever FS view the consuming agent has.
    """
    loc = unit.get("location") or {}
    file_path = loc.get("file_path")
    if path_resolver is not None and file_path is not None:
        resolved = path_resolver(file_path)
        if resolved is not None:
            file_path = resolved
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


def resolve_local_root(fs: Any) -> Optional[str]:
    """Return a host-disk root path for ``fs``, or ``None`` for remote
    backends (``GitlabFileSystem`` and friends) where trailmark cannot
    parse files directly.

    Walks ``fs`` and its known wrapper attributes
    (``base_fs`` / ``fs`` / ``_fs``) — ``MemoryOverlayFileSystem`` wraps
    a ``RootedLocalFileSystem`` whose ``root_path`` is the actual disk
    location. Stops at the first ``root_path`` it finds.
    """
    current = fs
    for _ in range(8):
        rp = getattr(current, "root_path", None)
        if rp is not None:
            return str(rp)
        inner = (
            getattr(current, "base_fs", None)
            or getattr(current, "fs", None)
            or getattr(current, "_fs", None)
        )
        if inner is None or inner is current:
            return None
        current = inner
    return None


def attach_graph_tools_if_local(
    fs: Any,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> list:
    """Return ``code_graph_tools`` wired against ``fs``'s local root, or
    ``[]`` when the fs isn't backed by a host-disk directory.

    Lets every worker factory opt into call-graph navigation with a
    single line, with no behaviour change for non-local filesystems
    (GitlabFS, S3, …). The graph tools share the agent's path
    convention through ``strip_prefix_resolver`` so results compose
    with the file-mutation tools sharing the same ``fs``.
    """
    root = resolve_local_root(fs)
    if root is None:
        return []
    return code_graph_tools(
        root,
        language=language,
        path_resolver=strip_prefix_resolver(root),
    )


def code_graph_tools(
    root: str | Path,
    *,
    language: str = DEFAULT_LANGUAGE,
    path_resolver: Optional[PathResolver] = None,
) -> list:
    """Build the code-graph tool set for an LLM agent.

    ``root`` must be a host-filesystem path (trailmark uses tree-sitter
    against real files; in-memory overlays are not supported).

    ``path_resolver`` is an optional ``Callable[[str], Optional[str]]``
    that rewrites the host-FS file paths trailmark returns into whatever
    the consuming agent expects. Without it, paths are surfaced
    untouched. Callers who run their agent against a
    ``RootedLocalFileSystem``-sandboxed overlay typically want
    ``strip_prefix_resolver(root)`` here so graph results compose with
    the agent's file-mutation tools.
    """
    holder = GraphEngineHolder(root=Path(root), language=language)

    def graph_summary() -> dict[str, Any]:
        """
        Report high-level diagnostics for the code graph.

        Use this first to gauge graph size and coverage before drilling in
        with the other graph tools.

        Returns counts of nodes, edges, and detected entrypoints plus
        dependency info, wrapped under "result".
        """
        return {"result": holder.engine().summary(), "kind": "graph_summary"}

    def find_symbol(symbol: str) -> dict[str, Any]:
        """
        Resolve a name to matching graph nodes with their source locations.

        Matches by full node id, exact name, or bare (case-insensitive) name,
        returning up to 50 nodes.

        Args:
            symbol: Symbol name or node id to look up.

        Returns the matching nodes; an empty list means the name is not in the
        graph.
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
                matches.append(_slim_unit(d, path_resolver=path_resolver))
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
        """
        Find the direct callers of a symbol (including inferred call edges).

        Args:
            symbol: Symbol name or node id to find callers of.
            max_results: Maximum number of callers to return.

        Returns the calling nodes, each tagged with edge confidence. An empty
        list with a "note" means the symbol could not be resolved in the
        graph; an empty list without a note means the symbol exists but has no
        known callers.
        """
        node_id = _resolve_id(symbol)
        graph = holder.graph()
        if node_id is None:
            return {
                "result": [],
                "total_items": 0,
                "kind": "callers",
                "note": f"symbol '{symbol}' not found in graph",
            }
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
                slim = _slim_unit(d, path_resolver=path_resolver)
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
        """
        Find the direct callees of a symbol (what it calls).

        Targets that cannot be resolved to a known node are still returned,
        tagged with kind="unresolved".

        Args:
            symbol: Symbol name or node id to find callees of.
            max_results: Maximum number of callees to return.

        Returns the called nodes. An empty list with a "note" means the symbol
        could not be resolved in the graph; an empty list without a note means
        the symbol exists but calls nothing tracked.
        """
        node_id = _resolve_id(symbol)
        graph = holder.graph()
        if node_id is None:
            return {
                "result": [],
                "total_items": 0,
                "kind": "callees",
                "note": f"symbol '{symbol}' not found in graph",
            }
        rows: list[dict[str, Any]] = []
        from dataclasses import asdict

        for edge in graph.edges:
            if edge.source_id != node_id or edge.kind.value != "calls":
                continue
            target_unit = graph.nodes.get(edge.target_id)
            if target_unit is not None:
                d = asdict(target_unit)
                d["kind"] = target_unit.kind.value
                row = _slim_unit(d, path_resolver=path_resolver)
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
        """
        Find call paths from one symbol to another.

        Args:
            src: Starting symbol (the caller end).
            dst: Target symbol (the callee end).
            max_paths: Maximum number of paths to return.

        Returns each path as an ordered list of node ids; an empty list means
        no call path connects them.
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
        """
        Find call paths from any detected entrypoint to a symbol.

        Useful for judging reachability — whether externally-triggered code
        can reach the target.

        Args:
            symbol: Target symbol to reach.
            max_paths: Maximum number of paths to return.
            max_depth: Maximum path length to explore.

        Returns each path as an ordered list of node ids; an empty list means
        no entrypoint reaches the symbol within max_depth.
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
        """
        List framework-detected entrypoints (the attack surface).

        These are externally-reachable handlers (HTTP routes, etc.) annotated
        with trust and asset metadata where known.

        Returns the entrypoint nodes; an empty list means none were detected.
        """
        rows = holder.engine().attack_surface() or []
        return {
            "result": rows,
            "total_items": len(rows),
            "kind": "attack_surface",
        }

    def complexity_hotspots(threshold: int = 10) -> dict[str, Any]:
        """
        List functions whose cyclomatic complexity meets or exceeds a
        threshold.

        Args:
            threshold: Minimum complexity to include (default 10).

        Returns the matching functions with their locations; an empty list
        means nothing is that complex.
        """
        rows = holder.engine().complexity_hotspots(int(threshold)) or []
        return {
            "result": [_slim_unit(r, path_resolver=path_resolver) for r in rows],
            "total_items": len(rows),
            "kind": "complexity_hotspots",
        }

    def functions_that_raise(exception: str) -> dict[str, Any]:
        """
        List functions that may raise a given exception type.

        Args:
            exception: Exception type name to search for.

        Returns the matching functions with their locations; an empty list
        means none were detected raising it.
        """
        rows = (
            holder.engine().functions_that_raise(str(exception)) or []
        )
        return {
            "result": [_slim_unit(r, path_resolver=path_resolver) for r in rows],
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
    "PathResolver",
    "code_graph_tools",
    "strip_prefix_resolver",
]
