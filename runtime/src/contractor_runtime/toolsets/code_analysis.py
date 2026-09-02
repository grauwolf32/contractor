"""Static contract shared by the staged code-analysis implementation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType

CODE_ANALYSIS_REF = "code-analysis@1"

SHALLOW_TOOLS = frozenset({"list_symbols", "search_def"})
GRAPH_TOOLS = frozenset(
    {
        "attack_surface",
        "complexity_hotspots",
        "entrypoint_paths_to",
        "find_callees",
        "find_callers",
        "find_symbol",
        "functions_that_raise",
        "graph_summary",
        "paths_between",
    }
)
EXPORTED_TOOLS = SHALLOW_TOOLS | GRAPH_TOOLS

PINNED_DEPENDENCIES = MappingProxyType(
    {
        "trailmark": "0.5.0",
        "tree-sitter": "0.25.2",
        "tree-sitter-language-pack": "1.14.3",
    }
)


def dependency_versions_match() -> bool:
    """Return whether the reviewed parser and graph distributions are installed."""

    try:
        return all(version(name) == expected for name, expected in PINNED_DEPENDENCIES.items())
    except PackageNotFoundError:
        return False
