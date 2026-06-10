"""Router-prefix grouping of OpenAPI paths for trace coverage budgeting.

Per-path fan-out starves large APIs: every path gets a fresh agent with a
fresh memory namespace, so sibling handlers behind the same router prefix
(shared middleware, shared auth, shared serializers) are re-discovered
from scratch N times and the per-path token budget is spent on repeated
navigation. Grouping by route prefix makes the *group* the unit of
memory, skill injection, and (in pathpar) fork/concurrency scheduling:
paths under ``/workshop/...`` share one namespace and one budget, so
context discovered for one sibling carries to the next.

``depth`` is the number of leading path segments that define a group;
``depth <= 0`` means one group per path (the pre-grouping behavior —
group key == ``path_key``). Group keys are normalized exactly like
``OpenApiPath.path_key`` so a single-path group at full depth keeps its
historical namespace key.
"""

from __future__ import annotations

from dataclasses import dataclass

from contractor.workflows.trace_annotation import OpenApiOperation, OpenApiPath


def group_key_for_path(path: str, depth: int) -> str:
    """Group key for ``path`` at ``depth`` leading segments.

    Normalization mirrors ``OpenApiPath.path_key``: segments joined with
    ``_``, parameter braces stripped, empty key collapsed to ``root``.
    ``depth <= 0`` returns the full-path key.
    """
    segments = [s for s in path.strip("/").split("/") if s]
    if depth > 0:
        segments = segments[:depth]
    key = "_".join(segments).replace("{", "").replace("}", "")
    return key or "root"


@dataclass(frozen=True)
class PathGroup:
    """A set of OpenAPI paths sharing a route prefix."""

    key: str
    paths: tuple[OpenApiPath, ...]

    @property
    def operations(self) -> list[OpenApiOperation]:
        return [op for path in self.paths for op in path.operations]


def group_paths_by_prefix(
    paths: list[OpenApiPath],
    *,
    depth: int = 1,
) -> list[PathGroup]:
    """Group ``paths`` by their first ``depth`` route segments.

    ``depth <= 0`` yields one group per path keyed by ``path_key`` —
    byte-identical namespaces to the historical per-path behavior.
    Group order follows first appearance; path order within a group is
    preserved.
    """
    if depth <= 0:
        return [PathGroup(key=p.path_key, paths=(p,)) for p in paths]

    grouped: dict[str, list[OpenApiPath]] = {}
    for path in paths:
        grouped.setdefault(group_key_for_path(path.path, depth), []).append(path)
    return [PathGroup(key=key, paths=tuple(ps)) for key, ps in grouped.items()]
