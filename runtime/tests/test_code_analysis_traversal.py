from __future__ import annotations

import inspect
import json
import random
import time
from collections.abc import Mapping

from contractor_runtime.toolsets.code_analysis.trailmark_child import (
    MAX_MODEL_RESPONSE_BYTES,
    MAX_TRAVERSAL_STEPS,
    _bounded_simple_paths,
    _fit_path_rows,
)


def _ring(size: int, fan_out: int) -> dict[str, tuple[str, ...]]:
    return {
        f"n{index:02d}": tuple(f"n{(index + step) % size:02d}" for step in range(1, fan_out + 1))
        for index in range(size)
    }


def test_high_branching_traversal_stops_after_limit_plus_one() -> None:
    branches = tuple(f"a{index:02d}" for index in range(20))
    leaves = tuple(f"b{index:02d}" for index in range(20))
    adjacency = {
        "source": branches,
        **{branch: leaves for branch in branches},
        **{leaf: ("target",) for leaf in leaves},
    }

    paths, truncated, traversal_steps = _bounded_simple_paths(
        adjacency,
        ("source",),
        "target",
        max_depth=4,
        limit=3,
    )

    assert paths == [
        ("source", "a00", "b00", "target"),
        ("source", "a00", "b01", "target"),
        ("source", "a00", "b02", "target"),
    ]
    assert truncated is True
    assert traversal_steps == 9
    assert "digraph_all_simple_paths" not in inspect.getsource(_bounded_simple_paths)


def test_traversal_avoids_cycles_and_counts_depth_in_nodes() -> None:
    adjacency = {
        "source": ("a",),
        "a": ("source", "b"),
        "b": ("a", "target"),
    }

    shallow, shallow_truncated, _ = _bounded_simple_paths(
        adjacency,
        ("source",),
        "target",
        max_depth=3,
        limit=10,
    )
    paths, truncated, _ = _bounded_simple_paths(
        adjacency,
        ("source",),
        "target",
        max_depth=4,
        limit=10,
    )

    assert shallow == []
    assert shallow_truncated is False
    assert paths == [("source", "a", "b", "target")]
    assert truncated is False
    assert all(len(path) == len(set(path)) for path in paths)


def test_entrypoint_sources_are_ordered_and_identity_path_is_valid() -> None:
    paths, truncated, traversal_steps = _bounded_simple_paths(
        {"later": ("target",), "earlier": ("target",)},
        ("earlier", "target", "later"),
        "target",
        max_depth=2,
        limit=10,
    )

    assert paths == [
        ("earlier", "target"),
        ("target",),
        ("later", "target"),
    ]
    assert truncated is False
    assert traversal_steps == 2


def test_unreachable_target_in_dense_cyclic_graph_costs_no_enumeration() -> None:
    adjacency = {**_ring(30, 4), "target": ()}

    started = time.monotonic()
    paths, truncated, traversal_steps = _bounded_simple_paths(
        adjacency,
        tuple(sorted(adjacency)),
        "target",
        max_depth=20,
        limit=20,
    )

    assert time.monotonic() - started < 1
    assert paths == [("target",)]
    assert truncated is False
    assert traversal_steps == 0


def test_target_beyond_remaining_depth_is_pruned_before_descending() -> None:
    adjacency = {**_ring(30, 4), "n00": (*_ring(30, 4)["n00"], "a"), "a": ("b",), "b": ("target",)}

    shallow, shallow_truncated, shallow_steps = _bounded_simple_paths(
        adjacency, ("n10",), "target", max_depth=5, limit=20
    )
    deep, deep_truncated, _ = _bounded_simple_paths(
        adjacency, ("n29",), "target", max_depth=5, limit=20
    )

    assert shallow == [] and shallow_truncated is False
    assert shallow_steps == 0
    assert deep == [("n29", "n00", "a", "b", "target")]
    assert deep_truncated is False


def test_step_budget_reports_an_incomplete_traversal_as_truncated() -> None:
    # Every node's shortest route to the target runs back through the source,
    # which a simple path cannot revisit: distance pruning cannot help here.
    ring = _ring(30, 4)
    adjacency = {**ring, "n00": (*ring["n00"], "target")}

    bounded, bounded_truncated, bounded_steps = _bounded_simple_paths(
        adjacency, ("n00",), "target", max_depth=20, limit=1, max_steps=1_000
    )
    started = time.monotonic()
    _, default_truncated, default_steps = _bounded_simple_paths(
        adjacency, ("n00",), "target", max_depth=20, limit=1
    )

    assert bounded == [] and bounded_truncated is True
    assert bounded_steps == 1_000
    assert default_truncated is True and default_steps == MAX_TRAVERSAL_STEPS
    assert time.monotonic() - started < 8


def test_step_budget_is_shared_by_every_entrypoint_source() -> None:
    ring = _ring(30, 4)
    adjacency = {**ring, "n00": (*ring["n00"], "target")}

    paths, truncated, traversal_steps = _bounded_simple_paths(
        adjacency,
        ("n00", "n00", "n00"),
        "target",
        max_depth=20,
        limit=50,
        max_steps=500,
    )

    assert paths == []
    assert truncated is True
    assert traversal_steps == 500


def test_distance_pruning_returns_exactly_the_unpruned_paths() -> None:
    generator = random.Random(99)
    for _ in range(200):
        size = generator.randint(2, 9)
        nodes = [f"n{index}" for index in range(size)]
        adjacency = {
            node: tuple(sorted(generator.sample(nodes, generator.randint(0, min(3, size)))))
            for node in nodes
        }
        target = generator.choice(nodes)
        sources = tuple(generator.sample(nodes, generator.randint(1, size)))
        max_depth = generator.randint(1, 6)
        limit = generator.randint(1, 8)

        paths, truncated, _ = _bounded_simple_paths(
            adjacency, sources, target, max_depth=max_depth, limit=limit
        )
        expected = _all_simple_paths(adjacency, sources, target, max_depth)

        assert paths == expected[:limit]
        assert truncated is (len(expected) > limit)


def _all_simple_paths(
    adjacency: Mapping[str, tuple[str, ...]],
    sources: tuple[str, ...],
    target: str,
    max_depth: int,
) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []

    def walk(path: list[str]) -> None:
        if path[-1] == target:
            found.append(tuple(path))
            return
        if len(path) >= max_depth:
            return
        for successor in adjacency.get(path[-1], ()):
            if successor not in path:
                walk([*path, successor])

    for source in sources:
        walk([source])
    return found


def test_path_projection_is_cut_before_model_response_ceiling() -> None:
    coverage = {
        "analyzedFiles": 1,
        "analyzedBytes": 1,
        "binaryFiles": 0,
        "unsupportedSourceFiles": 0,
        "oversizedFiles": 0,
        "parseErrors": 0,
        "incomplete": False,
        "reasons": [],
    }
    rows = [
        [
            {
                "symbolId": f"cas1.symbol-{index}-{node}",
                "name": "symbol",
                "kind": "function",
                "path": "nested/" + "x" * 4000,
                "line": 1,
                "endLine": 1,
                "column": 0,
            }
            for node in range(2)
        ]
        for index in range(50)
    ]

    selected, truncated = _fit_path_rows(rows, False, coverage)
    encoded = json.dumps(
        {"items": selected, "truncated": truncated, "coverage": coverage},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert 0 < len(selected) < len(rows)
    assert truncated is True
    assert len(encoded) <= MAX_MODEL_RESPONSE_BYTES
