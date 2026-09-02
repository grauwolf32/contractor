from __future__ import annotations

import inspect
import json

from contractor_runtime.toolsets.trailmark_child import (
    MAX_MODEL_RESPONSE_BYTES,
    _bounded_simple_paths,
    _fit_path_rows,
)


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
