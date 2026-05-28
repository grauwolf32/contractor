"""Contract tests for the agent-facing (frontend) tool surface.

These enforce, across whole tool registries, the conventions that were
previously only followed by hand:

1. every frontend tool has a non-empty docstring (it IS the model-facing spec);
2. ADK can build a function declaration whose parameters match the tool's real
   signature — this is the regression guard for the "wrapped tool silently
   loses its parameters" failure mode (see contractor/tools/result.py);
3. the result/error envelope helpers behave as specified.

To bring another module under contract, add its registry to ``REGISTRIES``.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from contractor.tools.code.annotations import annotation_tools
from contractor.tools.code.graph import code_graph_tools
from contractor.tools.code.tools import code_tools
from contractor.tools.fs.format import FileFormat
from contractor.tools.fs.read_tools import ro_file_tools
from contractor.tools.fs.write_tools import rw_file_tools
from contractor.tools.http import http_tools
from contractor.tools.likec4 import likec4_tools
from contractor.tools.memory import memory_tools
from contractor.tools.openapi.vacuum import openapi_linter_tools
from contractor.tools.result import err, guard, is_envelope, ok, ok_page
from contractor.tools.vuln import verification_tools, vulnerability_report_tools

# ADK injects these itself and strips them from the model-facing declaration:
# anything named tool_context, or any param typed ToolContext/CallbackContext
# (the param name varies — some tools call it ``ctx``).
_INJECTED_PARAMS = {"tool_context"}
_INJECTED_TYPES = ("ToolContext", "CallbackContext")


def _build_registries(tmp_path):
    """(id, [tool, ...]) pairs. Factories build tools without doing I/O."""
    return [
        ("code_graph", code_graph_tools(tmp_path)),
        ("code", code_tools(MemoryFileSystem())),
        ("fs_ro", ro_file_tools(MemoryFileSystem(), FileFormat(_format="json"))),
        ("fs_rw", rw_file_tools(MemoryFileSystem(), wrap_overlay=False)),
        ("http", http_tools("contract-test")),
        ("memory", memory_tools("contract-test")),
        ("vuln_report", vulnerability_report_tools("contract-test")),
        ("verification", verification_tools("contract-test")),
        ("likec4", likec4_tools(MemoryFileSystem())),
        ("openapi_linter", openapi_linter_tools("contract-test")),
        ("annotations", annotation_tools(MemoryFileSystem())),
    ]


def _expected_param_names(tool) -> set[str]:
    names = set()
    for name, p in inspect.signature(tool).parameters.items():
        if name in _INJECTED_PARAMS:
            continue
        # annotations are strings under `from __future__ import annotations`
        if any(t in str(p.annotation) for t in _INJECTED_TYPES):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        names.add(name)
    return names


def _adk_declared_param_names(tool) -> set[str]:
    """Param names ADK would advertise to the model for this tool."""
    from google.adk.tools.function_tool import FunctionTool

    ft = FunctionTool(func=tool)
    get_decl = getattr(ft, "_get_declaration", None)
    if get_decl is None:  # ADK API drift — nothing to assert
        pytest.skip("ADK FunctionTool._get_declaration unavailable")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decl = get_decl()
    if decl is None:
        return set()
    js = getattr(decl, "parameters_json_schema", None)
    if isinstance(js, dict):
        return set(js.get("properties", {}))
    params = getattr(decl, "parameters", None)
    if params is not None and getattr(params, "properties", None):
        return set(params.properties)
    return set()


def _all_tools(tmp_path):
    for reg_id, tools in _build_registries(tmp_path):
        for tool in tools:
            yield reg_id, tool


def test_every_frontend_tool_has_a_docstring(tmp_path):
    missing = [
        f"{reg}:{t.__name__}"
        for reg, t in _all_tools(tmp_path)
        if not (t.__doc__ or "").strip()
    ]
    assert not missing, f"frontend tools without a docstring: {missing}"


def test_adk_declaration_exposes_all_parameters(tmp_path):
    """Guards the wrapped-tool-loses-params regression for the whole registry."""
    mismatches = []
    for reg, tool in _all_tools(tmp_path):
        expected = _expected_param_names(tool)
        declared = _adk_declared_param_names(tool)
        if declared != expected:
            mismatches.append(
                f"{reg}:{tool.__name__} declared={sorted(declared)} "
                f"expected={sorted(expected)}"
            )
    assert not mismatches, "ADK declaration drifted from signature:\n" + "\n".join(
        mismatches
    )


def test_graph_tools_return_envelopes(tmp_path):
    """Over an empty project, every graph tool returns a result/error envelope."""
    tools = {t.__name__: t for t in code_graph_tools(tmp_path)}
    calls = {
        "graph_summary": (),
        "find_symbol": ("anything",),
        "find_callers": ("anything",),
        "find_callees": ("anything",),
        "paths_between": ("a", "b"),
        "entrypoint_paths_to": ("anything",),
        "attack_surface": (),
        "complexity_hotspots": (),
        "functions_that_raise": ("ValueError",),
    }
    assert set(calls) == set(tools), "calls map drifted from the graph registry"
    for name, args in calls.items():
        out = tools[name](*args)
        assert is_envelope(out), f"{name} returned non-envelope: {out!r}"
        # exactly one of result/error, never both
        assert ("result" in out) ^ ("error" in out), f"{name} mixed keys: {out!r}"


def test_graph_list_results_report_truncation_honestly(tmp_path):
    """total_items must never undercount the page, and truncated must agree.

    Guards the dishonest-truncation bug: a capped page reporting its own
    length as the total, so the model thinks it saw everything.
    """
    # A minimal real project so the graph builds (an empty dir yields an
    # error envelope, not a list).
    (tmp_path / "m.py").write_text(
        "def a():\n    return b()\n\ndef b():\n    return 1\n", encoding="utf-8"
    )
    tools = {t.__name__: t for t in code_graph_tools(tmp_path)}
    list_calls = {
        "find_symbol": ("x",),
        "find_callers": ("x",),
        "find_callees": ("x",),
        "paths_between": ("a", "b"),
        "entrypoint_paths_to": ("x",),
        "attack_surface": (),
        "complexity_hotspots": (),
        "functions_that_raise": ("ValueError",),
    }
    for name, args in list_calls.items():
        out = tools[name](*args)
        result = out.get("result")
        assert isinstance(result, list), f"{name} result is not a list: {out!r}"
        total = out["total_items"]
        assert total >= len(result), f"{name} total_items < returned: {out!r}"
        assert out["truncated"] == (total > len(result)), f"{name} bad flag: {out!r}"


# --- envelope helper unit tests ------------------------------------------


def test_ok_wraps_value_and_meta():
    assert ok([1], total_items=1, kind="x") == {
        "result": [1],
        "total_items": 1,
        "kind": "x",
    }


def test_err_has_no_success_keys():
    e = err("boom", code="not_found")
    assert e == {"error": "boom", "code": "not_found"}
    assert "result" not in e


def test_guard_wraps_plain_return():
    assert guard(lambda: 42) == {"result": 42}


def test_guard_passes_envelope_through():
    payload = ok([], total_items=0, kind="callers", note="missing")
    assert guard(lambda: payload) == payload


def test_guard_converts_exception_to_error_envelope():
    def boom():
        raise ValueError("nope")

    assert guard(boom) == {"error": "nope"}


def test_ok_page_not_truncated_when_total_matches():
    assert ok_page([1, 2], 2, kind="x") == {
        "result": [1, 2],
        "total_items": 2,
        "returned": 2,
        "truncated": False,
        "kind": "x",
    }


def test_ok_page_flags_truncation_when_total_exceeds_page():
    page = ok_page([1, 2], 10)
    assert page["returned"] == 2
    assert page["total_items"] == 10
    assert page["truncated"] is True
