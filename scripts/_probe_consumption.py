#!/usr/bin/env python3
"""Probe (a): verify _apply_forced_params consumes the ContextVars correctly."""
import sys
sys.path.insert(0, "/home/ruslan/src/contractor")

from contractor.utils.llm_compat import (
    _apply_forced_params,
    forced_response_format,
    forced_tool_choice,
)

RF = {"type": "json_schema", "json_schema": {"name": "SubtaskExecutionResult", "schema": {"type": "object"}}}

print("=== CASE 1: forced_tool_choice='none', forced_response_format=RF, kwargs has response_format=None ===")
forced_tool_choice.set("none")
forced_response_format.set(RF)
# mimic ADK lite_llm pre-population: response_format key present but value None
kwargs = {"temperature": 0.0, "response_format": None, "max_tokens": 4096}
print(f"  kwargs BEFORE: {kwargs}")
_apply_forced_params(kwargs)
print(f"  kwargs AFTER : {kwargs}")
assert kwargs["tool_choice"] == "none", f"tool_choice={kwargs.get('tool_choice')!r}"
assert kwargs["response_format"] == RF, f"response_format={kwargs.get('response_format')!r}"
print("  PASS: tool_choice=='none' AND response_format==RF (value-None overridden)")

print("\n=== CASE 2: forced unset (None) -> kwargs unchanged ===")
forced_tool_choice.set(None)
forced_response_format.set(None)
kwargs2 = {"temperature": 0.0, "response_format": None, "max_tokens": 4096}
before = dict(kwargs2)
print(f"  kwargs BEFORE: {kwargs2}")
_apply_forced_params(kwargs2)
print(f"  kwargs AFTER : {kwargs2}")
assert kwargs2 == before, f"kwargs mutated: {kwargs2} != {before}"
assert "tool_choice" not in kwargs2, "tool_choice should not be injected"
print("  PASS: kwargs unchanged, no tool_choice injected")

print("\n=== CASE 3: forced='none' but kwargs already has a real response_format (must NOT clobber) ===")
forced_tool_choice.set("none")
forced_response_format.set(RF)
existing = {"type": "json_object"}
kwargs3 = {"response_format": existing}
print(f"  kwargs BEFORE: {kwargs3}")
_apply_forced_params(kwargs3)
print(f"  kwargs AFTER : {kwargs3}")
assert kwargs3["tool_choice"] == "none"
assert kwargs3["response_format"] is existing, "should preserve caller's real response_format"
print("  PASS: tool_choice forced, existing real response_format preserved")

forced_tool_choice.set(None)
forced_response_format.set(None)
print("\nALL CONSUMPTION ASSERTIONS PASSED")
