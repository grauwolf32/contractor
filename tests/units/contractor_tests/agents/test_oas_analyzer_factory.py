"""Unit tests for the oas_analyzer prompt factory.

Regression for a chained-conditional bug in ``TaskDescription.format()``: it
returned objective+instructions only when instructions existed, examples-only
when they didn't, and "" when neither was present. As a result the
``general`` sub-agents (objective only) ran with an empty task body, and
idor/ssrf (objective+instructions+examples) silently dropped their examples.
Sections must compose additively instead.
"""
from __future__ import annotations

from contractor.agents.oas_analyzer.prompts.factory import (
    SectionPrompts,
    TaskDescription,
)


def test_objective_only():
    out = TaskDescription(objective="OBJ").format()
    assert "OBJECTIVE:\nOBJ" in out
    assert "INSTRUCTIONS:" not in out
    assert "EXAMPLES:" not in out


def test_objective_and_instructions():
    out = TaskDescription(objective="OBJ", instructions="INS").format()
    assert "OBJECTIVE:\nOBJ" in out
    assert "INSTRUCTIONS:\nINS" in out
    assert "EXAMPLES:" not in out


def test_objective_and_examples():
    out = TaskDescription(objective="OBJ", examples="EX").format()
    assert "OBJECTIVE:\nOBJ" in out
    assert "EXAMPLES:\nEX" in out
    assert "INSTRUCTIONS:" not in out


def test_all_three_sections_present_and_ordered():
    out = TaskDescription(
        objective="OBJ", instructions="INS", examples="EX"
    ).format()
    assert "OBJECTIVE:\nOBJ" in out
    assert "INSTRUCTIONS:\nINS" in out
    assert "EXAMPLES:\nEX" in out
    # additive composition keeps source order: objective -> instructions -> examples
    assert out.index("OBJECTIVE:") < out.index("INSTRUCTIONS:") < out.index("EXAMPLES:")


def test_objective_never_dropped_when_no_instructions_or_examples():
    # The "general" sub-agents (ddos/datasec/appsec) have objective only; they
    # must not end up with an empty body.
    out = TaskDescription(objective="analyze the schema").format()
    assert out.strip() != ""
    assert "analyze the schema" in out


def test_format_task_wraps_with_role_and_output_format():
    section = SectionPrompts(fmt="FMT", role="ROLE")
    out = section.format_task(TaskDescription(objective="OBJ", examples="EX"))
    assert "ROLE:\nROLE" in out
    assert "OBJECTIVE:\nOBJ" in out
    assert "EXAMPLES:\nEX" in out
    assert "OUTPUT FORMAT:\nFMT" in out
