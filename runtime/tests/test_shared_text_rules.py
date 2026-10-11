"""Text rules the Runtime shares with the Go Server through api/testdata fixtures.

The Go side reads the same files (internal/contracts/shared_text_rules_test.go,
internal/contracts/reporting/shared_text_rules_test.go and
internal/config/workspace_path_shared_test.go); make test-wire-cross-language
runs both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from contractor_runtime.agent_skills.package import _parse_manifest
from contractor_runtime.contracts import (
    AgentStateSnapshot,
    StageContentRequest,
    TerminationError,
    WorkspaceObservationSummary,
)
from contractor_runtime.contracts.base import (
    MAX_STATE_WORKSPACE_PATH_BYTES,
    MAX_WORKER_STATE_PATH_BYTES,
    MAX_WORKER_STATE_PATH_COMPONENTS,
    MAX_WORKSPACE_TARGET_BYTES,
    MAX_WORKSPACE_TARGET_COMPONENTS,
    WHITESPACE,
    _require_state_workspace_path,
    _require_workspace_target,
    is_blank,
    json_byte_size,
)
from contractor_runtime.observations import exact_observation_path
from contractor_runtime.toolsets.audit_results import contracts as audit_contracts
from contractor_runtime.toolsets.audit_results import v2 as audit_v2
from contractor_runtime.toolsets.security_findings import submission as finding_submission

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"
LINE_SEPARATOR = chr(0x2028)


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _workspace_path_cases() -> tuple[dict[str, Any], list[tuple[str, str, bool, bool]]]:
    """Strictly read workspace-path-cases.json, resolving each case's value."""

    fixture = _fixture("workspace-path-cases.json")
    if set(fixture) != {"description", "rules", "cases"} or not fixture["cases"]:
        raise ValueError("workspace-path-cases.json has an unexpected shape")
    cases = []
    for case in fixture["cases"]:
        keys = set(case) - {"value", "repeat"}
        if keys != {"name", "workspaceTarget", "workerStatePath"} or (
            ("value" in case) == ("repeat" in case)
        ):
            raise ValueError(f"{case.get('name')}: a case needs a name, both verdicts and a value")
        if "value" in case:
            value = case["value"]
        else:
            repeat = case["repeat"]
            if set(repeat) != {"unit", "count", "suffix"}:
                raise ValueError(f"{case['name']}: repeat needs unit, count and suffix")
            value = repeat["unit"] * repeat["count"] + repeat["suffix"]
        cases.append((case["name"], value, case["workspaceTarget"], case["workerStatePath"]))
    return fixture["rules"], cases


def _accepts(validator: Callable[[str], object], value: str) -> bool:
    try:
        validator(value)
    except ValueError:
        return False
    return True


def _reported_file(value: str) -> object:
    return WorkspaceObservationSummary.model_validate(
        {
            "scopedFiles": 0,
            "scopeComplete": False,
            "discoveredFiles": 0,
            "readFiles": 1,
            "matchedFiles": 0,
            "modifiedFiles": 0,
            "detailComplete": False,
            "filesRead": [value],
            "filesReadTruncated": False,
        }
    )


def test_shared_workspace_path_cases() -> None:
    rules, cases = _workspace_path_cases()
    assert rules == {
        "workspaceTarget": {
            "emptyAllowed": True,
            "maxBytes": MAX_WORKSPACE_TARGET_BYTES,
            "maxComponents": MAX_WORKSPACE_TARGET_COMPONENTS,
        },
        "workerStatePath": {
            "emptyAllowed": False,
            "maxBytes": MAX_WORKER_STATE_PATH_BYTES,
            "maxComponents": MAX_WORKER_STATE_PATH_COMPONENTS,
        },
    }
    for name, value, workspace_target, worker_state_path in cases:
        assert _accepts(_require_workspace_target, value) is workspace_target, name
        reported = value != "" and workspace_target
        assert _accepts(_reported_file, value) is reported, name
        assert _accepts(_require_state_workspace_path, value) is worker_state_path, name
        # The project paths the Runtime records in Worker State are exact
        # normalized paths of the same rule.
        assert _accepts(exact_observation_path, value) is worker_state_path, name


def _blank_text_cases() -> tuple[list[str], list[str]]:
    fixture = _fixture("blank-text-cases.json")
    if set(fixture) != {"description", "whitespace", "blank", "text"}:
        raise ValueError("blank-text-cases.json has an unexpected shape")
    if not fixture["blank"] or not fixture["text"]:
        raise ValueError("blank-text-cases.json needs blank and text values")
    assert fixture["whitespace"] == list(WHITESPACE)
    return fixture["blank"], fixture["text"]


def test_shared_blank_text_cases() -> None:
    blank, text = _blank_text_cases()
    python_whitespace = {chr(code) for code in range(0x110000) if chr(code).isspace()}
    # str.isspace adds exactly the information separators, which the Server treats as text.
    assert python_whitespace - set(WHITESPACE) == {chr(code) for code in range(0x1C, 0x20)}
    assert set(WHITESPACE) <= python_whitespace
    request = _fixture("valid/stage-content-request.json")
    for value in blank:
        assert is_blank(value), repr(value)
        with pytest.raises(ValidationError):
            StageContentRequest.model_validate_json(json.dumps({**request, "objective": value}))
        with pytest.raises(ValidationError):
            TerminationError(code="worker_failed", message=value, retryable=False)
    for value in text:
        assert not is_blank(value), repr(value)
        # The hub audit's example: an objective of U+001C passes the Server
        # and must not fail allocation work in the Runtime.
        StageContentRequest.model_validate_json(json.dumps({**request, "objective": value}))
        TerminationError(code="worker_failed", message=value, retryable=False)


def _skill_manifest(description: str, body: str) -> bytes:
    # A JSON string is a YAML double-quoted scalar with the same escapes.
    return f"---\nname: demo\ndescription: {json.dumps(description)}\n---\n{body}".encode()


SIBLING_TEXT_CHECKS: dict[str, Callable[[str], object]] = {
    "audit summary": audit_contracts._text,
    "audit-results@2 text": lambda value: audit_v2._text(value, "summary", "item"),
    "finding proposal text": lambda value: finding_submission._text("title", value, required=True),
    "skill description": lambda value: _parse_manifest(_skill_manifest(value, "Body\n"), "demo"),
    "skill body": lambda value: _parse_manifest(_skill_manifest("Demo", value), "demo"),
}


@pytest.mark.parametrize("check", sorted(SIBLING_TEXT_CHECKS))
def test_shared_blank_text_holds_for_audit_finding_and_skill_text(check: str) -> None:
    # Each of these has a Server twin that trims with strings.TrimSpace
    # (auditdomain.validateText, agentskills.parseManifest). NUL has rules of
    # its own there, so it is left out.
    blank, text = _blank_text_cases()
    validator = SIBLING_TEXT_CHECKS[check]
    for value in blank:
        assert not _accepts(validator, value), (check, value)
    for value in text:
        if "\x00" not in value:
            assert _accepts(validator, value), (check, value)


def test_shared_json_byte_size_cases() -> None:
    fixture = _fixture("json-byte-size-cases.json")
    assert set(fixture) == {"description", "cases"} and fixture["cases"]
    pydantic_json = TypeAdapter(Any)
    for case in fixture["cases"]:
        assert set(case) == {"name", "value", "bytes"}, case
        assert json_byte_size(case["value"]) == case["bytes"], case["name"]
        assert len(pydantic_json.dump_json(case["value"])) == case["bytes"], case["name"]


def _shared_bound_path_list(size: int) -> list[str]:
    """Build the list internal/contracts/reporting's sharedBoundPathList builds."""

    full_body, last_body_limit = 3 * 1300, 4000
    paths: list[str] = []
    remaining = size - 2  # brackets
    while remaining > 0:
        prefix = f"d/{len(paths):05d}/"
        overhead = len(prefix) + 2 + (1 if paths else 0)  # quotes and comma
        body = remaining - overhead
        if body > last_body_limit:
            body = full_body
        paths.append(prefix + LINE_SEPARATOR * (body // 3) + "x" * (body % 3))
        remaining -= overhead + body
    return paths


def _snapshot(mutate: Callable[[dict[str, Any]], None]) -> str:
    snapshot = _fixture("valid/agent-state-snapshot.json")
    mutate(snapshot["state"])
    return json.dumps(snapshot)


@pytest.mark.parametrize(("suffix", "valid"), [("aa", True), ("aaa", False)])
def test_shared_worker_state_arguments_bound(suffix: str, valid: bool) -> None:
    # The Go twin is TestSharedWorkerStateBoundsCountCompactUTF8: {"path":"..."}
    # costs 11 bytes and 1361 U+2028 cost 4083, so "aa" reaches the 4096-byte bound.
    arguments = {"path": LINE_SEPARATOR * 1361 + suffix}
    assert json_byte_size(arguments) == 4094 + len(suffix)

    def mutate(state: dict[str, Any]) -> None:
        state["metrics"]["toolCalls"][0]["arguments"] = arguments

    if valid:
        AgentStateSnapshot.model_validate_json(_snapshot(mutate))
    else:
        with pytest.raises(ValidationError):
            AgentStateSnapshot.model_validate_json(_snapshot(mutate))


@pytest.mark.parametrize(("extra", "valid"), [(0, True), (1, False)])
def test_shared_worker_state_path_list_bound(extra: int, valid: bool) -> None:
    paths = _shared_bound_path_list(MAX_STATE_WORKSPACE_PATH_BYTES + extra)
    assert json_byte_size(paths) == MAX_STATE_WORKSPACE_PATH_BYTES + extra

    def mutate(state: dict[str, Any]) -> None:
        state["currentInvocation"]["workspace"]["scopePaths"] = paths

    if valid:
        AgentStateSnapshot.model_validate_json(_snapshot(mutate))
    else:
        with pytest.raises(ValidationError):
            AgentStateSnapshot.model_validate_json(_snapshot(mutate))
