"""Strict accepted plan provenance; source authority remains Server-owned."""

import hashlib
import re
from typing import Any

import jcs

from contractor_runtime.contracts.base import _require_resource_id, _require_selector, is_blank
from contractor_runtime.toolsets.audit_results.dependencies import (
    _exact,
    validate_retained_dependencies,
)
from contractor_runtime.toolsets.common.check_plans import (
    MAX_CHECK_CAPABILITIES,
    MAX_CHECK_CAPABILITY_TOOLS,
    validate_check_plan,
)

ACCEPTED_PLAN_SCHEMA = "contractor.audit.accepted-plan.v1"
MAX_ACCEPTED_PLAN_BYTES = 64 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _members(value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("accepted plan shape is invalid")
    return value


def _digest(value: Any) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("accepted plan digest is invalid")


def _mapping(value: Any) -> None:
    if (
        type(value) is not str
        or is_blank(value)
        or "/" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 128
    ):
        raise ValueError("accepted plan mapping is invalid")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(jcs.canonicalize(value)).hexdigest()


def _capabilities(values: Any) -> None:
    if not isinstance(values, list) or len(values) > MAX_CHECK_CAPABILITIES:
        raise ValueError("accepted plan capability bound is invalid")
    previous = ""
    for value in values:
        _members(value, {"stage", "agent", "ref", "tools"})
        for name in ("stage", "agent"):
            if type(value[name]) is not str or _NAME.fullmatch(value[name]) is None:
                raise ValueError("accepted plan capability Worker is invalid")
        ref = _members(value["ref"], {"toolsetId", "version"})
        if type(ref["toolsetId"]) is not str or type(ref["version"]) is not str:
            raise ValueError("accepted plan capability selector is invalid")
        selector = ref["toolsetId"] + "@" + ref["version"]
        _require_selector("accepted plan capability", selector)
        key = value["stage"] + "\x00" + value["agent"] + "\x00" + selector
        if key <= previous:
            raise ValueError("accepted plan capabilities must be sorted and unique")
        previous = key
        tools = value["tools"]
        if (
            not isinstance(tools, list)
            or not 1 <= len(tools) <= MAX_CHECK_CAPABILITY_TOOLS
            or any(type(tool) is not str or is_blank(tool) or "\x00" in tool for tool in tools)
        ):
            raise ValueError("accepted plan tools are invalid")
        if tools != sorted(set(tools)):
            raise ValueError("accepted plan tools must be sorted and unique")


def validate_accepted_plan(
    value: Any,
    dependencies: Any,
    item_key: str,
    *,
    workflow_role: str | None = None,
    finding: Any = None,
) -> None:
    _members(
        value,
        {
            "schema",
            "owner_id",
            "project_id",
            "audit_id",
            "item_key",
            "baseline_digest",
            "profile_digest",
            "source",
            "plan",
            "plan_digest",
            "route",
            "authorization",
            "dependencies_digest",
        },
    )
    if value["schema"] != ACCEPTED_PLAN_SCHEMA:
        raise ValueError("accepted plan schema is invalid")
    for name in ("owner_id", "project_id", "audit_id"):
        if type(value[name]) is not str:
            raise ValueError("accepted plan identity is invalid")
        _require_resource_id("accepted plan identity", value[name])
    if (
        type(value["item_key"]) is not str
        or _IDENTIFIER.fullmatch(value["item_key"]) is None
        or value["item_key"] != item_key
    ):
        raise ValueError("accepted plan item identity differs")
    for name in ("baseline_digest", "profile_digest", "plan_digest"):
        _digest(value[name])
    source = _members(
        value["source"],
        {"kind", "receipt_id", "proposed_check_ordinal", "run_id", "proposal", "retained_proposal"},
    )
    if source["kind"] != "finding-receipt":
        raise ValueError("accepted plan source is unsupported")
    for name in ("receipt_id", "run_id"):
        if type(source[name]) is not str:
            raise ValueError("accepted plan source identity is invalid")
        _require_resource_id("accepted plan source", source[name])
    ordinal = source["proposed_check_ordinal"]
    if type(ordinal) is not int or not 0 <= ordinal < 512:
        raise ValueError("accepted plan source ordinal is invalid")
    for name in ("proposal", "retained_proposal"):
        _exact(source[name])
    for name in ("digest", "mediaType", "sizeBytes"):
        if source["proposal"][name] != source["retained_proposal"][name]:
            raise ValueError("accepted plan exact source metadata differs")
    if source["retained_proposal"]["ref"]["namespace"] != "finding-materials":
        raise ValueError("accepted plan retained namespace differs")
    validate_check_plan(value["plan"])
    if _sha(value["plan"]) != value["plan_digest"]:
        raise ValueError("accepted plan canonical plan digest differs")
    route = _members(
        value["route"],
        {
            "mapping_id",
            "mapping_digest",
            "workflow_role",
            "binding_digest",
            "workflow_digest",
            "skills_digest",
            "required_capabilities",
        },
    )
    _mapping(route["mapping_id"])
    _mapping(route["workflow_role"])
    for name in ("mapping_digest", "binding_digest", "workflow_digest", "skills_digest"):
        _digest(route[name])
    _capabilities(route["required_capabilities"])
    authorization = _members(
        value["authorization"],
        {"scope_digest", "active_checks", "approval_requirement", "limits_digest"},
    )
    _digest(authorization["scope_digest"])
    _digest(authorization["limits_digest"])
    if (
        type(authorization["active_checks"]) is not str
        or type(authorization["approval_requirement"]) is not str
    ):
        raise ValueError("accepted plan authorization policy is invalid")
    if authorization["active_checks"] not in {
        "automatic",
        "prohibited",
        "approval-required",
    } or authorization["approval_requirement"] not in {"none", "active-check", "human-review"}:
        raise ValueError("accepted plan authorization policy is invalid")
    if dependencies is None:
        if value["dependencies_digest"] is not None:
            raise ValueError("accepted plan dependency manifest is missing")
    else:
        validate_retained_dependencies(dependencies, item_key)
        if value["dependencies_digest"] != _sha(dependencies):
            raise ValueError("accepted plan dependency digest differs")
        for name in (
            "owner_id",
            "project_id",
            "audit_id",
            "item_key",
            "baseline_digest",
            "profile_digest",
        ):
            if value[name] != dependencies[name]:
                raise ValueError("accepted plan dependency identity differs")
        for name in ("workflow_digest", "skills_digest"):
            if route[name] != dependencies[name]:
                raise ValueError("accepted plan dependency consumer differs")
        if not any(
            dependency["name"] == "proposal"
            and dependency["kind"] == "finding-receipt"
            and dependency["origin"]["receipt_id"] == source["receipt_id"]
            and dependency["origin"]["run_id"] == source["run_id"]
            and dependency["origin"]["proposed_check_ordinal"] == ordinal
            and dependency["source"] == source["proposal"]
            and dependency["retained"] == source["retained_proposal"]
            for dependency in dependencies["dependencies"]
        ):
            raise ValueError("accepted plan source dependency differs")
    if workflow_role is not None:
        if workflow_role != route["workflow_role"] or not isinstance(finding, dict):
            raise ValueError("accepted plan task route differs")
        if (
            type(finding.get("proposed_check_ordinal")) is not int
            or finding.get("receipt_id") != source["receipt_id"]
            or finding.get("proposed_check_ordinal") != ordinal
            or finding.get("proposal_ref") != source["retained_proposal"]["ref"]
            or finding.get("proposal_digest") != source["retained_proposal"]["digest"]
            or finding.get("objective") != value["plan"]["objective"]
            or finding.get("method") != value["plan"]["method"]
        ):
            raise ValueError("accepted plan task source differs")
    if len(jcs.canonicalize(value)) > MAX_ACCEPTED_PLAN_BYTES:
        raise ValueError("accepted plan metadata exceeds its bound")
