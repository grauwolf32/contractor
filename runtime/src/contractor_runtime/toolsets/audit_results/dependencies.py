"""Strict retained dependency transport in trusted Audit inputs.

Server admission proves source membership and authorization. This reader checks
shape and exact task/manifest agreement without following embedded references.
"""

import hashlib
import re
from typing import Any

import jcs

from contractor_runtime.contracts import ArtifactRef, validate_media_type
from contractor_runtime.contracts.artifacts import validate_artifact_name
from contractor_runtime.contracts.base import _require_resource_id, _require_text

DEPENDENCY_SCHEMA = "contractor.audit.retained-dependencies.v1"
MAX_DEPENDENCIES = 32
MAX_DEPENDENCY_BYTES = 256 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def validate_retained_dependencies(value: Any, item_key: str) -> None:
    """Validate one bounded exact Server-authored item dependency closure."""
    required = {
        "schema",
        "owner_id",
        "project_id",
        "audit_id",
        "item_key",
        "baseline_digest",
        "profile_digest",
        "workflow_digest",
        "skills_digest",
        "dependencies",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema"] != DEPENDENCY_SCHEMA
    ):
        raise ValueError("retained dependency manifest is invalid")
    for name in ("owner_id", "project_id", "audit_id"):
        _resource_id(value[name])
    if type(value["item_key"]) is not str or "\x00" in value["item_key"]:
        raise ValueError("retained dependency item identity is invalid")
    _require_text("retained dependency item", value["item_key"], max_bytes=256)
    if value["item_key"] != item_key:
        raise ValueError("retained dependency item identity differs")
    for name in ("baseline_digest", "profile_digest", "workflow_digest", "skills_digest"):
        if type(value[name]) is not str or _DIGEST.fullmatch(value[name]) is None:
            raise ValueError("retained dependency digest is invalid")
    dependencies = value["dependencies"]
    if not isinstance(dependencies, list) or not 1 <= len(dependencies) <= MAX_DEPENDENCIES:
        raise ValueError("retained dependency count exceeds its bound")
    namespace = (
        "audit-"
        + hashlib.sha256(
            b"contractor.audit.identity.v1\x00audit\x00" + value["audit_id"].encode()
        ).hexdigest()
    )
    names = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != {
            "name",
            "kind",
            "projection",
            "source",
            "retained",
            "origin",
            "requires",
        }:
            raise ValueError("retained dependency descriptor is invalid")
        validate_artifact_name("dependency name", dependency["name"])
        names.append(dependency["name"])
        if dependency["projection"] != "exact":
            raise ValueError("retained dependency projection is unsupported")
        _exact(dependency["source"])
        _exact(dependency["retained"])
        expected_namespace = "finding-materials" if dependency["kind"] == "finding-receipt" else namespace
        retained_namespace = dependency["retained"]["ref"]["namespace"]
        if retained_namespace != expected_namespace and not (
            dependency["kind"] == "accepted-item-result" and retained_namespace == "finding-materials"
        ):
            raise ValueError("retained dependency namespace differs")
        for field in ("digest", "mediaType", "sizeBytes"):
            if dependency["source"][field] != dependency["retained"][field]:
                raise ValueError("retained dependency content differs")
        _origin(dependency["kind"], dependency["origin"])
    if names != sorted(set(names)):
        raise ValueError("retained dependency names must be sorted and unique")
    by_name = dict(zip(names, dependencies, strict=True))
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("retained dependency closure is cyclic")
        if name in visited:
            return
        visiting.add(name)
        requires = by_name[name]["requires"]
        if not isinstance(requires, list) or len(requires) >= len(dependencies):
            raise ValueError("retained dependency closure is invalid")
        for required_name in requires:
            validate_artifact_name("required dependency name", required_name)
        if requires != sorted(set(requires)) or any(key not in by_name for key in requires):
            raise ValueError("retained dependency closure is incomplete")
        for required_name in requires:
            visit(required_name)
        visiting.remove(name)
        visited.add(name)

    for name in names:
        visit(name)
    if len(jcs.canonicalize(value)) > MAX_DEPENDENCY_BYTES:
        raise ValueError("retained dependency bytes exceed their bound")


def _resource_id(value: Any) -> None:
    if type(value) is not str:
        raise ValueError("retained dependency identity is invalid")
    _require_resource_id("retained dependency identity", value)


def _exact(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"ref", "digest", "mediaType", "sizeBytes"}:
        raise ValueError("retained dependency exact metadata is incomplete")
    exact_ref = ArtifactRef.model_validate(value["ref"]).require_exact()
    _require_text("retained dependency revision", exact_ref.revision, max_bytes=64 * 1024)
    if type(value["digest"]) is not str or _DIGEST.fullmatch(value["digest"]) is None:
        raise ValueError("retained dependency exact digest is invalid")
    validate_media_type(value["mediaType"])
    if type(value["sizeBytes"]) is not int or value["sizeBytes"] < 0:
        raise ValueError("retained dependency exact size is invalid")


def _origin(kind: Any, origin: Any) -> None:
    variants = {
        "baseline": {"baseline_input"},
        "prepare-output": {"workflow_role", "logical_output", "execution_id", "attempt", "run_id"},
        "accepted-item-result": {"execution_id", "round_id", "item_id", "attempt", "run_id"},
        "finding-receipt": {"run_id", "receipt_id", "proposed_check_ordinal"},
    }
    if type(kind) is not str or kind not in variants or not isinstance(origin, dict):
        raise ValueError("retained dependency origin is unsupported")
    if set(origin) != {"kind"} | variants[kind] or origin["kind"] != kind:
        raise ValueError("retained dependency origin fields are invalid")
    for name in variants[kind] - {"attempt", "proposed_check_ordinal"}:
        if name in {"baseline_input", "workflow_role", "logical_output"}:
            if type(origin[name]) is not str or "/" in origin[name]:
                raise ValueError("retained dependency source name is invalid")
            _require_text("retained dependency source name", origin[name], max_bytes=128)
        else:
            _resource_id(origin[name])
    if "attempt" in origin and (
        type(origin["attempt"]) is not int or not 1 <= origin["attempt"] <= 10
    ):
        raise ValueError("retained dependency origin attempt is invalid")
    if "proposed_check_ordinal" in origin and (
        type(origin["proposed_check_ordinal"]) is not int
        or not 0 <= origin["proposed_check_ordinal"] < 512
    ):
        raise ValueError("retained dependency origin ordinal is invalid")
