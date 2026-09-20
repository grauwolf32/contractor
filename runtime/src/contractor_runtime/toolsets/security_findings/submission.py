"""Normalize a finding submission and its exact evidence references."""

from __future__ import annotations

import hashlib
import re

from pydantic import ValidationError

from contractor_runtime.contracts import API_VERSION, ArtifactRef
from contractor_runtime.toolsets.common.input_errors import ToolInputError

FINDING_SCHEMA = "contractor.audit.finding-proposal.v1"

# Retained Audit domain limits; see docs/spec/27 and internal/auditdomain/types.go.
MAX_TEXT_BYTES = 64 * 1024
MAX_VALUES = 512
MAX_EVIDENCE = 256
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")

# The intake identity contract is 1-128 ASCII identifier bytes.
INVOCATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def build_submission(
    *,
    invocation_id: str,
    client_key: str,
    title: str,
    description: str,
    evidence_refs: list[dict[str, str]],
    standard_refs: list[dict[str, str]] | None,
) -> dict[str, object]:
    _identifier("client_key", client_key)
    _text("title", title, required=True)
    _text("description", description, required=True)
    if not isinstance(invocation_id, str) or INVOCATION_ID.fullmatch(invocation_id) is None:
        raise ValueError("Worker invocation identity is invalid")

    if not isinstance(evidence_refs, list) or len(evidence_refs) > MAX_EVIDENCE:
        raise ToolInputError("evidence_refs must be an array of at most 256 exact ArtifactRefs")
    refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str, str]] = set()
    for raw in evidence_refs:
        try:
            ref = ArtifactRef.model_validate(raw).require_exact()
        except (ValidationError, ValueError) as error:
            raise ToolInputError(
                "evidence_refs contains an invalid exact ArtifactRef; provide namespace, name "
                "and a non-empty revision"
            ) from error
        assert ref.revision is not None
        key = (ref.namespace, ref.name, ref.revision)
        if key in seen_refs:
            raise ToolInputError("evidence_refs contains a duplicate; include each reference once")
        seen_refs.add(key)
        refs.append(ref.model_dump(by_alias=True, exclude_none=True))
    refs.sort(key=lambda value: (value["namespace"], value["name"], value["revision"]))

    standards = _object_list(
        "standard_refs", standard_refs, {"scheme", "version", "requirement_id"}, MAX_VALUES
    )
    for reference in standards:
        _identifier("standard_refs.scheme", reference["scheme"])
        _text("standard_refs.version", reference["version"], required=True)
        _identifier("standard_refs.requirement_id", reference["requirement_id"])
    submission_digest = hashlib.sha256(
        b"contractor.finding.submission.v1\0"
        + invocation_id.encode("utf-8")
        + b"\0"
        + client_key.encode("utf-8")
    ).hexdigest()
    proposal: dict[str, object] = {
        "schema": FINDING_SCHEMA,
        "client_key": client_key,
        "title": title,
        "description": description,
        "subject": None,
        "preconditions": [],
        "standard_refs": standards,
        "evidence_ids": [f"evidence-{index + 1}" for index in range(len(refs))],
        "proposed_checks": [],
        "severity_suggestion": "",
        "limitations": [],
    }
    return {
        "apiVersion": API_VERSION,
        "invocationId": invocation_id,
        "submissionId": f"finding-{submission_digest}",
        "proposal": proposal,
        "evidenceRefs": refs,
    }


def _object_list(
    field: str,
    value: list[dict[str, str]] | None,
    keys: set[str],
    maximum: int,
) -> list[dict[str, str]]:
    result = [] if value is None else value
    if not isinstance(result, list) or len(result) > maximum:
        raise ToolInputError(f"{field} must be a JSON array of at most {maximum} objects")
    for item in result:
        if not isinstance(item, dict) or set(item) != keys:
            raise ToolInputError(f"{field} objects must contain only {', '.join(sorted(keys))}")
        if any(not isinstance(candidate, str) for candidate in item.values()):
            raise ToolInputError(f"{field} values must be strings")
    return result


def _identifier(field: str, value: str) -> None:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ToolInputError(
            f"{field} must be an identifier of at most 160 letters, digits, '.', '_', ':' or '-'; "
            "start with a letter or digit"
        )


def _text(field: str, value: str, *, required: bool) -> None:
    try:
        valid = (
            isinstance(value, str)
            and "\x00" not in value
            and len(value.encode("utf-8")) <= MAX_TEXT_BYTES
        )
    except UnicodeError:
        valid = False
    if not valid:
        raise ToolInputError(f"{field} must be UTF-8 text of at most 64 KiB without NUL characters")
    if required and not value.strip():
        raise ToolInputError(f"{field} must be non-empty text")
