"""Strict common Audit completion documents, using the owning generated schema."""

import json
from importlib.resources import files

import jcs
from jsonschema import Draft202012Validator

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, _require_selector
from contractor_runtime.strict_json import strict_json_loads

MAX_DOCUMENT_BYTES = 8 << 20
_SCHEMA = json.loads(files(__package__).joinpath("audit_documents.schema.json").read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA)


def decode_audit_document(data: bytes, *, schema: str) -> dict:
    """Validate bounded JSON without fetching external schemas or granting authority."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_DOCUMENT_BYTES:
        raise ValueError("Audit document bytes")
    value = strict_json_loads(data.decode("utf-8"))
    nodes = 0

    def check(item, depth=1):
        nonlocal nodes
        nodes += 1
        if nodes > 250000 or depth > 64:
            raise ValueError("Audit document structure")
        if isinstance(item, str):
            if "\x00" in item or len(item.encode("utf-8")) > 65536:
                raise ValueError("Audit document string")
        elif isinstance(item, dict):
            for key, child in item.items():
                check(key, depth + 1)
                check(child, depth + 1)
                if (
                    key == "limitations"
                    and isinstance(child, list)
                    and any(not isinstance(text, str) or len(text.encode()) > 512 for text in child)
                ):
                    raise ValueError("Audit limitation bytes")
                if key == "hypothesis" and len(jcs.canonicalize(child)) > 32768:
                    raise ValueError("Audit hypothesis bytes")
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
        elif type(item) is int and not -MAX_EXACT_INTEGER <= item <= MAX_EXACT_INTEGER:
            raise ValueError("Audit integer bound")
        elif type(item) is float:
            raise ValueError("Audit integer token required")

    check(value)
    _VALIDATOR.validate(value)
    if value["schema"] != schema:
        raise ValueError("Audit document kind")
    if schema == "contractor.audit.context.v1":
        _require_selector("workflow_ref", value["workflow_ref"])
        for field in ("item_key", "task_id"):
            values = [item[field] for item in value["items"]]
            if len(values) != len(set(values)):
                raise ValueError("Audit context membership")
    return value
