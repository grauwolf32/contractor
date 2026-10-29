"""Canonical authoring codec for optional finding analysis evidence (FND-R191)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jcs
from jsonschema import Draft202012Validator

from contractor_runtime.strict_json import unique_json_object

MEDIA_TYPE = "application/vnd.contractor.finding-analysis+json"
_SCHEMA = json.loads(Path(__file__).with_name("finding_analysis.schema.json").read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA)


def canonical_analysis(text: str) -> bytes:
    """Validate the closed authoring shape and return bounded canonical evidence.

    Exact source coordinates, dependency membership and tool provenance are
    checked by the Server against the containing receipt during publication.
    """
    try:
        if len(text.encode("utf-8")) > 1024 * 1024:
            raise ValueError("finding analysis is too large")
        value = json.loads(text, object_pairs_hook=unique_json_object)
        pending: list[tuple[Any, int]] = [(value, 1)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > 32 or nodes > 100000:
                raise ValueError("finding analysis exceeds its structural limit")
            if isinstance(item, dict):
                pending.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str) and ("\0" in item or len(item.encode("utf-8")) > 16384):
                raise ValueError("finding analysis text is invalid")
        if not _VALIDATOR.is_valid(value):
            raise ValueError("finding analysis shape is invalid")
        encoded = jcs.canonicalize(value)
        if len(encoded) > 1024 * 1024:
            raise ValueError("finding analysis is too large")
        return encoded
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        # Never return schema exceptions containing private evidence values.
        raise ValueError("finding analysis is invalid") from error
