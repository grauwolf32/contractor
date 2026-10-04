"""Map explicit CWE claims into the existing standard-reference model."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from contractor_runtime.toolsets.common.input_errors import ToolInputError


@dataclass(frozen=True)
class _Catalog:
    scheme: str
    version: str
    weakness_ids: frozenset[str]


class CWEReferenceError(ToolInputError):
    """A CWE standard reference outside the pinned catalog; the reason is Runtime-authored."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"standard_refs: {reason}")
        self.reason = reason


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    # Generated from the versioned MITRE archive; source URL and hashes travel
    # with the data. No live taxonomy lookups occur during finding submission.
    content = files(__package__).joinpath("cwe_catalog.json").read_text(encoding="utf-8")
    document = json.loads(content)
    return _Catalog(document["scheme"], document["version"], frozenset(document["weakness_ids"]))


def cwe_reference(cwe: str | None) -> list[dict[str, str]]:
    if cwe is None:
        return []
    catalog = _catalog()
    if not isinstance(cwe, str) or cwe not in catalog.weakness_ids:
        raise ToolInputError(
            "cwe must identify a weakness in the pinned CWE catalog; omit when unknown"
        )
    return [
        {
            "scheme": catalog.scheme,
            "version": catalog.version,
            "requirement_id": cwe,
        }
    ]


def check_cwe_reference(reference: Mapping[str, str]) -> None:
    """Reject a CWE standard reference that the Server import would refuse.

    References of other schemes pass unchanged. Audit import accepts only the
    pinned catalog version and its non-deprecated weakness IDs.
    """
    catalog = _catalog()
    if reference["scheme"] != catalog.scheme:
        return
    if reference["version"] != catalog.version:
        raise CWEReferenceError(
            f"Use version {catalog.version} of the pinned CWE catalog, "
            "or pass the weakness ID as cwe."
        )
    requirement_id = reference["requirement_id"]
    if not isinstance(requirement_id, str) or requirement_id not in catalog.weakness_ids:
        raise CWEReferenceError(
            "Use a non-deprecated weakness ID from the pinned CWE catalog, "
            "or omit the CWE reference."
        )
