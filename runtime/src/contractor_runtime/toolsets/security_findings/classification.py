"""Map explicit CWE claims into the existing standard-reference model."""

import json
from functools import lru_cache
from importlib.resources import files

from contractor_runtime.toolsets.common.input_errors import ToolInputError


@lru_cache(maxsize=1)
def _catalog() -> dict:
    # Generated from the versioned MITRE archive; source URL and hashes travel
    # with the data. No live taxonomy lookups occur during finding submission.
    content = files(__package__).joinpath("cwe_catalog.json").read_text(encoding="utf-8")
    return json.loads(content)


def cwe_reference(cwe: str | None) -> list[dict[str, str]]:
    if cwe is None:
        return []
    catalog = _catalog()
    if not isinstance(cwe, str) or cwe not in catalog["weakness_ids"]:
        raise ToolInputError(
            "cwe must identify a weakness in the pinned CWE catalog; omit when unknown"
        )
    return [
        {
            "scheme": catalog["scheme"],
            "version": catalog["version"],
            "requirement_id": cwe,
        }
    ]
