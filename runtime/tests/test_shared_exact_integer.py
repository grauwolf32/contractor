"""The fixed exact-integer range, including decimal comparisons before rounding."""

import json
from pathlib import Path

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER
from contractor_runtime.toolsets.openapi.yaml_scalars import document_float, document_integer


def test_shared_exact_integer_boundary():
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads((root / "api/testdata/exact-integer-cases.json").read_text())
    assert fixture["maximum"] == MAX_EXACT_INTEGER
    assert fixture["cases"]
    for case in fixture["cases"]:
        assert (document_integer(case["text"]) is not None) is case["integer"], case
        assert (document_float(case["text"]) is not None) is case["float"], case

    # These declarative contracts retain literal bounds; hold each location
    # to the executable definition rather than adding another configurable value.
    schemas = {
        "common/finding_analysis.schema.json": [
            f"/$defs/Region/properties/{name}/maximum"
            for name in ("start_line", "end_line", "start_column", "end_column")
        ],
        "sitemap/document_schema.json": [
            f"/schemas/{name}/properties/{field}/maximum"
            for name, field in (
                ("SitemapGap", "count"),
                ("SitemapGroup", "omitted"),
                ("SitemapMember", "omitted"),
                ("SitemapSource", "generation"),
            )
        ],
    }
    for name, pointers in schemas.items():
        schema = json.loads((root / "runtime/src/contractor_runtime/toolsets" / name).read_text())
        for pointer in pointers:
            value = schema
            for token in pointer.strip("/").split("/"):
                value = value[token]
            assert value == fixture["maximum"], (name, pointer)
