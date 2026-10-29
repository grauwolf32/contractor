#!/usr/bin/env python3
"""Generate the Runtime wire reader schema from the public domain contract."""

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runtime/src/contractor_runtime/toolsets/sitemap/document_schema.json"


def generated() -> str:
    schemas = yaml.safe_load((ROOT / "api/openapi/contractor-public-v1.yaml").read_text())["components"]["schemas"]
    selected = {}

    def visit(value):
        if isinstance(value, dict):
            if "$ref" in value:
                name = value["$ref"].rsplit("/", 1)[-1]
                if name not in selected:
                    selected[name] = schemas[name]
                    visit(schemas[name])
            for key, child in value.items():
                if key != "$ref":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit({"$ref": "#/components/schemas/SitemapDocument"})
    return json.dumps({"generatedFrom": "api/openapi/contractor-public-v1.yaml", "schemas": selected}, sort_keys=True, indent=2) + "\n"


WIRE_OUTPUT = ROOT / "api/v1alpha1/project-sitemap.schema.json"


def generated_wire() -> str:
    # Private wrappers are authored in their private schema. Domain command
    # definitions are copied mechanically from the single public owner.
    document = json.loads(WIRE_OUTPUT.read_text())
    public = yaml.safe_load((ROOT / "api/openapi/contractor-public-v1.yaml").read_text())["components"]["schemas"]
    selected = {}

    def visit(value):
        if isinstance(value, dict):
            if "$ref" in value:
                name = value["$ref"].rsplit("/", 1)[-1]
                if name not in selected:
                    selected[name] = json.loads(json.dumps(public[name]))
                    visit(selected[name])
                value["$ref"] = "#/$defs/" + name
            if "x-max-utf8-bytes" in value:
                value["x-maxBytes"] = value.pop("x-max-utf8-bytes")
            for key, child in value.items():
                if key != "$ref":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit({"$ref": "#/components/schemas/SitemapMutation"})
    visit({"$ref": "#/components/schemas/SitemapGap"})
    # Public and allocation editor commands both derive source identity on the Server.
    document["$defs"].update(selected)
    # A batch reports each domain omission reason at most once. Derive these
    # constraints from the owning enum so private and public reasons cannot drift.
    gaps = document["$defs"]["SitemapCollectionRequest"]["properties"]["input"]["properties"]["gaps"]
    gaps["allOf"] = [
        {"contains": {"properties": {"reason": {"const": reason}}}, "minContains": 0, "maxContains": 1}
        for reason in selected["SitemapGap"]["properties"]["reason"]["enum"]
    ]
    return json.dumps(document, indent=2) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in ((OUTPUT, generated()), (WIRE_OUTPUT, generated_wire())):
        if args.check:
            if not path.exists() or path.read_text() != content:
                parser.error("sitemap schemas differ; run scripts/generate_sitemap_schema.py")
        else:
            path.write_text(content)
