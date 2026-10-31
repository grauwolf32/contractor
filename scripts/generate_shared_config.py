#!/usr/bin/env python3
"""Bind shared policy to schemas/test data and incremental persistence checks.

Edit the owning Server constant, then run make shared-config-generate.
Historical SQL stays immutable; a policy change emits the next migration.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "api/openapi/contractor-public-v1.yaml"


def schema_bindings():
    paths = []
    for route in ("/v1/artifacts/{namespace}/{name}", "/v1/projects/{projectId}/artifacts/{namespace}/{name}"):
        base = ("paths", route, "put")
        paths += [base + ("x-max-request-body-bytes",), base + ("requestBody", "content", "*/*", "schema", "maxLength")]
    base = ("components", "responses", "ArtifactBytes")
    paths += [base + ("headers", "Content-Length", "schema", "maximum"), base + ("content", "*/*", "schema", "maxLength")]
    for name, field in (("SitemapExactArtifact", "sizeBytes"), ("GitImportResult", "size"), ("AuditExactArtifact", "sizeBytes"), ("AuditReportSummaryArtifact", "sizeBytes"), ("FindingExactArtifact", "sizeBytes"), ("ArtifactWriteResponse", "size"), ("ArtifactMetadata", "size")):
        paths.append(("components", "schemas", name, "properties", field, "maximum"))
    return {
        PUBLIC: paths,
        "api/v1alpha1/artifact-metadata.schema.json": [("properties", "size", "maximum")],
        # SitemapExactArtifact itself is copied from public by generate_sitemap_schema.
        "api/v1alpha1/project-sitemap.schema.json": [("$defs", "SitemapInputDescriptor", "properties", "sizeBytes", "maximum")],
    }


def replace_scalars(source, bindings, value):
    """Change only named scalar nodes; preserve human-authored YAML formatting."""
    nodes = {}

    def walk(node, path):
        if isinstance(node, yaml.MappingNode):
            for key, child in node.value:
                walk(child, path + (key.value,))
        elif isinstance(node, yaml.SequenceNode):
            for index, child in enumerate(node.value):
                walk(child, path + (str(index),))
        else:
            nodes[path] = node

    walk(yaml.compose(source), ())
    edits = []
    for path in bindings:
        node = nodes[path]  # A removed/renamed consumer must update its binding.
        if node.tag != "tag:yaml.org,2002:int":
            raise ValueError(f"non-integer shared policy binding: {path}")
        edits.append((node.start_mark.index, node.end_mark.index))
    for start, end in sorted(edits, reverse=True):
        source = source[:start] + str(value) + source[end:]
    return source


def bind_config_schema(source, schema):
    """The public projection uses the private owning schema, without its IDs."""
    node = yaml.compose(source)
    for key in ("components", "schemas", "SharedConfig"):
        node = next(child for name, child in node.value if name.value == key)
    schema = {key: value for key, value in schema.items() if key not in ("$schema", "$id", "title")}
    lines = json.dumps(schema, indent=2).splitlines()
    rendered = lines[0] + "\n" + "\n".join("    " + line for line in lines[1:])
    return source[:node.start_mark.index] + rendered + source[node.end_mark.index:]


def pagination_bindings(source):
    """Follow only operations declaring common Limit, not equal-valued domain policies."""
    document = yaml.safe_load(source)
    bindings = {("components", "parameters", "Limit", "schema", "maximum")}
    for item in document["paths"].values():
        for operation in item.values():
            if not isinstance(operation, dict):
                continue
            parameters = item.get("parameters", []) + operation.get("parameters", [])
            if not any(p.get("$ref") == "#/components/parameters/Limit" for p in parameters):
                continue
            for status, response in operation.get("responses", {}).items():
                if not str(status).startswith("2"):
                    continue
                if "$ref" in response:
                    response = document["components"]["responses"][response["$ref"].split("/")[-1]]
                schema = response.get("content", {}).get("application/json", {}).get("schema", {})
                if "$ref" not in schema:
                    raise ValueError("common list response needs a named page schema")
                name = schema["$ref"].split("/")[-1]
                properties = document["components"]["schemas"][name]["properties"]
                arrays = [key for key, value in properties.items() if value.get("type") == "array"]
                if len(arrays) != 1:
                    raise ValueError(f"common list {name} needs one page array")
                bindings.add(("components", "schemas", name, "properties", arrays[0], "maxItems"))
    return bindings


def outputs(root, config):
    cap = config["limits"]["artifactPayloadMaxBytes"]
    result = {}
    for name, bindings in schema_bindings().items():
        result[name] = replace_scalars((root / name).read_text(), bindings, cap)
    # Golden boundary cases and retained-history floor derive from the same
    # ceiling; telemetry and reclamation work limits are independent policies.
    for name, binding, value in (
        ("api/testdata/v1alpha1/valid/artifact-metadata-result-boundary.json", ("size",), cap),
        ("api/testdata/v1alpha1/artifact-history-cases.json", ("policy", "maxBytesFloor"), 2 * cap),
        ("api/testdata/http-capture-cases.json", ("limits", "submissionBytes"), cap),
    ):
        result[name] = replace_scalars((root / name).read_text(), [binding], value)
    public_cap = config["limits"]["publicPageMaxItems"]
    source = result[PUBLIC]
    result[PUBLIC] = replace_scalars(source, pagination_bindings(source), public_cap)
    result[PUBLIC] = replace_scalars(result[PUBLIC], [("components", "parameters", "Limit", "schema", "default")], min(50, public_cap))
    name = "api/testdata/v1alpha1/artifact-list-cases.json"
    result[name] = replace_scalars((root / name).read_text(), [("maxLimit",)], config["limits"]["artifactListMaxItems"])
    schema = json.loads((root / "api/v1alpha1/shared-config.schema.json").read_text())
    result[PUBLIC] = bind_config_schema(result[PUBLIC], schema)
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    result["api/testdata/v1alpha1/valid/shared-config.json"] = raw
    result["runtime/tests/testdata/shared-config.json"] = raw
    result["runtime/tests/shared_config_defaults.py"] = (
        '"""Generated by make shared-config-generate; test defaults only."""\n\n'
        f"MAX_ARTIFACT_BYTES = {cap}\n"
    )
    migrations = root / "internal/persistence/migrations"
    effective = None
    for path in sorted(migrations.glob("*.sql")):
        # Matches both the baseline and subsequent ALTER CHECK forms.
        matches = re.findall(r"artifact_blobs_size_check[\s\S]*?size_bytes\s*<=\s*(\d+)", path.read_text())
        if matches:
            effective = int(matches[-1])
    if effective is None:
        raise ValueError("Artifact size CHECK not found")
    if effective != cap:
        number = max(int(p.name.split("_")[0]) for p in migrations.glob("*.sql")) + 1
        name = f"internal/persistence/migrations/{number:06d}_shared_artifact_payload_limit.sql"
        result[name] = (
            "-- Generated by make shared-config-generate from contracts.MaxArtifactPayloadBytes.\n"
            "-- Keep historical migrations immutable. Existing oversized rows reject a decrease.\n"
            "ALTER TABLE artifact_blobs\n"
            "    DROP CONSTRAINT artifact_blobs_size_check,\n"
            "    ADD CONSTRAINT artifact_blobs_size_check\n"
            f"        CHECK (size_bytes >= 0 AND size_bytes <= {cap});\n"
        )
    return result


def generate(root, config, check=False):
    stale = []
    for name, content in outputs(root, config).items():
        path = root / name
        if not path.exists() or path.read_text() != content:
            stale.append(name)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    if check and stale:
        raise ValueError("stale shared config bindings; run make shared-config-generate: " + ", ".join(stale))
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(subprocess.check_output(["go", "run", "./tools/sharedconfig"], cwd=ROOT))
    generate(ROOT, config, args.check)


if __name__ == "__main__":
    main()
