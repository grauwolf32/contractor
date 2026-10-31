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
    for route in (
        "/v1/artifacts/{namespace}/{name}",
        "/v1/projects/{projectId}/artifacts/{namespace}/{name}",
    ):
        base = ("paths", route, "put")
        paths += [
            base + ("x-max-request-body-bytes",),
            base + ("requestBody", "content", "*/*", "schema", "maxLength"),
        ]
    base = ("components", "responses", "ArtifactBytes")
    paths += [
        base + ("headers", "Content-Length", "schema", "maximum"),
        base + ("content", "*/*", "schema", "maxLength"),
    ]
    for name, field in (
        ("SitemapExactArtifact", "sizeBytes"),
        ("GitImportResult", "size"),
        ("AuditExactArtifact", "sizeBytes"),
        ("AuditReportSummaryArtifact", "sizeBytes"),
        ("FindingExactArtifact", "sizeBytes"),
        ("ArtifactWriteResponse", "size"),
        ("ArtifactMetadata", "size"),
    ):
        paths.append(("components", "schemas", name, "properties", field, "maximum"))
    return {
        PUBLIC: paths,
        "api/v1alpha1/artifact-metadata.schema.json": [("properties", "size", "maximum")],
        # SitemapExactArtifact itself is copied from public by generate_sitemap_schema.
        "api/v1alpha1/project-sitemap.schema.json": [
            ("$defs", "SitemapInputDescriptor", "properties", "sizeBytes", "maximum")
        ],
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
    return source[: node.start_mark.index] + rendered + source[node.end_mark.index :]


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


def worker_bindings():
    return {
        "workerResultMaxBytes": {
            "api/v1alpha1/worker-completion.schema.json": [
                ("$defs", "workerResult", "properties", "result", "maxLength"),
                ("$defs", "workerResult", "properties", "result", "x-maxBytes"),
            ]
        },
        "workerCompletionMaxBytes": {
            "api/v1alpha1/worker-completion.schema.json": [("x-maxBytes",)]
        },
        "workerFailureMessageMaxBytes": {
            "api/v1alpha1/worker-completion.schema.json": [
                ("$defs", "workerFailure", "properties", "message", "maxLength"),
                ("$defs", "workerFailure", "properties", "message", "x-maxBytes"),
            ]
        },
        "workerResultMaxArtifacts": {
            "api/v1alpha1/worker-completion.schema.json": [
                ("$defs", "workerResult", "properties", "artifacts", "maxProperties")
            ]
        },
        "workerObservationMaxTools": {
            "api/v1alpha1/worker-completion.schema.json": [
                ("$defs", "workerObservations", "properties", "tools", "maxProperties")
            ]
        },
        "workerWorkspaceMaxFilesRead": {
            "api/v1alpha1/worker-completion.schema.json": [
                ("$defs", "workspaceObservationSummary", "properties", "filesRead", "maxItems")
            ]
        },
        "workerStateMaxBytes": {"api/v1alpha1/agent-state.schema.json": [("x-maxBytes",)]},
        "workerStateWorkspaceMaxPaths": {
            "api/v1alpha1/agent-state.schema.json": [
                ("$defs", "workspaceObservation", "properties", "scopePaths", "maxItems"),
                ("$defs", "workspaceObservation", "properties", "interactions", "maxItems"),
            ]
        },
        "workerStateWorkspacePathsMaxBytes": {
            "api/v1alpha1/agent-state.schema.json": [
                ("$defs", "workspaceObservation", "properties", "scopePaths", "x-maxBytes"),
                ("$defs", "workspaceObservation", "properties", "interactions", "x-paths-maxBytes"),
            ]
        },
    }


def worker_result_cases(root, limits):
    """Rebuild boundary recipes and byte expectations from the owning policy."""
    cases = json.loads((root / "api/testdata/v1alpha1/worker-result-size-cases.json").read_text())
    base = json.loads(
        (root / "api/testdata/v1alpha1/valid/worker-completion-empty-observations.json").read_text()
    )
    text_cap, envelope_cap = limits["workerResultMaxBytes"], limits["workerCompletionMaxBytes"]

    def encoded_size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())

    for case in cases:
        name = case["name"]
        if name.startswith("exact_integer"):
            repeat = 1
        elif name == "text_over_limit":
            repeat = text_cap + 1
        elif name == "text_at_limit":
            repeat = text_cap
        elif name.startswith("envelope_"):
            repeat = 1
        elif name.startswith("separator_envelope_"):
            repeat = max(1, (text_cap - len(case["prefix"].encode())) // len(case["unit"].encode()))
        elif name.startswith("literal_") or name == "mixed_escapes":
            repeat = max(1, text_cap // (2 * len(case["unit"].encode())))
        else:
            repeat = max(1, (text_cap - len(case["prefix"].encode())) // len(case["unit"].encode()))
        case["repeat"] = repeat
        value = json.loads(json.dumps(base))
        value["result"]["result"] = case["prefix"] + case["unit"] * repeat
        case["padding"] = 0
        if "envelope_" in name:
            value["result"]["artifacts"] = {
                "": {"namespace": "review", "name": "report", "revision": "r1"}
            }
            target = envelope_cap + int(name.endswith("over_limit"))
            padding = target - encoded_size(value)
            if padding < 1:
                raise ValueError("Worker completion ceiling cannot hold the boundary fixture")
            case["padding"] = padding
            value["result"]["artifacts"] = {
                "p" * padding: {"namespace": "review", "name": "report", "revision": "r1"}
            }
        if case["integer"] is not None:
            value["stateRevision"] = case["integer"]
            value["result"]["observations"]["tools"] = {
                "read_file": {"calls": case["integer"], "failures": 0}
            }
        case["json_bytes"] = encoded_size(value)
        case["valid"] = (
            len(value["result"]["result"].encode()) <= text_cap
            and case["json_bytes"] <= envelope_cap
            and (case["integer"] is None or case["integer"] <= 2**53 - 1)
        )
    return json.dumps(cases, ensure_ascii=False, indent=2) + "\n"


def input_bindings():
    public = ("components", "schemas")
    private = ("$defs", "runMetadataLabels")
    return {
        "gitImportUrlMaxBytes": {
            PUBLIC: [
                public + ("GitImportRequest", "properties", "repositoryUrl", key)
                for key in ("maxLength", "x-max-utf8-bytes")
            ]
        },
        "runMetadataMaxLabels": {
            PUBLIC: [public + ("RunMetadataLabels", "maxProperties")],
            "api/v1alpha1/allocation.schema.json": [private + ("maxProperties",)],
        },
        "runMetadataKeyMaxBytes": {
            PUBLIC: [public + ("RunMetadataLabels", "propertyNames", "allOf", "0", "maxLength")],
            "api/v1alpha1/allocation.schema.json": [
                private + ("propertyNames", "allOf", "0", "maxLength")
            ],
        },
        "runMetadataValueMaxBytes": {
            PUBLIC: [
                public + ("RunMetadataLabels", "additionalProperties", key)
                for key in ("maxLength", "x-max-utf8-bytes")
            ],
            "api/v1alpha1/allocation.schema.json": [
                private + ("additionalProperties", key) for key in ("maxLength", "x-maxBytes")
            ],
        },
        "auditMaxLabels": {PUBLIC: [public + ("AuditLabels", "allOf", "1", "maxProperties")]},
    }


def bind_input_policy(root, limits, result):
    for policy, files in input_bindings().items():
        for name, bindings in files.items():
            result[name] = replace_scalars(
                result.get(name, (root / name).read_text()), bindings, limits[policy]
            )
    # Selector envelopes and canonical provenance derive from different concrete encodings.
    source = result[PUBLIC]
    document = yaml.safe_load(source)
    selector_paths = []
    for route, item in document["paths"].items():
        for method, operation in item.items():
            if not isinstance(operation, dict):
                continue
            for index, parameter in enumerate(operation.get("parameters", [])):
                if (
                    parameter.get("schema", {}).get("items", {}).get("$ref")
                    == "#/components/schemas/RunMetadataLabelSelector"
                ):
                    selector_paths.append(
                        ("paths", route, method, "parameters", str(index), "schema", "maxItems")
                    )
    source = replace_scalars(source, selector_paths, limits["runMetadataMaxLabels"])
    for schema, cap in (
        ("GitSource", 3 * limits["gitImportUrlMaxBytes"]),
        (
            "RunMetadataLabelSelector",
            limits["runMetadataKeyMaxBytes"] + 1 + limits["runMetadataValueMaxBytes"],
        ),
    ):
        base = ("components", "schemas", schema)
        if schema == "GitSource":
            base += ("properties", "repositoryUrl")
        source = replace_scalars(
            source, [base + (key,) for key in ("maxLength", "x-max-utf8-bytes")], cap
        )
    result[PUBLIC] = source
    name = "api/testdata/public/utf8-byte-bound-cases.json"
    fixture = json.loads((root / name).read_text())
    rules = {
        "LIM-GIT-23": limits["gitImportUrlMaxBytes"],
        "LIM-GIT-24": 3 * limits["gitImportUrlMaxBytes"],
        "LIM-RUN-5": limits["runMetadataValueMaxBytes"],
        "LIM-RUN-11": limits["runMetadataKeyMaxBytes"] + 1 + limits["runMetadataValueMaxBytes"],
    }
    for field in fixture["fields"]:
        if field["rule"] in rules:
            field["maxBytes"] = rules[field["rule"]]
    result[name] = json.dumps(fixture, ensure_ascii=False, indent=2) + "\n"
    name = "api/testdata/v1alpha1/run-metadata-label-cases.json"
    cases = json.loads((root / name).read_text())
    for category, case_name, key, value in (
        ("valid", "maximum-ascii-value", "note", "a" * limits["runMetadataValueMaxBytes"]),
        ("invalid", "oversized-value", "purpose", "a" * (limits["runMetadataValueMaxBytes"] + 1)),
        ("valid", "maximum-ascii-key", "a" * limits["runMetadataKeyMaxBytes"], "v"),
        ("invalid", "oversized-key", "a" * (limits["runMetadataKeyMaxBytes"] + 1), "v"),
    ):
        case = next((c for c in cases[category] if c["name"] == case_name), None)
        if case is None:
            case = {"name": case_name}
            cases[category].append(case)
        case["value"] = {key: value}
    next(c for c in cases["invalid"] if c["name"] == "too-many")["value"] = {
        f"k{i:02d}": "v" for i in range(limits["runMetadataMaxLabels"] + 1)
    }
    result[name] = json.dumps(cases, sort_keys=True, separators=(",", ":")) + "\n"
    name = "api/testdata/v1alpha1/placement-label-cases.json"
    result[name] = replace_scalars(
        (root / name).read_text(), [("maxAuditLabels",)], limits["auditMaxLabels"]
    )


def metadata_routine_with_limits(routine, limits):
    patterns = [
        (r"(jsonb_object_keys\(labels\)\)\s*<=\s*)\d+", limits["runMetadataMaxLabels"]),
        (r"(octet_length\(entry.key\) NOT BETWEEN 1 AND )\d+", limits["runMetadataKeyMaxBytes"]),
        (r"(octet_length\(entry.value #>> '\{\}'\) > )\d+", limits["runMetadataValueMaxBytes"]),
    ]
    changed = routine
    for pattern, cap in patterns:
        changed, count = re.subn(pattern, lambda m: m[1] + str(cap), changed)
        if count != 1:
            raise ValueError("metadata SQL policy binding changed")
    return changed


def append_input_migrations(root, limits, result):
    paths = sorted((root / "internal/persistence/migrations").glob("*.sql"))
    latest_git = None
    routine = None
    for path in paths:
        sql = path.read_text()
        matches = re.findall(
            r"artifact_versions_git_repository_url_check[\s\S]*?octet_length\(git_repository_url\)[\s\S]*?(?:<=|BETWEEN\s+1\s+AND)\s*(\d+)",
            sql,
        )
        if matches:
            latest_git = int(matches[-1])
        matches = re.findall(
            r"CREATE(?: OR REPLACE)? FUNCTION contractor_valid_run_metadata_labels\([^;]*?[\s\S]*?^(?:END;|\$[_a-zA-Z0-9]*\$;)",
            sql,
            re.MULTILINE,
        )
        if matches:
            routine = matches[-1]
    if latest_git is None or routine is None:
        raise ValueError("Git/metadata persistence policy not found")
    numbers = [int(p.name.split("_")[0]) for p in paths]
    numbers += [
        int(Path(name).name.split("_")[0])
        for name in result
        if name.startswith("internal/persistence/migrations/")
    ]
    number = max(numbers) + 1
    cap = 3 * limits["gitImportUrlMaxBytes"]
    if latest_git != cap:
        result[f"internal/persistence/migrations/{number:06d}_shared_git_source_limit.sql"] = (
            "-- Generated by make shared-config-generate; preserve historical migrations.\n"
            "ALTER TABLE artifact_versions\n"
            "    DROP CONSTRAINT artifact_versions_git_repository_url_check,\n"
            "    ADD CONSTRAINT artifact_versions_git_repository_url_check\n"
            f"        CHECK (octet_length(git_repository_url) BETWEEN 1 AND {cap});\n"
        )
        number += 1
    changed = metadata_routine_with_limits(routine, limits)
    if changed != routine:
        changed = re.sub(r"^CREATE(?: OR REPLACE)? FUNCTION", "CREATE OR REPLACE FUNCTION", changed)
        result[f"internal/persistence/migrations/{number:06d}_shared_run_metadata_limits.sql"] = (
            "-- Generated by make shared-config-generate; preserve historical migrations.\n"
            + changed
            + "\n"
            "-- Revalidate existing rows, so a decrease cannot silently invalidate data.\n"
            "ALTER TABLE workflow_runs\n"
            "    DROP CONSTRAINT workflow_runs_metadata_labels_check,\n"
            "    ADD CONSTRAINT workflow_runs_metadata_labels_check\n"
            "        CHECK (contractor_valid_run_metadata_labels(metadata_labels));\n"
        )


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
    for policy, files in worker_bindings().items():
        for name, bindings in files.items():
            source = result.get(name, (root / name).read_text())
            result[name] = replace_scalars(source, bindings, config["limits"][policy])
    name = "api/testdata/http-capture-cases.json"
    result[name] = replace_scalars(
        result[name], [("limits", "requestBodyBytes")], config["limits"]["httpRequestBodyMaxBytes"]
    )
    name = "api/testdata/http-capture-cases.json"
    result[name] = replace_scalars(
        result[name],
        [("limits", "responseBodyBytes")],
        config["limits"]["httpResponseBodyMaxBytes"],
    )
    name = "api/testdata/http-capture-cases.json"
    result[name] = replace_scalars(
        result[name], [("limits", "headerBytes")], config["limits"]["httpHeaderBlockMaxBytes"]
    )
    public_cap = config["limits"]["publicPageMaxItems"]
    source = result[PUBLIC]
    result[PUBLIC] = replace_scalars(source, pagination_bindings(source), public_cap)
    result[PUBLIC] = replace_scalars(
        result[PUBLIC],
        [("components", "parameters", "Limit", "schema", "default")],
        min(50, public_cap),
    )
    name = "api/testdata/v1alpha1/artifact-list-cases.json"
    result[name] = replace_scalars(
        (root / name).read_text(), [("maxLimit",)], config["limits"]["artifactListMaxItems"]
    )
    result["api/testdata/v1alpha1/worker-result-size-cases.json"] = worker_result_cases(
        root, config["limits"]
    )
    bind_input_policy(root, config["limits"], result)
    schema = json.loads((root / "api/v1alpha1/shared-config.schema.json").read_text())
    result[PUBLIC] = bind_config_schema(result[PUBLIC], schema)
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    result["api/testdata/v1alpha1/valid/shared-config.json"] = raw
    result["runtime/tests/testdata/shared-config.json"] = raw
    result["runtime/tests/shared_config_defaults.py"] = (
        '"""Generated by make shared-config-generate; test defaults only."""\n\n'
        f"MAX_ARTIFACT_BYTES = {cap}\n"
        f"MAX_WORKER_RESULT_BYTES = {config['limits']['workerResultMaxBytes']}\n"
        f"MAX_WORKER_COMPLETION_BYTES = {config['limits']['workerCompletionMaxBytes']}\n"
        f"MAX_WORKER_FAILURE_MESSAGE_BYTES = {config['limits']['workerFailureMessageMaxBytes']}\n"
        f"MAX_WORKER_RESULT_ARTIFACTS = {config['limits']['workerResultMaxArtifacts']}\n"
        f"MAX_WORKER_OBSERVATION_TOOLS = {config['limits']['workerObservationMaxTools']}\n"
        f"MAX_WORKER_FILES_READ = {config['limits']['workerWorkspaceMaxFilesRead']}\n"
        f"MAX_AGENT_STATE_SNAPSHOT_BYTES = {config['limits']['workerStateMaxBytes']}\n"
        f"MAX_STATE_WORKSPACE_PATHS = {config['limits']['workerStateWorkspaceMaxPaths']}\n"
        f"MAX_STATE_WORKSPACE_PATH_BYTES = {config['limits']['workerStateWorkspacePathsMaxBytes']}\n"
        f"MAX_REQUEST_BODY_BYTES = {config['limits']['httpRequestBodyMaxBytes']}\n"
        f"MAX_RESPONSE_BODY_BYTES = {config['limits']['httpResponseBodyMaxBytes']}\n"
        f"MAX_HEADER_BYTES = {config['limits']['httpHeaderBlockMaxBytes']}\n"
        f"MAX_RUN_METADATA_LABELS = {config['limits']['runMetadataMaxLabels']}\n"
        f"MAX_RUN_METADATA_LABEL_KEY_BYTES = {config['limits']['runMetadataKeyMaxBytes']}\n"
        f"MAX_RUN_METADATA_LABEL_VALUE_BYTES = {config['limits']['runMetadataValueMaxBytes']}\n"
    )
    migrations = root / "internal/persistence/migrations"
    effective = None
    for path in sorted(migrations.glob("*.sql")):
        # Matches both the baseline and subsequent ALTER CHECK forms.
        matches = re.findall(
            r"artifact_blobs_size_check[\s\S]*?size_bytes\s*<=\s*(\d+)", path.read_text()
        )
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
    append_input_migrations(root, config["limits"], result)
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
        raise ValueError(
            "stale shared config bindings; run make shared-config-generate: " + ", ".join(stale)
        )
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(subprocess.check_output(["go", "run", "./tools/sharedconfig"], cwd=ROOT))
    generate(ROOT, config, args.check)


if __name__ == "__main__":
    main()
