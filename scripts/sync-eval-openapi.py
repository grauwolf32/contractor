#!/usr/bin/env python3
"""Copy managed Eval schemas into the public contract without remote references."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
spec_path = ROOT / "api/openapi/contractor-public-v1.yaml"
source = json.loads((ROOT / "api/evals/v1/managed.schema.json").read_text())["$defs"]

def refs(value, prefix="Eval"):
    if isinstance(value, dict):
        result = {key: (child.replace("#/$defs/", "#/components/schemas/Eval") if key == "$ref" else refs(child, prefix + key.title())) for key, child in value.items()}
        if "enum" in result and all(isinstance(v, str) for v in result["enum"]):
            result.setdefault("type", "string")
            result["x-enum-varnames"] = [prefix + "".join(part.title() for part in re.split(r"[^A-Za-z0-9]+", v)) for v in result["enum"]]
        return result
    if isinstance(value, list):
        return [refs(child, prefix + str(i)) for i, child in enumerate(value)]
    return value

def block(label, value):
    return f"    {label}: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"

schema_marker = "    # BEGIN GENERATED MANAGED EVAL SCHEMAS\n"
schema_end = "    # END GENERATED MANAGED EVAL SCHEMAS\n"
schemas = schema_marker + "".join(block("Eval" + name, refs(value,"Eval"+name)) for name, value in source.items())
schemas += block("EvalErrorDetails", {"type": "object", "additionalProperties": False, "required": ["kind", "recovery"], "properties": {"kind": {"type": "string", "const": "eval"}, "recovery": source["APIError"]["properties"]["recovery"]}}) + schema_end

routes = [
    ("get", "/v1/eval-capabilities", "getEvalCapabilities", None, "Capabilities", 200, False),
    ("get", "/v1/projects/{projectId}/eval-datasets", "listEvalDatasets", None, "DatasetPage", 200, False),
    ("post", "/v1/projects/{projectId}/eval-datasets", "importEvalDataset", "DatasetInput", "Dataset", 201, False),
    ("get", "/v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases", "listEvalCases", None, "CasePage", 200, False),
    ("post", "/v1/projects/{projectId}/eval-experiments", "createEvalExperiment", "CreateExperiment", "ExperimentReceipt", 201, False),
    ("get", "/v1/eval-experiments", "listEvalExperiments", None, "ExperimentPage", 200, False),
    ("get", "/v1/eval-experiments/{id}", "getEvalExperiment", None, "Experiment", 200, False),
    ("patch", "/v1/eval-experiments/{id}", "updateEvalDraft", "DraftUpdate", "ExperimentReceipt", 200, True),
    ("delete", "/v1/eval-experiments/{id}", "deleteEvalExperiment", "Delete", "ExperimentReceipt", 202, True),
    ("post", "/v1/eval-experiments/{id}/commands", "commandEvalExperiment", "Command", "CommandReceipt", 202, True),
    ("get", "/v1/eval-experiments/{id}/commands/{commandId}", "getEvalCommand", None, "CommandReceipt", 200, False),
    ("get", "/v1/eval-experiments/{id}/members", "listEvalMembers", None, "MemberPage", 200, False),
    ("post", "/v1/eval-experiments/{id}/members/{memberId}/submissions", "submitEvalMember", "Submission", "SubmissionReceipt", 202, False),
]

def response(schema, etag=False):
    result = {"description": "Owner-scoped evaluation response", "headers": {"X-Contractor-API-Version": {"$ref": "#/components/headers/ContractorAPIVersion"}, "X-Request-ID": {"$ref": "#/components/headers/RequestId"}, "Cache-Control": {"$ref": "#/components/headers/NoStore"}, "Idempotency-Replayed": {"$ref": "#/components/headers/IdempotencyReplayed"}}, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Eval" + schema}}}}
    if etag:
        result["headers"]["ETag"] = {"$ref": "#/components/headers/ETag"}
    return result

paths = {}
for method, path, operation, body, result, status, cas in routes:
    parameters = []
    for name in re.findall(r"\{([^}]+)\}", path):
        kind = "MemberID" if name == "memberId" else "Id" if name == "datasetId" else "Opaque"
        parameters.append({"name": name, "in": "path", "required": True, "schema": {"$ref": "#/components/schemas/Eval" + kind}})
    if operation.startswith("list") or operation == "getEvalCapabilities":
        parameters += [{"name": "limit", "in": "query", "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25}}, {"name": "cursor", "in": "query", "schema": {"type": "string", "minLength": 1, "maxLength": 8192}}]
    filters = ["kind"] if operation == "getEvalCapabilities" else []
    if operation == "listEvalExperiments":
        filters = ["projectId", "state", "datasetId", "controlMode"]
    if operation == "listEvalMembers":
        filters = ["viewSnapshot", "filter", "variantId"]
    for name in filters:
        schema = {"type": "string", "minLength": 1, "maxLength": 256}
        if name in ("state", "controlMode"):
            schema = source["Experiment"]["properties"][name]
        if name == "kind":
            schema = {"type": "string", "enum": ["workflow", "audit"]}
        if name == "filter":
            schema = {"type": "string", "enum": ["all", "unresolved", "failed", "unscored", "unsupported", "blocked", "conflicting"]}
        parameters.append({"name": name, "in": "query", "schema": refs(schema,"Eval"+operation.title()+name.title())})
    responses = {str(status): response(result, result in ("Experiment", "Dataset", "ExperimentReceipt") or operation == "commandEvalExperiment")}
    for code in [400, 401, 403, 404, 409, 412, 422, 500]:
        responses[str(code)] = {"$ref": f"#/components/responses/Error{code}"}
    for code, description in [(405, "Method is not allowed"), (428, "Evaluation revision precondition is required"), (503, "Evaluation service is unavailable")]:
        responses[str(code)] = {"description": description, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}
    if operation == "commandEvalExperiment":
        responses["201"] = response("ExperimentReceipt", True)
    op = {"operationId": operation, "tags": ["Evals"], "summary": operation, "x-contractor-implementation": "implemented", "parameters": parameters, "responses": responses}
    if body:
        parameters.append({"$ref": "#/components/parameters/IdempotencyKey"})
        if cas:
            parameters.append({"$ref": "#/components/parameters/RequiredIfMatch"})
        op["requestBody"] = {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Eval" + body}}}}
    paths.setdefault(path, {})[method] = op

path_marker = "  # BEGIN GENERATED MANAGED EVAL PATHS\n"
path_end = "  # END GENERATED MANAGED EVAL PATHS\n"
path_block = path_marker + "".join("  " + path + ": " + json.dumps(operations, separators=(",", ":")) + "\n" for path, operations in paths.items()) + path_end
text = spec_path.read_text()
for start, end, replacement, insertion in [(schema_marker, schema_end, schemas, "  schemas:\n"), (path_marker, path_end, path_block, "paths:\n")]:
    if start in text:
        text = text[:text.index(start)] + replacement + text[text.index(end) + len(end):]
    else:
        text = text.replace(insertion, insertion + replacement, 1)
error_ref = "            - {$ref: '#/components/schemas/EvalErrorDetails'}\n"
if error_ref not in text:
    text = text.replace("            - {$ref: '#/components/schemas/AuditProfileUnsupportedDetails'}\n", "            - {$ref: '#/components/schemas/AuditProfileUnsupportedDetails'}\n" + error_ref)
if "  - {name: Evals," not in text:
    text = text.replace("tags:\n", "tags:\n  - {name: Evals, description: Managed Workflow and Audit evaluations}\n", 1)
spec_path.write_text(text)
