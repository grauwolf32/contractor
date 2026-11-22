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
    if schema == "contractor.audit.proposals.v1" and len(data) > 1 << 20:
        raise ValueError("Audit proposal bytes")
    if schema == "contractor.audit.hypothesis.v1" and len(data) > 32768:
        raise ValueError("Audit hypothesis bytes")
    value = strict_json_loads(data.decode("utf-8"))
    nodes = 0

    def check_text(text):
        if "\x00" in text or len(text.encode("utf-8")) > 65536:
            raise ValueError("Audit document string")

    def check(item, depth=1):
        nonlocal nodes
        nodes += 1
        if nodes > 250000 or depth > 64:
            raise ValueError("Audit document structure")
        if isinstance(item, str):
            check_text(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                # LIM-ACT-3 excludes member names from the value budget;
                # their independent UTF-8/NUL bound still applies.
                check_text(key)
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
    if schema in {"contractor.audit.program.v1", "contractor.audit.proposals.v1"} and any(
        len(jcs.canonicalize(seed)) > 32768 for seed in value["hypotheses"].values()
    ):
        raise ValueError("Audit seed bytes")
    if schema == "contractor.audit.proposals.v1" and len(jcs.canonicalize(value)) > 1 << 20:
        raise ValueError("Audit canonical proposal bytes")
    if schema == "contractor.audit.program.v1":
        _validate_program(value)
    if schema == "contractor.audit.proposal-context.v1":
        originals = value["input_sources"]
        if set(originals) != set(value["materials"]):
            raise ValueError("Audit material membership")
        distinct = {jcs.canonicalize(m) for m in originals.values()}
        if len(distinct) > 32 or sum(map(len, distinct)) > 1 << 20:
            raise ValueError("Audit cumulative materials")
        if any(
            m["scope_kind"] != "run" or m["scope_id"] != value["run_id"]
            for m in value["materials"].values()
        ):
            raise ValueError("Audit local material authority")
        tasks = set()
        for item in value.get("observations", []):
            if item["task_id"] in tasks:
                raise ValueError("Audit observation membership")
            tasks.add(item["task_id"])
            if item["key"] not in value["materials"] and "gap" not in item:
                raise ValueError("Audit observation source")
        decode_audit_document(
            jcs.canonicalize(value["program"]), schema="contractor.audit.program.v1"
        )
    return value


def _validate_program(program):
    limits = program["limits"]
    if not (limits["batch_size"] <= limits["max_tasks_per_iteration"] <= limits["max_tasks"]):
        raise ValueError("Audit work budget")
    if (
        limits["max_submitted_runs"]
        < (limits["max_tasks"] + limits["batch_size"] - 1) // limits["batch_size"]
        or limits["max_hypothesis_revisions"] < limits["max_hypotheses"]
        or len(program["hypotheses"]) > limits["max_hypotheses"]
        or len(program["initial_work"]) > limits["max_tasks_per_iteration"]
        or len(jcs.canonicalize(program)) > 4 << 20
    ):
        raise ValueError("Audit program bounds")
    bindings = program["workflows"]
    parameters = program.get("parameters", {})
    used_parameters = set()
    for binding in bindings.values():
        for name in binding.get("parameter_inputs", {}).values():
            if name not in parameters:
                raise ValueError("Audit parameter alias")
            used_parameters.add(name)
        if binding.get("max_attempts", limits["max_attempts"]) > limits["max_attempts"]:
            raise ValueError("Audit binding attempt budget")
        for source in binding["inputs"].values():
            if source["source"] == "audit-input" and source["name"] not in program["inputs"]:
                raise ValueError("Audit input alias")
    if used_parameters != set(parameters):
        raise ValueError("Audit parameter has no Workflow binding")
    seen = set()
    for item in program["initial_work"]:
        if item["key"] in seen or item["workflow"] not in bindings:
            raise ValueError("Audit initial work alias")
        seen.add(item["key"])
        binding = bindings[item["workflow"]]
        if "hypothesis" in item and (
            item["hypothesis"] not in program["hypotheses"]
            or "result" not in binding["outputs"]
            or not any(source["source"] == "task-context" for source in binding["inputs"].values())
        ):
            raise ValueError("Audit initial assertion binding")
        for slot, handle in item.get("inputs", {}).items():
            source = binding["inputs"].get(slot, {})
            if (
                source.get("source") not in {"audit-input", "task-input"}
                or handle not in program["inputs"]
            ):
                raise ValueError("Audit initial material selection")
    if "continuation" in program and program["continuation"]["workflow"] not in bindings:
        raise ValueError("Audit continuation binding")


def validate_audit_material(value: dict) -> None:
    """Validate one normalized exact material using the shared generated contract."""
    _VALIDATOR.evolve(schema=_SCHEMA["$defs"]["artifact"]).validate(value)
    from contractor_runtime.contracts.artifacts import ArtifactRef

    ArtifactRef.model_validate(value["ref"]).require_exact()
