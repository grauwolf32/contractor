"""Independent schema conformance for the common Audit boundary.

This does not register Runtime completion support. The existing completion
assembler is replaced only with its production execution caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from contractor_runtime.contracts.base import _require_selector
from contractor_runtime.strict_json import strict_json_loads

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "api/audit/v1/contract.schema.json").read_text())
COMMON = json.loads((ROOT / "api/v1alpha1/common.schema.json").read_text())
REGISTRY = Registry().with_resources(
    (doc["$id"], Resource.from_contents(doc)) for doc in (SCHEMA, COMMON)
)
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, registry=REGISTRY)


def _decode(data: str):
    if not 0 < len(data.encode("utf-8")) <= 8 << 20:
        raise ValueError("document byte bound")
    value = strict_json_loads(data)
    VALIDATOR.validate(value)

    def bytes_and_depth(v, depth=1):
        if depth > 64:
            raise ValueError("depth")
        if type(v) is float:
            raise ValueError("integer JSON token required")
        if isinstance(v, str) and ("\x00" in v or len(v.encode("utf-8")) > 65536):
            raise ValueError("string bytes")
        if isinstance(v, dict):
            for key, item in v.items():
                if (
                    key == "parameters"
                    and isinstance(item, dict)
                    and any(
                        len(text.encode()) > 4096 for text in item.values() if isinstance(text, str)
                    )
                ):
                    raise ValueError("parameter bytes")
                if (
                    key == "limitations"
                    and isinstance(item, list)
                    and any(len(text.encode()) > 512 for text in item if isinstance(text, str))
                ):
                    raise ValueError("limitation bytes")
                bytes_and_depth(key, depth + 1)
                bytes_and_depth(item, depth + 1)
        elif isinstance(v, list):
            for item in v:
                bytes_and_depth(item, depth + 1)

    bytes_and_depth(value)
    if value["schema"] == "contractor.audit.context.v1":
        _require_selector("workflow_ref", value["workflow_ref"])
    if value["schema"] == "contractor.audit.program.v1":
        for binding in value["workflows"].values():
            _require_selector("ref", binding["ref"])
    if value["schema"] == "contractor.audit.hypothesis.v1" and len(data.encode()) > 32768:
        raise ValueError("hypothesis bytes")
    if value["schema"] == "contractor.audit.mutation.v1" and len(data.encode()) > 1 << 20:
        raise ValueError("mutation bytes")
    if value["schema"] == "contractor.audit.program.v1" and len(data.encode()) > 4 << 20:
        raise ValueError("program bytes")
    return value


def _validate_case(case):
    value = _decode(case["document"])
    if value["schema"] == "contractor.audit.program.v1":
        bindings, limits = value["workflows"], value["limits"]
        keys = [task["key"] for task in value["initial_work"]]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate work")
        for task in value["initial_work"]:
            for slot, name in task.get("inputs", {}).items():
                binding = bindings[task["workflow"]]
                if (
                    slot not in binding["inputs"]
                    or binding["inputs"][slot]["source"] != "audit-input"
                    or name not in value["inputs"]
                ):
                    raise ValueError("initial material selection")
            if task["workflow"] not in bindings or (
                "hypothesis" in task and task["hypothesis"] not in value["hypotheses"]
            ):
                raise ValueError("unknown task binding")
            if "hypothesis" in task:
                binding = bindings[task["workflow"]]
                if "result" not in binding["outputs"] or not any(
                    source["source"] == "task-context" for source in binding["inputs"].values()
                ):
                    raise ValueError("checking input/result")
        for binding in bindings.values():
            for source in binding["inputs"].values():
                if source["source"] == "audit-input" and source["name"] not in value["inputs"]:
                    raise ValueError("unknown material")
        if not limits["batch_size"] <= limits["max_tasks_per_iteration"] <= limits["max_tasks"]:
            raise ValueError("incoherent work budgets")
        if (
            limits["max_submitted_runs"]
            < (limits["max_tasks"] + limits["batch_size"] - 1) // limits["batch_size"]
        ):
            raise ValueError("incoherent Run budget")
        if limits["max_hypothesis_revisions"] < limits["max_hypotheses"]:
            raise ValueError("incoherent revision budget")
        if "continuation" in value and value["continuation"]["workflow"] not in bindings:
            raise ValueError("unknown continuation")
        return
    if value["schema"] in {"contractor.audit.control.v1", "contractor.audit.task.v1"}:
        return
    if value["schema"] == "contractor.audit.proposals.v1":
        trusted = _decode(case["context"])
        if trusted["schema"] != "contractor.audit.proposal-context.v1":
            raise ValueError("proposal context required")
        _validate_case({"document": json.dumps(trusted["program"])})
        program = trusted["program"]
        seeds = set(program["hypotheses"]) | set(value["hypotheses"])
        if set(program["hypotheses"]) & set(value["hypotheses"]):
            raise ValueError("seed collision")
        if len(value["hypotheses"]) + len(value["work"]) > 128:
            raise ValueError("proposal slice")
        seen = set()
        for task in value["work"]:
            if task["key"] in seen or task["workflow"] not in program["workflows"]:
                raise ValueError("work binding")
            seen.add(task["key"])
            binding = program["workflows"][task["workflow"]]
            for slot, name in task.get("inputs", {}).items():
                if (
                    slot not in binding["inputs"]
                    or binding["inputs"][slot]["source"] != "audit-input"
                    or name not in trusted["materials"]
                ):
                    raise ValueError("proposal material selection")
                declaration = program["inputs"][binding["inputs"][slot]["name"]]
                if (
                    "*/*" not in declaration["media_types"]
                    and trusted["materials"][name]["media_type"] not in declaration["media_types"]
                ):
                    raise ValueError("selected media type")
            if "hypothesis" in task:
                binding = program["workflows"][task["workflow"]]
                if (
                    task["hypothesis"] not in seeds
                    or "result" not in binding["outputs"]
                    or not any(
                        source["source"] == "task-context" for source in binding["inputs"].values()
                    )
                ):
                    raise ValueError("hypothesis binding")
        return
    context = _decode(case["context"])
    members = {member["item_key"]: member for member in context["items"]}
    if len(members) != len(context["items"]) or len(members) != len(value["items"]):
        raise ValueError("membership")
    seen = set()
    for item in value["items"]:
        key = item["item_key"]
        if key not in members or key in seen:
            raise ValueError("membership")
        seen.add(key)
        member = members[key]
        if (
            "hypothesis" in member
            and len(
                json.dumps(member["hypothesis"], ensure_ascii=False, separators=(",", ":")).encode()
            )
            > 32768
        ):
            raise ValueError("hypothesis snapshot bytes")
        if ("hypothesis" in member) != ("verdict" in item):
            raise ValueError("checking context")
        if item.get("verdict") in {"supported", "refuted"} and (
            not item["evidence"] or not item["tested_conditions"]
        ):
            raise ValueError("unsupported conclusion")
        if item.get("verdict") == "inconclusive" and not item["limitations"]:
            raise ValueError("missing gap")
        if any(evidence not in member["evidence"] for evidence in item["evidence"]):
            raise ValueError("foreign evidence")
        expected = dict(
            audit_id=context["audit_id"], run_id=context["run_id"], task_id=member["task_id"]
        )
        if "hypothesis" in member:
            expected["hypothesis"] = {key: member["hypothesis"][key] for key in ("id", "revision")}
        if value["schema"] == "contractor.audit.result.v1" and (
            item["origin"] != expected
            or item.get("finding_receipts") != member.get("finding_receipts")
        ):
            raise ValueError("forged provenance")


def test_shared_audit_common_documents():
    for name in ("programs.json", "results.json", "controls.json", "proposals.json", "tasks.json"):
        fixtures = json.loads((ROOT / "api/testdata/audit-common" / name).read_text())
        assert fixtures["cases"]
        for case in fixtures["cases"]:
            if case["valid"]:
                _validate_case(case)
            else:
                with pytest.raises((ValueError, jsonschema.ValidationError)):
                    _validate_case(case)


def test_shared_audit_common_boundaries():
    bounds = json.loads((ROOT / "api/testdata/audit-common/boundaries.json").read_text())
    raw = json.dumps(bounds["hypothesis"])
    _decode(raw.ljust(bounds["hypothesis_bytes"]))
    with pytest.raises(ValueError):
        _decode(raw.ljust(bounds["hypothesis_bytes"] + 1))
    records = [dict(bounds["hypothesis"], id=f"hypothesis-{i}") for i in range(128)]
    raw = json.dumps(dict(schema="contractor.audit.mutation.v1", records=records))
    _decode(raw.ljust(bounds["mutation_bytes"]))
    with pytest.raises(ValueError):
        _decode(raw.ljust(bounds["mutation_bytes"] + 1))
    records.append(bounds["hypothesis"])
    with pytest.raises(jsonschema.ValidationError):
        _decode(json.dumps(dict(schema="contractor.audit.mutation.v1", records=records)))
