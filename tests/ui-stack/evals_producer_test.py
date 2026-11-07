"""Standard-library regressions for the actual managed Eval producer."""

import copy
import json
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def check(producer, root):
    fixture = runpy.run_path(str(producer))
    encoded = fixture["encoded"]
    shared = json.loads((root / "api/testdata/audit-common/results.json").read_text())
    example = shared["cases"][0]
    context = json.loads(example["context"])
    context["workflow_ref"] = "audit-source-check@1"
    source_items = fixture["audit_assertion_source"]()["items"]
    template = context["items"][0]
    outcomes = []
    context["items"] = []
    for index, source in enumerate(source_items):
        item = copy.deepcopy(template)
        item["item_key"] = f"item-{index:02d}"
        item["task_id"] = f"task-{index}"
        hypothesis = item["hypothesis"]
        hypothesis["id"] = f"hypothesis-{index}"
        hypothesis["statement"] = source["statement"]
        hypothesis["grounds"] = source["grounds"]
        hypothesis["scope"] = {
            **source["scope"],
            "source_version": source["version"],
            "applicability": source["applicability"],
            "allowed_methods": encoded(source["allowed_methods"]).decode(),
            "required_evidence": encoded(source["required_evidence"]).decode(),
        }
        item["evidence"][0].update(scope_kind="run", scope_id="run-1")
        context["items"].append(item)
        outcome = copy.deepcopy(example["expected"]["items"][0])
        outcome.update(
            item_key=item["item_key"],
            evidence=item["evidence"],
            origin={
                "audit_id": "audit-1",
                "run_id": "run-1",
                "task_id": item["task_id"],
                "hypothesis": {"id": hypothesis["id"], "revision": 1},
            },
        )
        outcomes.append(outcome)
    document = {"schema": "contractor.audit.result.v1", "items": outcomes}
    # Preserve noncanonical whitespace in the actual bytes and their digest.
    raw = json.dumps(document, indent=2).encode() + b"\n"
    result_ref = {"namespace": "outputs", "name": "result", "revision": "result-1"}
    context_ref = {"namespace": "inputs", "name": "context", "revision": "context-1"}
    paths = {
        "/v1/runs/run-1/artifacts/inputs/context?revision=context-1": encoded(context),
        "/v1/runs/run-1/artifacts/outputs/result?revision=result-1": raw,
    }
    inventory = [
        {
            "execution": {"kind": "audit", "id": "audit-1"},
            "parent": None,
            "state": "succeeded",
            "available": True,
        },
        {
            "execution": {"kind": "run", "id": "run-1"},
            "parent": {"kind": "audit", "id": "audit-1"},
            "workflow": "check",
            "iteration": 1,
            "state": "succeeded",
            "available": True,
        },
    ]
    sample = json.loads(
        (root / "api/testdata/evals/valid/result-audit.json").read_text()
    )
    row = {
        "execution": sample["execution"],
        "member": {"memberId": sample["memberId"]},
        "usage": sample["usage"],
    }
    experiment = {
        "planSha256": sample["planSha256"],
        "setup": {"source": sample["source"]},
    }

    class API:
        def __init__(self):
            self.reads = []
            self.report = {"status": "unavailable"}
            self.paths = paths.copy()

        def get(self, path):
            self.reads.append(path)
            return {
                "/v1/audits/audit-1": {"projectId": "eval-project"},
                "/v1/audits/audit-1/report": self.report,
                "/v1/runs/run-1": {
                    "inputs": {"context": context_ref},
                    "outputs": {"result": result_ref},
                },
            }[path]

        def get_bytes(self, path):
            self.reads.append(path)
            return self.paths[path]

    api = API()
    result = fixture["result_document"](api, experiment, row, inventory)
    assert result["outputs"] == {} and len(result["evidence"]) == 1
    evidence = result["evidence"][0]
    assert evidence["id"] == "task-results"
    assert evidence["artifact"] == {
        "scope": "run",
        "scopeId": "run-1",
        **result_ref,
        "sha256": fixture["digest"](raw),
        "mediaType": "application/json",
        "sizeBytes": len(raw),
    }
    assert result["usage"] == row["usage"]
    assert len(api.reads) == 4 and set(paths) <= set(api.reads)

    class Response:
        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return raw

    client = object.__new__(fixture["Client"])
    client.base, client.token, client.replays = "http://fixture.test", "fixture", 0
    with patch("urllib.request.urlopen", return_value=Response()):
        assert client.get_bytes("/artifact") == raw
        assert client.get("/artifact") == document

    # A current source-checklist has no aggregate output contract. Workflow
    # output checks and all private assessment content remain present.
    class SeedAPI:
        def request(self, method, path, body=None, **_kwargs):
            if method == "PUT":
                parts = path.split("/")
                return {
                    "artifact": {
                        "namespace": parts[-2],
                        "name": parts[-1],
                        "revision": "fixture-1",
                    }
                }
            if path == "/v1/projects":
                return {"projectId": "eval-project"}
            return body

    with tempfile.TemporaryDirectory() as directory:
        fixture["seed"](SeedAPI(), Path(directory))
        audit = json.loads((Path(directory) / "audit-dataset.json").read_text())
        workflow = json.loads((Path(directory) / "workflow-dataset.json").read_text())
        assert all(case["outputs"] == {} for case in audit["cases"])
        assert all(case["outputs"]["report"]["required"] for case in workflow["cases"])
        assert audit["privateChecks"] == workflow["privateChecks"]
        registration = fixture["register"](
            SeedAPI(), "eval-project", "audit", audit["cases"]
        )["registration"]
        assert len(registration["manifest"]["members"]) == fixture["MEMBERS"] == 8
        assert all(
            variant["outputMapping"] == {} for variant in registration["variants"]
        )

    rejected = 0
    for change in (
        "missing-item",
        "foreign-origin",
        "wrong-key",
        "wrong-verdict",
        "empty-conditions",
        "unbound-evidence",
        "foreign-context",
        "foreign-inventory",
        "extra-inventory",
        "pending-report",
        "unexpected-ready",
    ):
        broken = API()
        broken_inventory = copy.deepcopy(inventory)
        value = copy.deepcopy(document)
        if change == "missing-item":
            value["items"].pop()
        elif change == "foreign-origin":
            value["items"][0]["origin"]["run_id"] = "foreign-run"
        elif change == "wrong-key":
            value["items"][0]["item_key"] = "invented"
        elif change == "wrong-verdict":
            value["items"][0]["verdict"] = "refuted"
        elif change == "empty-conditions":
            value["items"][0]["tested_conditions"] = []
        elif change == "unbound-evidence":
            value["items"][0]["evidence"] = []
        elif change == "foreign-context":
            foreign = copy.deepcopy(context)
            foreign["audit_id"] = "foreign-audit"
            broken.paths[next(iter(paths))] = encoded(foreign)
        elif change == "foreign-inventory":
            broken_inventory[1]["parent"]["id"] = "foreign-audit"
        elif change == "extra-inventory":
            broken_inventory.append(copy.deepcopy(inventory[1]))
        elif change == "pending-report":
            broken.report = {"status": "pending"}
        elif change == "unexpected-ready":
            broken.report = {"status": "ready", "machineArtifact": {}}
        broken.paths[list(paths)[1]] = encoded(value)
        try:
            fixture["result_document"](broken, experiment, row, broken_inventory)
        except (AssertionError, KeyError):
            rejected += 1
        else:
            raise AssertionError(("producer accepted invalid evidence", change))
    assert rejected == 11
    return {
        "result": result,
        "context": context,
        "document": document,
        "rejections": rejected,
        "dataset": audit,
        "registration": registration,
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2]))))
