"""Standard-library regressions for the actual managed Eval producer."""

import copy
import json
import runpy
import sys
import tempfile
import urllib.parse
from pathlib import Path
from unittest.mock import patch


def check(producer, root):
    fixture = runpy.run_path(str(producer))
    encoded = fixture["encoded"]
    shared = json.loads((root / "api/testdata/audit-common/results.json").read_text())
    example = shared["cases"][0]
    context = json.loads(example["context"])
    context["workflow_ref"] = "audit-source-check@3"
    source_items = fixture["audit_assertion_source"]()["items"]
    template = context["items"][0]
    outcomes, accepted, tasks = [], [], {}
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
        item["evidence"][0]["ref"].update(namespace="inputs", name="source")
        context["items"].append(item)
        outcome = copy.deepcopy(json.loads(example["document"])["items"][0])
        outcome.update(
            item_key=item["item_key"],
            evidence=item["evidence"],
        )
        retained = copy.deepcopy(item["evidence"][0])
        retained.update(scope_kind="audit", scope_id="audit-1")
        retained["ref"].update(namespace="audit-inputs", revision="audit-source-1")
        accepted.append(
            {
                **outcome,
                "evidence": [retained],
                "origin": {
                    "audit_id": "audit-1",
                    "run_id": "run-1",
                    "task_id": item["task_id"],
                    "hypothesis": {"id": hypothesis["id"], "revision": 1},
                },
            }
        )
        outcomes.append(outcome)
        tasks[item["task_id"]] = {
            "document": {
                "schema": "contractor.audit.task.v1",
                "audit_id": "audit-1",
                "task_id": item["task_id"],
                "workflow": "check",
                "objective": item["objective"],
                "parameters": {},
                "inputs": {"source": retained},
                "hypothesis": {"id": hypothesis["id"], "revision": 1},
            },
            "hypothesis_snapshot": hypothesis,
            "state": "settled",
            "iteration": 1,
            "attempts": 1,
            "last_execution_id": "execution-1",
        }
    document = {"schema": "contractor.audit.conclusions.v1", "items": outcomes}
    wrapper = {"schema": "contractor.audit.result.v1", "items": accepted}
    wrapper_raw = encoded(wrapper)
    retained_context = copy.deepcopy(context)
    for item in retained_context["items"]:
        item["evidence"] = list(tasks[item["task_id"]]["document"]["inputs"].values())
    for task in tasks.values():
        task["result"] = {
            "scope_kind": "audit",
            "scope_id": "audit-1",
            "ref": {
                "namespace": "audit-results",
                "name": "result",
                "revision": "task-result-1",
            },
            "digest": fixture["digest"](wrapper_raw),
            "media_type": "application/json",
            "size_bytes": len(wrapper_raw),
        }
    # Preserve noncanonical whitespace in the actual bytes and their digest.
    raw = json.dumps(document, indent=2).encode() + b"\n"
    result_ref = {"namespace": "outputs", "name": "result", "revision": "result-1"}
    context_ref = {"namespace": "inputs", "name": "context", "revision": "context-1"}
    paths = {
        "/v1/runs/run-1/artifacts/inputs/context?revision=context-1": encoded(context),
        "/v1/runs/run-1/artifacts/outputs/result?revision=result-1": raw,
    }

    def wrapper_path(task_id, revision="task-result-1"):
        return "/v1/audits/audit-1/artifacts?" + urllib.parse.urlencode(
            {"key": f"task/{task_id}/result", "revision": revision}
        )

    for task_id in tasks:
        paths[wrapper_path(task_id)] = wrapper_raw
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
            "intentId": "execution-1",
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
            self.tasks = copy.deepcopy(tasks)
            self.run = {
                "inputs": {
                    "context": context_ref,
                    "source": copy.deepcopy(context["items"][0]["evidence"][0]["ref"]),
                },
                "outputs": {"result": result_ref},
            }

        def get(self, path):
            self.reads.append(path)
            if path.startswith("/v1/audits/audit-1/tasks/"):
                return self.tasks[path.rsplit("/", 1)[1]]
            return {
                "/v1/audits/audit-1": {"projectId": "eval-project"},
                "/v1/audits/audit-1/report": self.report,
                "/v1/runs/run-1": self.run,
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
    assert len(api.reads) == 8 and set(paths) <= set(api.reads)

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
        "raw-result-wrapper",
        "forged-raw-origin",
        "missing-wrapper-item",
        "wrapper-conclusions",
        "wrapper-hypothesis",
        "wrapper-source",
        "wrapper-hash",
        "wrapper-size",
        "wrapper-revision",
        "task-hypothesis",
        "task-original-source",
        "task-execution",
        "raw-input-binding",
    ):
        broken = API()
        broken_inventory = copy.deepcopy(inventory)
        value = copy.deepcopy(document)
        saved = copy.deepcopy(wrapper)
        first_task = broken.tasks["task-0"]
        if change == "missing-item":
            value["items"].pop()
        elif change == "foreign-origin":
            saved["items"][0]["origin"]["run_id"] = "foreign-run"
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
        elif change == "raw-result-wrapper":
            value = copy.deepcopy(wrapper)
        elif change == "forged-raw-origin":
            value["items"][0]["origin"] = copy.deepcopy(wrapper["items"][0]["origin"])
        elif change == "missing-wrapper-item":
            saved["items"].pop()
        elif change == "wrapper-conclusions":
            saved = copy.deepcopy(document)
        elif change == "wrapper-hypothesis":
            saved["items"][0]["origin"]["hypothesis"]["revision"] = 2
        elif change == "wrapper-source":
            saved["items"][0]["evidence"][0]["ref"]["revision"] = "foreign-source"
        elif change == "task-hypothesis":
            first_task["document"]["hypothesis"]["revision"] = 2
        elif change == "task-original-source":
            first_task["document"]["inputs"]["source"]["digest"] = "sha256:" + "b" * 64
        elif change == "task-execution":
            first_task["last_execution_id"] = "foreign-execution"
        elif change == "raw-input-binding":
            broken.run["inputs"]["source"]["revision"] = "foreign-revision"
        saved_bytes = encoded(saved)
        broken.paths[wrapper_path("task-0")] = saved_bytes
        first_task["result"].update(
            digest=fixture["digest"](saved_bytes), size_bytes=len(saved_bytes)
        )
        if change == "wrapper-hash":
            first_task["result"]["digest"] = "sha256:" + "b" * 64
        elif change == "wrapper-size":
            first_task["result"]["size_bytes"] += 1
        elif change == "wrapper-revision":
            first_task["result"]["ref"]["revision"] = "obsolete-revision"
        broken.paths[list(paths)[1]] = encoded(value)
        try:
            fixture["result_document"](broken, experiment, row, broken_inventory)
        except (AssertionError, KeyError):
            rejected += 1
        else:
            raise AssertionError(("producer accepted invalid evidence", change))
    assert rejected == 24
    invalid_conclusions = []
    for kind in (
        "old-result-wrapper",
        "forged-origin",
        "wrong-key",
        "foreign-evidence",
    ):
        invalid = copy.deepcopy(document)
        if kind == "old-result-wrapper":
            invalid = copy.deepcopy(wrapper)
        elif kind == "forged-origin":
            invalid["items"][0]["origin"] = copy.deepcopy(wrapper["items"][0]["origin"])
        elif kind == "wrong-key":
            invalid["items"][0]["item_key"] = "invented"
        else:
            invalid["items"][0]["evidence"][0]["scope_id"] = "foreign-run"
        invalid_conclusions.append(invalid)
    invalid_wrappers = []
    for kind in (
        "raw-conclusions",
        "wrong-origin",
        "wrong-hypothesis",
        "foreign-source",
        "missing-item",
    ):
        invalid = copy.deepcopy(wrapper)
        if kind == "raw-conclusions":
            invalid = copy.deepcopy(document)
        elif kind == "wrong-origin":
            invalid["items"][0]["origin"]["task_id"] = "foreign-task"
        elif kind == "wrong-hypothesis":
            invalid["items"][0]["origin"]["hypothesis"]["revision"] = 2
        elif kind == "foreign-source":
            invalid["items"][0]["evidence"][0]["ref"]["revision"] = "foreign-source"
        else:
            invalid["items"].pop()
        invalid_wrappers.append(invalid)
    return {
        "result": result,
        "context": context,
        "document": document,
        "rejections": rejected,
        "dataset": audit,
        "registration": registration,
        "wrapper": wrapper,
        "retainedContext": retained_context,
        "tasks": list(tasks.values()),
        "invalidConclusions": invalid_conclusions,
        "invalidWrappers": invalid_wrappers,
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2]))))
