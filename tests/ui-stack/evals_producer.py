"""Disposable release fixture. Run with python -I -S: Python standard library only."""

import argparse
import hashlib
import http.client
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PRIVATE = "PRIVATE_MANAGED_EVAL_RELEASE_TRUTH"
OUTPUT = b"Deterministic managed evaluation source\n"
MEMBERS = 8
CHECK_DIGEST = "sha256:" + hashlib.sha256(b"release-fixture-check@1").hexdigest()


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class APIError(AssertionError):
    def __init__(self, method, path, status, body):
        super().__init__((method, path, status, body))
        self.status = status
        self.body = body


class Client:
    def __init__(self, base):
        self.base = base
        self.token = os.environ["EVAL_GATE_TOKEN"]
        self.replays = 0

    def request(
        self,
        method,
        path,
        body=None,
        *,
        key=None,
        revision=None,
        raw=None,
        media=None,
        lost=False,
        decode_json=True,
    ):
        data = raw if raw is not None else encoded(body) if body is not None else None
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": media or "application/json",
        }
        if key:
            headers["Idempotency-Key"] = key
        if revision is not None:
            headers["If-Match"] = f'"{revision}"'
        if method == "PUT":
            headers["If-None-Match"] = "*"
        if lost:
            # Close the connection after the Server commits and sends headers;
            # discard the entire response body, then replay exactly once.
            url = urllib.parse.urlsplit(self.base)
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            if response.status not in {200, 201, 202}:
                raise AssertionError(
                    (path, response.status, response.read().decode()[:500])
                )
            connection.close()
        request = urllib.request.Request(self.base + path, data, headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(16 * 1024 * 1024 + 1)
                assert len(payload) <= 16 * 1024 * 1024
                if lost:
                    assert response.headers.get("Idempotency-Replayed") == "true", path
                    self.replays += 1
                if decode_json and response.headers.get("Content-Type", "").startswith(
                    "application/json"
                ):
                    return json.loads(payload)
                return payload
        except urllib.error.HTTPError as error:
            raise APIError(
                method, path, error.code, error.read().decode()[:1000]
            ) from error

    def get(self, path):
        return self.request("GET", path)

    def get_bytes(self, path):
        return self.request("GET", path, decode_json=False)

    def pages(self, path, **query):
        for attempt in range(5):
            try:
                return self.stable_pages(path, **query)
            except APIError as error:
                # An ordinary execution may publish a newer view while its
                # producer polls. Start a whole new snapshot, never mix pages.
                if (
                    error.status != 409
                    or "eval_view_changed" not in error.body
                    or attempt == 4
                ):
                    raise
        raise AssertionError("pagination retry exhausted")

    def stable_pages(self, path, **query):
        items, seen, first = [], set(), None
        while True:
            page = self.get(path + "?" + urllib.parse.urlencode({"limit": 1, **query}))
            first = first or page
            for field in ("inventoryRevision", "viewSnapshot"):
                assert page.get(field) == first.get(field)
            items.extend(page["items"])
            if not page["page"]["hasMore"]:
                return items, first
            cursor = page["page"]["nextCursor"]
            assert cursor not in seen and len(seen) < 10000
            seen.add(cursor)
            query["cursor"] = cursor

    def wait(self, path, predicate):
        until = time.monotonic() + 180
        while time.monotonic() < until:
            value = self.get(path)
            if predicate(value):
                return value
            time.sleep(0.3)
        raise AssertionError(("deadline", path, value))


def upload(client, project, name, media, raw):
    value = client.request(
        "PUT",
        f"/v1/projects/{project}/artifacts/eval-fixtures/{name}",
        raw=raw,
        media=media,
    )
    return {
        "scope": "project",
        "scopeId": project,
        **value["artifact"],
        "mediaType": media,
        "sizeBytes": len(raw),
        "sha256": digest(raw),
    }


def audit_assertion_source():
    return {
        "schema": "contractor.audit.assertions.v1",
        "items": [
            {
                "key": "check-" + key,
                "version": "1",
                "statement": "Inspect fixture " + key,
                "scope": {},
                "grounds": "An authored deterministic fixture requirement",
                "applicability": "Fixture",
                "allowed_methods": ["static-trace"],
                "required_evidence": ["source-trace"],
            }
            for key in ("one", "two")
        ],
    }


def seed(client, directory):
    project = client.request(
        "POST",
        "/v1/projects",
        {
            "kind": "evaluation",
            "name": "Managed release gate",
            "description": "Disposable deterministic fixture",
        },
        key="eval-gate-project",
    )
    project_id = project["projectId"]
    source = upload(client, project_id, "plain-source", "text/plain", OUTPUT)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("app.py", "def fixture():\n    return True\n")
    audit_source = upload(
        client, project_id, "source-archive", "application/zip", archive.getvalue()
    )
    checklist_ref = upload(
        client,
        project_id,
        "checklist",
        "application/json",
        encoded(audit_assertion_source()),
    )
    datasets = {}
    for kind in ("workflow", "audit"):
        dataset = {
            "datasetId": "release-" + kind,
            "name": "Release " + kind,
            "cases": [],
            "privateChecks": [
                {
                    "id": "review",
                    "revision": "r1",
                    "rubric": "Review exact evidence",
                    "expected": {"fixture": PRIVATE},
                }
            ],
        }
        for identifier in ("case-one", "case-two"):
            dataset["cases"].append(
                {
                    "id": identifier,
                    "task": {
                        "kind": "check",
                        "objective": "Inspect retained fixture",
                        "parameters": {},
                    },
                    "inputs": {"source": source}
                    if kind == "workflow"
                    else {"source": audit_source, "checklist": checklist_ref},
                    "requires": [],
                    "outputs": {
                        "report": {
                            "mediaTypes": ["text/plain"],
                            "required": True,
                        }
                    }
                    if kind == "workflow"
                    else {},
                }
            )
        path = directory / (kind + "-dataset.json")
        path.write_text(json.dumps(dataset))
        datasets[kind] = {"file": str(path), "document": dataset}
    result = {"projectId": project_id, "datasets": datasets}
    (directory / "fixture.json").write_text(json.dumps(result))
    return {"projectId": project_id}


def register(client, project, kind, cases):
    identity = "release-external-" + kind
    variants = [
        {
            "id": arm,
            "kind": kind,
            "selector": f"eval-{'copy' if kind == 'workflow' else 'audit'}-{arm}@1",
            "outputMapping": {"report": "result"} if kind == "workflow" else {},
        }
        for arm in ("a", "b")
    ]
    source_hash = digest(
        encoded({"id": identity, "cases": cases, "variants": variants})
    )
    manifest = {
        "schema_version": "playground.public-projection/v1",
        "source_schema_version": "playground.plan/v1",
        "source_record_sha256": source_hash,
        "experiment_id": identity,
        "created_at": "2026-09-20T00:00:00Z",
        "members": [],
    }
    recipes = []
    for case in cases:
        for sample in (1, 2):
            for variant in variants:
                member = hashlib.sha256(
                    json.dumps(
                        [identity, "release", case["id"], sample, variant["id"]],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                manifest["members"].append(
                    {
                        "member_id": member,
                        "suite_id": "release",
                        "case_id": case["id"],
                        "sample": sample,
                        "variant_id": variant["id"],
                        "case_sha256": digest(encoded(case)),
                        "binding_sha256": digest(encoded(variant)),
                        "eligibility": "eligible",
                    }
                )
                recipes.append({"memberId": member, "case": case})
    registration = {
        "schemaVersion": "contractor.eval-registration/v1",
        "sourcePlanSha256": source_hash,
        "manifest": manifest,
        "source": {
            "system": "stdlib-gate",
            "id": identity,
            "revision": "r1",
            "sourceSha256": source_hash,
        },
        "variants": variants,
        "recipes": recipes,
        "checks": [
            {
                "id": "fixture",
                "evaluator": "fixture-check@1",
                "implementationSha256": CHECK_DIGEST,
                "required": True,
            }
        ],
        "comparison": {
            "baseline": "a",
            "candidate": "b",
            "requiredEqual": ["source", "tasks"],
            "allowedDifferences": ["instructions"],
            "gates": {"minCandidateEndToEndPass": 1, "maxQualityDrop": 0},
        },
        "budgets": {
            "maxMembers": MEMBERS,
            "maxInFlight": 1,
            "wallMs": 540000,
            "maxObservedTotalTokens": None,
        },
    }
    return client.request(
        "POST",
        f"/v1/projects/{project}/eval-experiments",
        {
            "name": "External release " + kind,
            "controlMode": "external",
            "registration": registration,
        },
        key=identity,
        lost=True,
    )


def audit_task_evidence(client, parent, inventory):
    # Two authored assertions are checked in one ordinary Run. An aggregate
    # report is optional. Correlate raw Run conclusions with both accepted
    # Task wrappers before using their original bytes as Eval evidence.
    assert len(inventory) == 2, inventory
    owner, child = inventory
    assert owner["execution"] == parent and owner["parent"] is None, owner
    assert owner["available"] and owner["state"] == "succeeded", owner
    assert child["parent"] == parent and child["execution"]["kind"] == "run", child
    assert child["available"] and child["state"] == "succeeded", child
    assert child["workflow"] == "check" and child["iteration"] == 1, child
    audit_id, run_id = parent["id"], child["execution"]["id"]
    report = client.get(f"/v1/audits/{audit_id}/report")
    assert report == {"status": "unavailable"}, report
    run = client.get(f"/v1/runs/{run_id}")

    def read(ref):
        query = urllib.parse.urlencode({"revision": ref["revision"]})
        path = f"/v1/runs/{run_id}/artifacts/{ref['namespace']}/{ref['name']}"
        return client.get_bytes(path + "?" + query)

    context = json.loads(read(run["inputs"]["context"]))
    assert context["schema"] == "contractor.audit.context.v1", context
    assert (context["audit_id"], context["run_id"]) == (audit_id, run_id), context
    assert context["workflow_ref"] == "audit-source-check@4", context
    raw = read(run["outputs"]["result"])
    conclusions = json.loads(raw)
    assert conclusions["schema"] == "contractor.audit.conclusions.v1", conclusions
    expected = {item["statement"]: item for item in audit_assertion_source()["items"]}
    assert len(context["items"]) == len(conclusions["items"]) == len(expected) == 2
    assert [item["item_key"] for item in context["items"]] == ["item-00", "item-01"]
    outcomes = {item["item_key"]: item for item in conclusions["items"]}
    assert set(outcomes) == {item["item_key"] for item in context["items"]}
    statements = set()
    tasks, normalized = {}, {}
    for item in context["items"]:
        outcome = outcomes[item["item_key"]]
        hypothesis = item["hypothesis"]
        assert hypothesis["schema"] == "contractor.audit.hypothesis.v1", hypothesis
        assert hypothesis["revision"] == 1, hypothesis
        assert hypothesis["origin"]["audit_id"] == audit_id, hypothesis
        source = expected[hypothesis["statement"]]
        statements.add(hypothesis["statement"])
        assert hypothesis["grounds"] == source["grounds"], hypothesis
        assert hypothesis["scope"] == {
            **source["scope"],
            "source_version": source["version"],
            "applicability": source["applicability"],
            "allowed_methods": encoded(source["allowed_methods"]).decode(),
            "required_evidence": encoded(source["required_evidence"]).decode(),
        }, hypothesis
        # A checking Run authors conclusions, never trusted origin. The public
        # Task owner exposes the accepted document and its normalized wrapper.
        assert "origin" not in outcome, outcome
        task = client.get(f"/v1/audits/{audit_id}/tasks/{item['task_id']}")
        document = task["document"]
        assert document["schema"] == "contractor.audit.task.v1", document
        assert (document["audit_id"], document["task_id"]) == (
            audit_id,
            item["task_id"],
        ), document
        assert document["workflow"] == "check", document
        assert document["objective"] == item["objective"], document
        pin = {"id": hypothesis["id"], "revision": hypothesis["revision"]}
        assert document["hypothesis"] == pin, document
        assert task["hypothesis_snapshot"] == hypothesis, task
        assert task["state"] == "settled" and task["iteration"] == 1, task
        assert task["attempts"] == 1 and not task.get("gap_code"), task
        assert task["last_execution_id"] == child["intentId"], task
        tasks[item["task_id"]] = task
        origin = {
            "audit_id": audit_id,
            "run_id": run_id,
            "task_id": item["task_id"],
            "hypothesis": pin,
        }
        assert outcome["verdict"] == "supported", outcome
        assert outcome["rationale"] and outcome["tested_conditions"], outcome
        assert outcome["evidence"] == item["evidence"] and outcome["evidence"], outcome
        assert outcome["limitations"] == [], outcome
        retained = []
        for material in outcome["evidence"]:
            assert (material["scope_kind"], material["scope_id"]) == (
                "run",
                run_id,
            ), material
            assert material["ref"]["namespace"] == "inputs", material
            assert material["ref"] == run["inputs"][material["ref"]["name"]], material
            original = document["inputs"][material["ref"]["name"]]
            assert (original["scope_kind"], original["scope_id"]) == (
                "audit",
                audit_id,
            ), original
            for field in ("digest", "media_type", "size_bytes"):
                assert material[field] == original[field], (material, original)
            if original not in retained:
                retained.append(original)
        normalized[item["item_key"]] = {
            **outcome,
            "origin": origin,
            "evidence": retained,
        }
    assert statements == set(expected), statements
    for task_id, task in tasks.items():
        material = task["result"]
        assert (material["scope_kind"], material["scope_id"]) == (
            "audit",
            audit_id,
        ), material
        assert material["media_type"] == "application/json", material
        query = urllib.parse.urlencode(
            {"key": f"task/{task_id}/result", "revision": material["ref"]["revision"]}
        )
        saved = client.get_bytes(f"/v1/audits/{audit_id}/artifacts?" + query)
        assert digest(saved) == material["digest"], material
        assert len(saved) == material["size_bytes"], material
        wrapper = json.loads(saved)
        assert wrapper["schema"] == "contractor.audit.result.v1", wrapper
        assert len(wrapper["items"]) == len(normalized) == 2, wrapper
        accepted = {item["item_key"]: item for item in wrapper["items"]}
        assert accepted == normalized, (accepted, normalized)
    return {
        "scope": "run",
        "scopeId": run_id,
        **run["outputs"]["result"],
        "sha256": digest(raw),
        "mediaType": "application/json",
        "sizeBytes": len(raw),
    }


def result_document(client, experiment, row, inventory):
    execution = row["execution"]
    parent = execution["ref"]
    if parent["kind"] == "run":
        remote = client.get("/v1/runs/" + parent["id"])
        ref = remote["outputs"]["result"]
        base = f"/v1/runs/{parent['id']}/artifacts/{ref['namespace']}/{ref['name']}"
        raw = client.get(base + "?revision=" + ref["revision"])
        assert raw == OUTPUT
        artifact = {
            "scope": "run",
            "scopeId": parent["id"],
            **ref,
            "sha256": digest(raw),
            "mediaType": "text/plain",
            "sizeBytes": len(raw),
        }
        outputs = {"report": artifact}
        evidence = [{"id": "report", "artifact": artifact}]
    else:
        artifact = audit_task_evidence(client, parent, inventory)
        outputs = {}
        evidence = [{"id": "task-results", "artifact": artifact}]
    assert row["usage"] is not None
    return {
        "schemaVersion": "contractor.eval-result-input/v1",
        "planSha256": experiment["planSha256"],
        "memberId": row["member"]["memberId"],
        "source": experiment["setup"]["source"],
        "execution": execution,
        "collection": {"status": "complete", "gaps": []},
        "outputs": outputs,
        "evidence": evidence,
        "usage": row["usage"],
        "previousResultSha256": None,
    }


def run_external(client, fixture):
    evidence = {}
    for kind in ("workflow", "audit"):
        experiment = register(
            client,
            fixture["projectId"],
            kind,
            fixture["datasets"][kind]["document"]["cases"],
        )
        path = "/v1/eval-experiments/" + experiment["experimentId"]
        experiment = client.wait(
            path, lambda value: value.get("freshness") == "current"
        )
        rows, _ = client.pages(path + "/members", filter="all")
        assert len(rows) == MEMBERS
        start = deadline = None
        for index, original in enumerate(rows):
            member = original["member"]["memberId"]
            prefix = path + "/members/" + member
            client.request(
                "POST",
                prefix + "/submissions",
                {"planSha256": experiment["planSha256"]},
                key="submit-" + member,
                lost=True,
            )
            current = client.get(path)
            if index == 0:
                start, deadline = current["startedAt"], current["deadlineAt"]
            assert (current["startedAt"], current["deadlineAt"]) == (start, deadline)
            until = time.monotonic() + 120
            while time.monotonic() < until:
                members, _ = client.pages(path + "/members", filter="all")
                row = next(
                    row for row in members if row["member"]["memberId"] == member
                )
                if (
                    row["execution"]["state"] in {"succeeded", "failed", "cancelled"}
                    and row["usage"]
                ):
                    break
                time.sleep(0.3)
            else:
                raise AssertionError(("terminal deadline", row))
            assert row["execution"]["state"] == "succeeded", row
            inventory, inventory_page = client.pages(prefix + "/executions")
            assert inventory_page["inventoryComplete"]
            result = result_document(client, experiment, row, inventory)
            receipt = client.request(
                "POST", prefix + "/results", result, key="result-" + member, lost=True
            )
            result_hash = receipt["recordSha256"]
            assessment = {
                "schemaVersion": "contractor.eval-assessment-input/v1",
                "resultSha256": result_hash,
                "source": {
                    "kind": "external",
                    "producerId": "stdlib-gate",
                    "recordSha256": digest(
                        encoded({"private": PRIVATE, "result": result_hash})
                    ),
                },
                "checks": [
                    {
                        "id": "fixture",
                        "evaluator": "fixture-check@1",
                        "implementationSha256": CHECK_DIGEST,
                        "status": "pass",
                        "reason": "Verified retained fixture",
                        "evidenceRefs": [item["id"] for item in result["evidence"]],
                    }
                ],
                "previousAssessmentSha256": None,
            }
            assessment_receipt = client.request(
                "POST",
                prefix + "/assessments",
                assessment,
                key="assessment-" + member,
                lost=True,
            )
            current = client.get(path)
            client.request(
                "POST",
                path + "/selections",
                {
                    "planSha256": experiment["planSha256"],
                    "selections": [
                        {
                            "memberId": member,
                            "resultSha256": result_hash,
                            "assessmentSha256": assessment_receipt["recordSha256"],
                        }
                    ],
                },
                key="select-" + member,
                revision=current["revision"],
            )
        current = client.get(path)
        command = client.request(
            "POST",
            path + "/commands",
            {"kind": "finalize", "planSha256": experiment["planSha256"]},
            key="finalize-" + kind,
            revision=current["revision"],
            lost=True,
        )
        client.wait(
            path + "/commands/" + command["commandId"],
            lambda value: value["state"] == "completed",
        )
        current = client.wait(
            path,
            lambda value: (
                value["state"] == "finished" and value.get("freshness") == "current"
            ),
        )
        report = client.get(path + "/report")
        assert PRIVATE not in json.dumps(report)
        assert current["expectedMembers"] == MEMBERS
        evidence[kind] = {
            "experimentId": experiment["experimentId"],
            "expectedMembers": MEMBERS,
            "startedAt": start,
            "deadlineAt": deadline,
            "report": report,
        }
    return {"experiments": evidence, "lostResponseReplays": client.replays}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seed", "external"))
    parser.add_argument("--api", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    client = Client(args.api)
    if args.action == "seed":
        result = seed(client, args.directory)
    else:
        fixture = json.loads((args.directory / "fixture.json").read_text())
        result = run_external(client, fixture)
        (args.directory / "external-evidence.json").write_text(json.dumps(result))
    print(json.dumps({"action": args.action, "replays": client.replays}))


if __name__ == "__main__":
    main()
