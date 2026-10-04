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
                if response.headers.get("Content-Type", "").startswith(
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
    checklist = {
        "schema": "contractor.audit.checklist.v1",
        "items": [
            {
                "key": "check-" + key,
                "version": "1",
                "statement": "Inspect fixture " + key,
                "applicability": "Fixture",
                "allowed_methods": ["static-trace"],
                "required_evidence": ["source-trace"],
                "review_policy": "automatic",
            }
            for key in ("one", "two")
        ],
    }
    checklist_ref = upload(
        client, project_id, "checklist", "application/json", encoded(checklist)
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
                            "mediaTypes": [
                                "text/plain"
                                if kind == "workflow"
                                else "application/json"
                            ],
                            "required": True,
                        }
                    },
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
            "executionConfig": {},
            "outputMapping": {"report": "result" if kind == "workflow" else "report"},
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
    else:
        remote = client.get("/v1/audits/" + parent["id"])
        report = client.get("/v1/audits/" + parent["id"] + "/report")
        ref = report["machineArtifact"]
        artifact = {
            "scope": "project",
            "scopeId": remote["projectId"],
            **ref["ref"],
            "sha256": ref["digest"],
            "mediaType": ref["mediaType"],
            "sizeBytes": ref["sizeBytes"],
        }
        assert len(inventory) == 3, inventory
    assert row["usage"] is not None
    return {
        "schemaVersion": "contractor.eval-result-input/v1",
        "planSha256": experiment["planSha256"],
        "memberId": row["member"]["memberId"],
        "source": experiment["setup"]["source"],
        "execution": execution,
        "collection": {"status": "complete", "gaps": []},
        "outputs": {"report": artifact},
        "evidence": [{"id": "report", "artifact": artifact}],
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
        experiment = client.wait(path, lambda value: value.get("freshness") == "current")
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
                        "evidenceRefs": ["report"],
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
