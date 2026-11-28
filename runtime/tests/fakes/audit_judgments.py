"""Acknowledged private API fixtures; persistence invariants use real PostgreSQL."""

import json

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.audit_judgments import AuditJudgmentClient
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditResultEncoder


class JudgmentTransport:
    def __init__(self, inputs):
        self.context = json.loads(inputs.context)
        self.items = {}
        self.receipts = {}
        self.requests = []

    async def submit_audit_judgments(self, allocation_id, value):
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.requests.append(body)
        identity = value["submissionId"]

        def response(status, payload):
            if status == 409:
                return {
                    "stale": {
                        "apiVersion": payload["apiVersion"],
                        "currentGenerations": payload["currentGenerations"],
                    }
                }
            return {"receipt": payload}

        if identity in self.receipts:
            original, receipt = self.receipts[identity]
            assert original == body
            return response(200, {**receipt, "replayed": True})
        tasks = {item["item_key"]: item["task_id"] for item in self.context["items"]}
        prospective = dict(self.items)
        generations = []
        for update in value["items"]:
            conclusion = update["conclusion"]
            key = conclusion["item_key"]
            current = prospective.get(key)
            generation = current[0] if current else 0
            if not current or current[1] != conclusion:
                if update["baseGeneration"] != generation:
                    return response(
                        409,
                        {
                            "apiVersion": "contractor/v1alpha1",
                            "code": "audit_result_stale",
                            "message": "Changed",
                            "retryable": False,
                            "requestId": "fixture",
                            "currentGenerations": [
                                {
                                    "itemKey": entry["conclusion"]["item_key"],
                                    "taskId": tasks[entry["conclusion"]["item_key"]],
                                    "generation": self.items.get(
                                        entry["conclusion"]["item_key"], (0, None)
                                    )[0],
                                }
                                for entry in value["items"]
                            ],
                        },
                    )
                generation += 1
                prospective[key] = (generation, conclusion)
            generations.append({"itemKey": key, "generation": generation, "taskId": tasks[key]})
        receipt = {
            "apiVersion": "contractor/v1alpha1",
            "material": {
                "scope_kind": "audit",
                "scope_id": self.context["audit_id"],
                "ref": {"namespace": "audit-fixture", "name": identity, "revision": "saved-1"},
                "digest": sha256_digest(body),
                "media_type": "application/json",
                "size_bytes": len(body),
            },
            "generations": generations,
            "replayed": False,
        }
        self.items = prospective
        self.receipts[identity] = (body, receipt)
        return response(201, receipt)


def acknowledged_judgments(inputs, encoder=None):
    return AuditJudgmentClient(
        ArtifactClient(inputs.owner.allocation_id, JudgmentTransport(inputs)),
        encoder or CanonicalAuditResultEncoder(),
        inputs,
    )
