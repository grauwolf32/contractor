"""Pinned anonymous preflight fixtures with no live target or configuration reads."""

from __future__ import annotations

import json

from contractor_runtime.contracts.engagement_allocation import EngagementAllocationProjection
from fakes.spec import allocation_spec

_PROJECTION = {
    "schema": "contractor.engagement.allocation.v1",
    "auditId": "audit-1",
    "engagement": {"namespace": "engagements", "name": "target", "revision": "r1"},
    "engagementDigest": "sha256:" + "a" * 64,
    "authorizedBy": "owner-1",
    "authorizedAt": "2026-10-28T12:00:00Z",
}


def preflight_spec(*, body_limit=262144):
    spec = allocation_spec()
    projection = {
        name: _PROJECTION[name]
        for name in (
            "schema",
            "auditId",
            "engagement",
            "engagementDigest",
            "authorizedBy",
            "authorizedAt",
        )
    }
    projection = json.loads(json.dumps(projection))
    projection["http"] = {
        "adapter": "http-target@1",
        "routes": [
            {
                "origin": "http://target.example",
                "pathPrefixes": ["/"],
                "methods": ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
                "addressRanges": ["10.20.0.0/16"],
            }
        ],
        "exclusions": [],
        "identities": [
            {"id": "anonymous", "kind": "anonymous", "origins": ["http://target.example"]}
        ],
        "protectedDestinations": [
            {"hostname": "control.example", "addresses": ["10.30.0.1"], "port": 443}
        ],
    }
    spec.engagement = EngagementAllocationProjection.model_validate_json(json.dumps(projection))
    return spec
