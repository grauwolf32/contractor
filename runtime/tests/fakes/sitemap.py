"""Scripted private responses; no target transport or collection behavior."""

import hashlib
import json
from urllib.parse import parse_qs, urlsplit

from contractor_runtime.artifacts import ArtifactTransportError
from fakes.allocation_rpc import AllocationRPCFixture
from fakes.artifact_responses import ArtifactHTTPResponse

ARTIFACT_DIGEST = "sha256:" + "a" * 64


def identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def document(groups: int = 3, members: int = 1, note_bytes: int = 0) -> dict:
    provenance = {"sourceId": "public", "kind": "http"}
    result = {
        "version": "project-sitemap@1",
        "projection": "sitemap-safe-projection@1",
        "groups": [],
        "sources": [{"provenance": provenance, "generation": 1, "policyDigest": "a" * 64}],
        "rules": [],
        "markers": [],
        "gaps": [{"sourceId": "", "reason": "projection", "count": 1}],
    }
    for index in range(groups):
        rows = []
        for member in range(members):
            path = f"/route/{index}/{member}"
            rows.append(
                {
                    "id": identity(path),
                    "path": path,
                    "summaries": [],
                    "examples": [],
                    "evidence": [],
                    "annotations": [
                        {"provenance": provenance, "note": "A" * note_bytes, "tags": []}
                    ]
                    if note_bytes and member == 0
                    else [],
                    "omitted": 0,
                }
            )
        result["groups"].append(
            {
                "id": identity(f"group-{index}"),
                "origin": "https://example.test",
                "method": "GET",
                "route": f"/route/{index}",
                "structural": [],
                "members": rows,
                "declarations": [],
                "omitted": 0,
            }
        )
    return result


class SitemapTransport(AllocationRPCFixture):
    def __init__(self, value: dict | None = None) -> None:
        self.document = value or document()
        self.revision = "revision-original"
        self.requests: list[tuple[str, str, dict, bytes]] = []
        self.access_operations: set[str] | None = None
        self.collect_observations = False
        self.access_error: tuple[int, str, bool] | None = None
        self.read_error: tuple[int, str, bool] | None = None
        self.post_error: tuple[int, str, bool] | None = None
        self.ambiguous_once = False

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.requests.append((method, path, dict(headers), body))
        route = urlsplit(path).path
        if method == "GET":
            query = parse_qs(urlsplit(path).query)
            assert route in {
                "/allocations/allocation/artifacts/sitemap/map",
                "/allocations/allocation/artifacts/sitemap/map/access",
            }
            assert query.get("scope") == ["project"]
            operation = query.get("operation", [None])[0]
            assert operation is not None
            if self.access_operations is not None and operation not in self.access_operations:
                return self.error((403, "artifact_access_denied", False))
            if self.access_error:
                return self.error(self.access_error)
            if route.endswith("/access"):
                return ArtifactHTTPResponse(204, {}, b"")
            if self.read_error:
                return self.error(self.read_error)
            return ArtifactHTTPResponse(
                200,
                {
                    "content-type": "application/vnd.contractor.project-sitemap+json",
                    "etag": f'"{self.revision}"',
                    "x-contractor-binding-created-at": "2026-10-01T12:00:00Z",
                    "x-contractor-revision-created-at": "2026-10-02T12:00:00Z",
                },
                json.dumps(self.document, separators=(",", ":")).encode(),
            )
        if self.ambiguous_once:
            self.ambiguous_once = False
            raise ArtifactTransportError("response lost")
        if self.post_error:
            return self.error(self.post_error)
        if route.endswith("/snapshots"):
            return self.json(
                {
                    "apiVersion": "contractor/v1alpha1",
                    "result": {
                        "artifact": {
                            "ref": {
                                "namespace": "research",
                                "name": "snapshot",
                                "revision": "saved-revision",
                            },
                            "digest": ARTIFACT_DIGEST,
                            "mediaType": "application/vnd.contractor.project-sitemap-context+json",
                            "sizeBytes": 10,
                        },
                        "replayed": False,
                    },
                }
            )
        return self.json(
            {
                "apiVersion": "contractor/v1alpha1",
                "result": {
                    "artifact": {
                        "namespace": "sitemap",
                        "name": "map",
                        "revision": "accepted-revision",
                    },
                    "replayed": False,
                    "changed": True,
                },
            }
        )

    @staticmethod
    def json(value):
        return ArtifactHTTPResponse(
            200, {"content-type": "application/json"}, json.dumps(value).encode()
        )

    @staticmethod
    def error(value):
        status, code, retryable = value
        return ArtifactHTTPResponse(
            status,
            {"content-type": "application/json"},
            json.dumps(
                {
                    "code": code,
                    "message": "operation rejected",
                    "retryable": retryable,
                    "requestId": "test-request",
                }
            ).encode(),
        )
