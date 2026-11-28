"""Named RPC adapters for existing scripted domain fixtures.

The request recorder keeps its historical tuple shape for domain assertions;
production uses RPCConnections and the native interop suites cover it.
"""

import json
from urllib.parse import quote, urlencode

from contractor_runtime._bulk_rpc import PayloadDownload
from contractor_runtime.artifacts import ArtifactTransportError
from contractor_runtime.strict_json import strict_json_loads
from fakes.artifact_responses import _strong_etag, require_json
from fakes.artifact_responses import raise_for_status as raise_rpc_status


class AllocationRPCFixture:
    async def read_artifact(self, allocation_id, value, *, max_response_bytes):
        response = await self.request(
            "GET",
            self._target(allocation_id, value, ""),
            headers={"Accept": "*/*"},
            body=b"",
            max_response_bytes=max_response_bytes,
        )
        raise_rpc_status(response)
        return PayloadDownload(
            {
                "apiVersion": "contractor/v1alpha1",
                "artifact": {**value["artifact"], "revision": _strong_etag(response.headers)},
                "mediaType": response.headers.get("content-type", ""),
                "size": len(response.body),
                "bindingCreatedAt": response.headers.get("x-contractor-binding-created-at", ""),
                "revisionCreatedAt": response.headers.get("x-contractor-revision-created-at", ""),
            },
            response.body,
        )

    async def write_artifact(self, allocation_id, value, data):
        response = await self.request(
            "PUT",
            self._target(allocation_id, value, ""),
            headers={
                "Accept": "application/json",
                "Content-Type": value["mediaType"],
                **(
                    {"If-None-Match": "*"}
                    if value.get("createOnly")
                    else {"If-Match": '"' + value["expectedRevision"] + '"'}
                ),
            },
            body=data,
            max_response_bytes=1 << 20,
        )
        raise_rpc_status(response)
        require_json(response.headers)
        result = strict_json_loads(response.body)
        if result.get("artifact", {}).get("revision") != _strong_etag(response.headers):
            raise ArtifactTransportError("fixture write returned inconsistent metadata")
        return {
            **result,
            "bindingCreatedAt": response.headers.get("x-contractor-binding-created-at", ""),
            "revisionCreatedAt": response.headers.get("x-contractor-revision-created-at", ""),
        }

    async def submit_finding(self, allocation_id, submission):
        value = {
            key: item
            for key, item in submission.header.items()
            if key not in {"proposalSize", "derivedAnchorsSize", "httpEvidenceSize"}
        }
        names = {1: "proposal", 2: "derived_anchors", 3: "httpEvidence"}
        value.update({names[kind]: strict_json_loads(data) for kind, data in submission.documents})
        return await self._rpc_json(
            "POST", f"/allocations/{allocation_id}/finding-proposals", value
        )

    async def submit_audit_evidence(self, allocation_id, value):
        return await self._rpc_json("POST", f"/allocations/{allocation_id}/audit-evidence", value)

    async def _rpc_json(self, method, path, value=None, headers=None, maximum=1 << 20):
        response = await self.request(
            method,
            path,
            headers=headers or {"Content-Type": "application/json"},
            body=b""
            if value is None
            else json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(),
            max_response_bytes=maximum,
        )
        raise_rpc_status(response)
        require_json(response.headers)
        try:
            return strict_json_loads(response.body)
        except ValueError:
            raise ArtifactTransportError("invalid fixture RPC response") from None

    async def list_artifacts(self, allocation_id, value):
        return await self._rpc_json(
            "GET", f"/allocations/{allocation_id}/artifacts?" + urlencode(value)
        )

    @staticmethod
    def _target(allocation_id, value, suffix):
        ref = value["artifact"]
        query = {key: value[key] for key in ("scope", "operation") if key in value}
        query.update({key: item for key, item in ref.items() if key == "revision"})
        path = (
            f"/allocations/{allocation_id}/artifacts/"
            f"{quote(ref['namespace'], safe='')}/{quote(ref['name'], safe='')}{suffix}"
        )
        return path + ("?" + urlencode(query) if query else "")

    async def read_artifact_metadata(self, allocation_id, value):
        return await self._rpc_json("GET", self._target(allocation_id, value, "/metadata"))

    async def check_artifact_access(self, allocation_id, value):
        response = await self.request(
            "GET",
            self._target(allocation_id, value, "/access"),
            headers={},
            body=b"",
            max_response_bytes=1 << 20,
        )
        raise_rpc_status(response)
        if response.status_code != 204 or response.body:
            raise ArtifactTransportError("invalid access response")
        return {}

    async def _sitemap_rpc(self, allocation_id, value, operation):
        body = {key: item for key, item in value.items() if key != "idempotencyKey"}
        return await self._rpc_json(
            "POST",
            f"/allocations/{allocation_id}/sitemap/{operation}",
            body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Idempotency-Key": value["idempotencyKey"],
            },
            maximum=16 << 10,
        )

    async def mutate_sitemap(self, allocation_id, value):
        return await self._sitemap_rpc(allocation_id, value, "mutations")

    async def snapshot_sitemap(self, allocation_id, value):
        return await self._sitemap_rpc(allocation_id, value, "snapshots")

    async def collect_sitemap(self, allocation_id, value):
        return await self._sitemap_rpc(allocation_id, value, "observations")

    async def list_findings(self, allocation_id, value):
        return await self._rpc_json(
            "GET", f"/allocations/{allocation_id}/findings?" + urlencode(value), maximum=256 << 10
        )

    async def read_finding(self, allocation_id, value):
        return await self._rpc_json(
            "GET",
            f"/allocations/{allocation_id}/findings/{value['findingId']}?revision={value['revision']}",
            maximum=256 << 10,
        )

    async def select_findings(self, allocation_id, value):
        return await self._rpc_json(
            "POST", f"/allocations/{allocation_id}/finding-selections", value, maximum=256 << 10
        )
