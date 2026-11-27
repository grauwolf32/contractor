"""Named RPC adapters for existing scripted domain fixtures.

The request recorder keeps its historical tuple shape for domain assertions;
production uses AllocationConnections and the native interop suites cover it.
"""

import json
from urllib.parse import quote, urlencode

import grpc

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactTransportError,
    raise_for_status,
    require_json,
)
from contractor_runtime.strict_json import strict_json_loads


def raise_rpc_status(response):
    try:
        raise_for_status(response)
    except ArtifactAPIError as error:
        status = {
            400: grpc.StatusCode.INVALID_ARGUMENT,
            401: grpc.StatusCode.UNAUTHENTICATED,
            403: grpc.StatusCode.PERMISSION_DENIED,
            404: grpc.StatusCode.NOT_FOUND,
            409: grpc.StatusCode.FAILED_PRECONDITION,
            412: grpc.StatusCode.FAILED_PRECONDITION,
            413: grpc.StatusCode.RESOURCE_EXHAUSTED,
            422: grpc.StatusCode.INVALID_ARGUMENT,
            429: grpc.StatusCode.RESOURCE_EXHAUSTED,
            500: grpc.StatusCode.INTERNAL,
            502: grpc.StatusCode.UNAVAILABLE,
            503: grpc.StatusCode.UNAVAILABLE,
            504: grpc.StatusCode.DEADLINE_EXCEEDED,
        }.get(error.status_code, grpc.StatusCode.UNKNOWN)
        if error.code == "artifact_transfer_capacity":
            status = grpc.StatusCode.RESOURCE_EXHAUSTED
        elif error.code == "storage_transaction_conflict":
            status = grpc.StatusCode.ABORTED
        raise ArtifactAPIError(status, error.code, error.retryable, error.request_id) from None


class AllocationRPCFixture:
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
