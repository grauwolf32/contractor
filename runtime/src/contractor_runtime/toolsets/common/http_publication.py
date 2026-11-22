"""Selected volatile HTTP captures published through ordinary Artifact CAS."""

import asyncio
import json

import jcs

from contractor_runtime import threads
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.artifacts import (
    HTTP_EXCHANGE_NAMESPACE,
    http_exchange_artifact_name,
)
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.common.artifact_write import (
    ArtifactReconciliationFailed,
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.http_capture import (
    HTTP_EXCHANGE_MEDIA_TYPE,
    HTTP_EXCHANGE_SCHEMA,
    HTTPExchangeArtifact,
)


class HTTPSelectionLimitError(ValueError):
    """The existing result evidence budget cannot admit the selected bytes."""


class HTTPSelectionUnavailableError(ValueError):
    """A selector has no retained capture owned by this invocation."""


class HTTPExchangePublisher:
    """No traffic or independent journal; only explicitly selected captures persist."""

    def __init__(self, client, captures, run_id, timeout):
        self.client, self.captures, self.run_id, self.timeout = client, captures, run_id, timeout

    async def prepare(self, invocation_id, request_ids, *, max_bytes):
        # Resolve the whole selection before writing anything; foreign/evicted
        # captures cannot cause partial selection or mutation of a result.
        try:
            captures = [
                (request_id, self.captures.select_observation(str(request_id), invocation_id)[0])
                for request_id in request_ids
            ]
        except ValueError:
            raise HTTPSelectionUnavailableError() from None
        if sum(len(raw) for _, raw in captures) > max_bytes:
            raise HTTPSelectionLimitError()
        selected = []
        used = 0
        for request_id, raw in captures:
            data = await threads.to_thread_until_done(
                lambda raw=raw: jcs.canonicalize(
                    {
                        "schema": HTTP_EXCHANGE_SCHEMA,
                        "origin": {"run_id": self.run_id, "invocation_id": invocation_id},
                        "capture": json.loads(raw),
                    }
                ),
                name="http-selection-encode",
            )
            await threads.to_thread_until_done(
                HTTPExchangeArtifact.model_validate_json, data, name="http-envelope-validate"
            )
            used += len(data)
            if used > max_bytes:
                raise HTTPSelectionLimitError()
            selected.append((request_id, data))
        return selected

    async def publish(self, invocation_id, request_id, data):
        target = ArtifactRef(
            namespace=HTTP_EXCHANGE_NAMESPACE,
            name=http_exchange_artifact_name(invocation_id, "http", str(request_id)),
        )
        async with asyncio.timeout(3 * self.timeout + 1):
            result = await write_artifact_exact(
                self.client,
                target,
                data=data,
                media_type=HTTP_EXCHANGE_MEDIA_TYPE,
                expected_revision=None,
                max_bytes=len(data),
                reject_first=lambda error: (
                    definite_write_rejection(error) and error.code != "artifact_conflict"
                ),
                reject_replay=lambda error: (
                    definite_write_rejection(error) and error.code != "artifact_conflict"
                ),
            )
        if (
            result.artifact.namespace != target.namespace
            or result.artifact.name != target.name
            or result.media_type != HTTP_EXCHANGE_MEDIA_TYPE
            or result.artifact.revision is None
        ):
            raise ArtifactReconciliationFailed()
        return jcs.canonicalize(
            {
                "scope_kind": "run",
                "scope_id": self.run_id,
                "ref": result.artifact.model_dump(exclude_none=True),
                "digest": sha256_digest(data),
                "media_type": HTTP_EXCHANGE_MEDIA_TYPE,
                "size_bytes": len(data),
            }
        )
