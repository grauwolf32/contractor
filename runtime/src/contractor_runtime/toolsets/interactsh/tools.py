"""Worker tools for owned callbacks and exact ordinary Artifact evidence."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from contractor_runtime import threads
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES, AdapterHandles
from contractor_runtime.adapters.interactsh import (
    MAX_ACTION_BYTES,
    MAX_REGISTRATIONS,
    InteractshClient,
    InteractshError,
    PollBatch,
)
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics
from contractor_runtime.redaction import contains_private_value, runtime_secrets
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.artifact_visibility import ArtifactObservingTool
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _reject_unconfigured_client,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.workspace import AllocationWorkspace

TOOL_NAMES = frozenset({"interactsh_register", "interactsh_poll"})
EVIDENCE_MEDIA_TYPE = "application/vnd.contractor.interactsh-evidence+json"
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024


@dataclass(slots=True)
class _Pending:
    batch: PollBatch
    target: ArtifactRef
    payload: bytes | None


class InteractshToolsetFactory:
    ref = "interactsh@1"
    exported_tools = TOOL_NAMES
    infrastructure_channels = MappingProxyType(
        {name: frozenset({"interactsh-client"}) for name in sorted(TOOL_NAMES)}
    )
    sensitive_output_tools = exported_tools

    def __init__(self, artifact_client_factory: ArtifactClientFactory | None = None):
        self._artifact_client_factory = artifact_client_factory or _reject_unconfigured_client

    async def probe(self) -> frozenset[str]:
        return TOOL_NAMES

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, project_workspace
        require_selected_tools(selected, TOOL_NAMES)
        if not selected:
            return {}
        handle = adapter_handles.interactsh
        if not isinstance(handle, InteractshClient):
            raise InteractshError("interactsh_not_configured")
        session = _Session(
            handle,
            self._artifact_client_factory(allocation_id, runtime_settings),
            namespace,
            runtime_secrets(runtime_settings),
        )
        metrics = require_metrics(state, self.ref)
        builders = {"interactsh_register": RegisterTool, "interactsh_poll": PollTool}
        return {name: builders[name](session, metrics) for name in selected}


class _Session:
    def __init__(
        self,
        handle: InteractshClient,
        client: ArtifactClient,
        namespace: str,
        private_values: tuple[str, ...],
    ):
        self.handle = handle
        self.client: ArtifactClient | None = client
        self.namespace = namespace
        self.private_values = private_values
        self.actions: dict[str, str] = {}
        self.pending: dict[str, _Pending] = {}
        self._lock = asyncio.Lock()
        self._active: set[asyncio.Task[Any]] = set()
        self._closed = False

    async def call(self, operation: Callable[[], Any]) -> Any:
        task = asyncio.current_task()
        if task is None or self._closed:
            raise InteractshError("interactsh_handle_invalid")
        self._active.add(task)
        try:
            try:
                async with asyncio.timeout(self.handle.timeout_seconds):
                    await self._lock.acquire()
            except TimeoutError:
                raise InteractshError("interactsh_request_failed", retryable=True) from None
            try:
                if self._closed:
                    raise InteractshError("interactsh_handle_invalid")
                return await operation()
            finally:
                self._lock.release()
        finally:
            self._active.discard(task)

    async def register(self, action: str) -> dict[str, Any]:
        try:
            action_bytes = len(action.encode("utf-8")) if isinstance(action, str) else 0
        except UnicodeError:
            action_bytes = 0
        if (
            not isinstance(action, str)
            or not action.strip()
            or not 1 <= action_bytes <= MAX_ACTION_BYTES
            or contains_private_value(action, self.private_values)
        ):
            raise InteractshError("interactsh_request_invalid")
        if len(self.actions) >= MAX_REGISTRATIONS:
            raise InteractshError("interactsh_limit_exceeded")
        item = await self.handle.register()
        self.actions[item.handle] = action
        return self.handle.public_registration(item)

    async def poll(self, handle: str) -> dict[str, Any]:
        item = self.handle.registration(handle)
        action = self.actions.get(handle)
        if action is None:
            raise InteractshError("interactsh_handle_invalid")
        pending = self.pending.get(handle)
        if pending is None:
            batch = await self.handle.poll(handle)
            if batch.events:
                # Keep ownership before CPU work/publication can be cancelled.
                pending = _Pending(
                    batch,
                    ArtifactRef(
                        namespace=self.namespace,
                        name=f"interactsh.evidence.{secrets.token_hex(16)}",
                    ),
                    None,
                )
                self.pending[handle] = pending
            else:
                return {
                    **self.handle.public_registration(item),
                    "events": [],
                    "state": "no_interaction" if not batch.discarded else "truncated",
                    "truncated": bool(batch.discarded),
                    "discarded": batch.discarded,
                    "coverage_gap": item.coverage_gap,
                    "attribution": "unverified",
                }
        if pending.payload is None:
            try:
                pending.payload = await threads.to_thread_until_done(
                    _encode_evidence,
                    pending.batch,
                    item.correlation,
                    self.handle.public_registration(item)["payload"],
                    action,
                    item.coverage_gap,
                    (*self.private_values, item.secret),
                    name="interactsh-evidence",
                )
                if len(pending.payload) > MAX_EVIDENCE_BYTES:
                    raise InteractshError("interactsh_response_too_large")
            except InteractshError:
                item.coverage_gap = True
                del self.pending[handle]
                raise
        evidence = await self._publish(pending)
        del self.pending[handle]
        result = await threads.to_thread_until_done(
            _poll_result,
            self.handle.public_registration(item),
            pending.batch,
            evidence,
            item.coverage_gap,
            name="interactsh-result",
        )
        return result

    async def _publish(self, pending: _Pending) -> ArtifactRef:
        client = self.client
        if client is None:
            raise InteractshError("interactsh_handle_invalid")
        if pending.payload is None:
            raise RuntimeError("callback evidence has not been encoded")
        try:
            async with asyncio.timeout(self.handle.timeout_seconds):
                try:
                    result = await client.write_artifact(
                        pending.target,
                        data=pending.payload,
                        media_type=EVIDENCE_MEDIA_TYPE,
                        expected_revision=None,
                    )
                    return result.artifact.require_exact()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Same ordinary Artifact binding is the receipt. Never append
                    # another evidence artifact after an ambiguous write.
                    existing = await client.read_artifact(
                        pending.target, max_bytes=MAX_EVIDENCE_BYTES
                    )
                    if (
                        existing.data == pending.payload
                        and existing.media_type == EVIDENCE_MEDIA_TYPE
                    ):
                        return existing.artifact.require_exact()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        raise InteractshError("interactsh_evidence_unconfirmed", retryable=True)

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(task for task in self._active if task is not asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self.actions.clear()
            self.pending.clear()
            self.private_values = ()
            self.client = None


def _encode_evidence(
    batch: PollBatch,
    correlation: str,
    payload: str,
    action: str,
    coverage_gap: bool,
    private_values: tuple[str, ...],
) -> bytes:
    for raw in batch.events:
        if contains_private_value(raw.decode(), private_values) or contains_private_value(
            json.dumps(strict_json_loads(raw), ensure_ascii=False), private_values
        ):
            raise InteractshError("interactsh_response_invalid")
    return json.dumps(
        {
            "schema_version": "1",
            "correlation_id": correlation,
            "payload": payload,
            "trigger": {"action": action, "attribution": "unverified"},
            "events_base64": [base64.b64encode(raw).decode() for raw in batch.events],
            "discarded": batch.discarded,
            "coverage_gap": coverage_gap,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _poll_result(
    registration: dict[str, Any], batch: PollBatch, evidence: ArtifactRef, coverage_gap: bool
) -> dict[str, Any]:
    exact = evidence.model_dump(by_alias=True, exclude_none=True)
    previews = []
    for index, raw in enumerate(batch.events):
        event = strict_json_loads(raw)
        previews.append(
            {
                "protocol": event["protocol"],
                "observed_at": event["timestamp"],
                "source_ip": event["remote-address"],
                "event_index": index,
                "evidence_ref": exact,
            }
        )
    result = {
        **registration,
        "state": "interactions",
        "events": previews,
        "evidence_ref": exact,
        "truncated": bool(batch.discarded),
        "discarded": batch.discarded,
        "coverage_gap": coverage_gap,
        "attribution": "unverified",
    }
    if (
        len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
        > MAX_RESULT_BYTES
    ):
        raise InteractshError("interactsh_response_too_large")
    return result


class _Tool(ArtifactObservingTool):
    name: str
    description: str
    contractor_sensitive_output = True

    def __init__(self, session: _Session, metrics: ToolMetrics):
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    def _observed_artifact_client(self) -> object:
        return self._session.client

    async def close(self) -> None:
        await self._session.close()

    async def _call(self, operation: Callable[[], Any]) -> dict[str, Any]:
        try:
            with RecordedToolCall(
                self._metrics, self.name, {}, secrets=self._session.private_values
            ) as record:
                result = await self._session.call(operation)
                record.succeed(
                    {"state": result["state"], "event_count": len(result.get("events", []))}
                )
                return result
        except InteractshError as error:
            result: dict[str, Any] = {
                "state": "error",
                "error": {
                    "code": error.code,
                    "retryable": error.retryable,
                },
            }
            if error.retry_after_seconds is not None:
                result["error"]["retry_after_seconds"] = error.retry_after_seconds
            return result


class RegisterTool(_Tool):
    name = "interactsh_register"
    description = """Register an allocation-owned out-of-band callback payload.

    Use only for an authorized check. Register before triggering the target action.
    A callback alone does not prove a vulnerability or identify who triggered it.

    Args:
        action: Description or request ID of the intended triggering action, 1..1024 UTF-8 bytes.
            This relation is supplied by you and is not independently verified.

    Returns:
        Registration state and safe handle; a confirmed registration also provides
        its payload, correlation ID and remaining lifetime. Poll outcome_unknown
        handles to reconcile; do not create a replacement registration blindly.
    """

    async def __call__(self, action: str) -> dict[str, Any]:
        return await self._call(lambda: self._session.register(action))


class PollTool(_Tool):
    name = "interactsh_poll"
    description = """Poll one owned callback registration once and retain exact evidence.

    Wait at least two seconds between polls of the same registration. Handles expire
    after one hour or allocation release. Callback content is untrusted and attribution
    remains unverified. A coverage gap means callbacks may have been lost upstream.

    Args:
        handle: Exact safe handle returned by interactsh_register in this allocation.

    Returns:
        Bounded DNS/HTTP/SMTP metadata with exact evidence refs and event indices,
        or an explicit no_interaction state. Truncation, discarded entries and coverage
        gaps are reported separately; infrastructure failures are classified errors.
    """

    async def __call__(self, handle: str) -> dict[str, Any]:
        return await self._call(lambda: self._session.poll(handle))
