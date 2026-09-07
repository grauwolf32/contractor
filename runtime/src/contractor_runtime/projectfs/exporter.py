"""Runtime-owned overlay persistence before graceful Worker results."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from contractor_runtime.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    AllocationWorkspaceExport,
    ArtifactRef,
    WorkerResult,
)
from contractor_runtime.projectfs.overlay import (
    MAX_WORKSPACE_EXPORT_BYTES,
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    OverlayWorkspaceSession,
    WorkspaceStateError,
)
from contractor_runtime.projectfs.storage import WorkspaceStorageError

WORKSPACE_DIFF_MEDIA_TYPE = "text/x-diff"
MAX_CAS_ATTEMPTS = 3
MAX_EXPORTED_RESULT_ARTIFACTS = 128
MAX_EXPORTED_RESULT_JSON_BYTES = 256 * 1024
SAFE_EXPORT_CAUSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class WorkspaceExportError(RuntimeError):
    """Stable, content-free failure at the pre-terminal export barrier."""

    code = "workspace_export_failed"

    def __init__(self, cause: str, *, retryable: bool) -> None:
        self.cause = cause if SAFE_EXPORT_CAUSE.fullmatch(cause) is not None else "export_error"
        self.retryable = retryable
        super().__init__(f"workspace export failed ({self.cause})")


@dataclass(frozen=True, slots=True)
class WorkspaceExportResult:
    result: WorkerResult
    state_bytes: int
    diff_bytes: int
    result_workspace_digest: str


class WorkspaceAutoExporter:
    """Write one cumulative state/diff pair and then advance the checkpoint."""

    def __init__(
        self,
        *,
        workspace: OverlayWorkspaceSession,
        client: ArtifactClient,
        namespace: str,
        slots: AllocationWorkspaceExport,
    ) -> None:
        self._workspace = workspace
        self._client = client
        self._namespace = namespace
        self._state_slot = slots.state
        self._diff_slot = slots.diff
        self._reserved_slots = frozenset({slots.state, slots.diff})

    @property
    def reserved_slots(self) -> frozenset[str]:
        return self._reserved_slots

    async def export(self, result: WorkerResult) -> WorkspaceExportResult:
        if self._reserved_slots & result.artifacts.keys():
            raise WorkspaceExportError("reserved_result_slot", retryable=False)
        try:
            bundle = await self._workspace.prepare_export(
                max_payload_bytes=min(MAX_ARTIFACT_BYTES, MAX_WORKSPACE_EXPORT_BYTES)
            )
            state_ref = await self._write_binding(
                self._state_slot,
                bundle.state,
                WORKSPACE_OVERLAY_MEDIA_TYPE,
            )
            diff_ref = await self._write_binding(
                self._diff_slot,
                bundle.diff,
                WORKSPACE_DIFF_MEDIA_TYPE,
            )
            final = self._inject_and_validate(result, state_ref, diff_ref)
            await self._workspace.commit_export(bundle)
        except WorkspaceExportError:
            raise
        except asyncio.CancelledError:
            raise
        except ArtifactAPIError as error:
            raise WorkspaceExportError(error.code, retryable=error.retryable) from None
        except ArtifactTransportError:
            raise WorkspaceExportError("artifact_transport", retryable=True) from None
        except WorkspaceStateError:
            raise WorkspaceExportError("workspace_state_invalid", retryable=False) from None
        except WorkspaceStorageError as error:
            cause = str(error.args[0]) if error.args else "workspace_unavailable"
            raise WorkspaceExportError(cause, retryable=False) from None
        except (TypeError, ValueError):
            raise WorkspaceExportError("invalid_export_result", retryable=False) from None
        except (OSError, TimeoutError):
            raise WorkspaceExportError("workspace_unavailable", retryable=True) from None
        except Exception:
            raise WorkspaceExportError("unexpected_export_failure", retryable=False) from None
        return WorkspaceExportResult(
            result=final,
            state_bytes=len(bundle.state),
            diff_bytes=len(bundle.diff),
            result_workspace_digest=bundle.result_workspace_digest,
        )

    async def _write_binding(self, name: str, data: bytes, media_type: str) -> ArtifactRef:
        target = ArtifactRef(namespace=self._namespace, name=name)
        for attempt in range(MAX_CAS_ATTEMPTS):
            expected_revision: str | None
            try:
                current = await self._client.read_artifact(target)
            except ArtifactAPIError as error:
                if error.status_code != 404 or error.code != "artifact_not_found":
                    raise
                expected_revision = None
            else:
                if current.media_type == media_type and current.data == data:
                    return current.artifact.require_exact()
                expected_revision = current.artifact.require_exact().revision
            try:
                written = await self._client.write_artifact(
                    target,
                    data=data,
                    media_type=media_type,
                    expected_revision=expected_revision,
                )
                return written.artifact.require_exact()
            except ArtifactAPIError as error:
                if error.code != "artifact_conflict" or attempt + 1 == MAX_CAS_ATTEMPTS:
                    raise
        raise WorkspaceExportError("artifact_conflict", retryable=True)

    def _inject_and_validate(
        self,
        result: WorkerResult,
        state_ref: ArtifactRef,
        diff_ref: ArtifactRef,
    ) -> WorkerResult:
        artifacts = dict(result.artifacts)
        artifacts[self._state_slot] = state_ref
        artifacts[self._diff_slot] = diff_ref
        if len(artifacts) > MAX_EXPORTED_RESULT_ARTIFACTS:
            raise WorkspaceExportError("invalid_export_result", retryable=False)
        final = WorkerResult.model_validate(
            {
                **result.model_dump(mode="python", by_alias=True, exclude_none=True),
                "artifacts": artifacts,
            }
        )
        if (
            len(final.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > MAX_EXPORTED_RESULT_JSON_BYTES
        ):
            raise WorkspaceExportError("invalid_export_result", retryable=False)
        return final
