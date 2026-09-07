"""Trusted Audit preparation and invocation-local completion strategy."""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import replace

from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.audit_completion_contracts import (
    MAX_COLLECTED_BYTES,
    MAX_PACKAGE_BYTES,
    MAX_REMINDERS,
    AuditInvocationOwner,
    AuditTrustedInputs,
    CompleteCompletion,
    ContinueCompletion,
    FailCompletion,
)
from contractor_runtime.audit_packages import _decode_task_input
from contractor_runtime.audit_result_collector import AuditCollectionError, InvocationAuditCollector
from contractor_runtime.audit_result_encoding import AuditResultError
from contractor_runtime.audit_result_publication import (
    DeterministicAuditResultPublisher,
    PublicationArtifactClient,
)
from contractor_runtime.contracts import (
    ArtifactRef,
    StageContentRequest,
    WorkerCompletionContract,
    WorkerCompletionDiagnostics,
    WorkerFailure,
    WorkerObservations,
    WorkerResult,
)


class PreparedAuditCompletion:
    """Allocation-owned exact assignment; only the active invocation owns drafts."""

    def __init__(
        self,
        contract: WorkerCompletionContract,
        inputs: AuditTrustedInputs,
        client: PublicationArtifactClient,
    ):
        self.contract = contract.model_copy(deep=True)
        self._inputs = inputs
        self._client = client
        self._collector: InvocationAuditCollector | None = None
        self._reminders = 0
        self.diagnostics = None
        self.diagnostics_sink = None
        self.phase_sink = None

    def reset_diagnostics(self):
        self._reminders = 0
        self.diagnostics = WorkerCompletionDiagnostics(
            kind="audit-check-results@1",
            phase="collecting",
            acceptedCount=0,
            totalCount=len(self._inputs.owner.item_keys),
            reminderCount=0,
        )
        self._record_diagnostics()

    def _record_diagnostics(self):
        if self.diagnostics_sink is not None:
            self.diagnostics_sink(self.diagnostics)

    def record_progress(self, accepted_count):
        self.diagnostics = self.diagnostics.model_copy(update={"accepted_count": accepted_count})
        self._record_diagnostics()

    async def record_phase(self, phase, failure_code=None):
        value = WorkerCompletionDiagnostics.model_validate(
            {
                **self.diagnostics.model_dump(by_alias=True),
                "phase": phase,
                "failureCode": failure_code,
                "reminderCount": self._reminders,
            }
        )
        if value == self.diagnostics:
            return
        self.diagnostics = value
        self._record_diagnostics()
        if self.phase_sink is not None:
            await self.phase_sink(self.diagnostics)

    @classmethod
    async def prepare(cls, *, contract, allocation_id, client, timeout):
        if client.allocation_id != allocation_id:
            raise AuditResultError("audit_result_invalid")
        async with asyncio.timeout(timeout):
            task = await client.read_artifact(contract.task, max_bytes=MAX_PACKAGE_BYTES)
            manifest = await client.read_artifact(
                contract.execution_manifest,
                max_bytes=MAX_COLLECTED_BYTES,
            )
        for value, exact, media in (
            (task, contract.task, "application/zip"),
            (manifest, contract.execution_manifest, "application/json"),
        ):
            if (
                not isinstance(value, ArtifactValue)
                or value.artifact != exact
                or value.media_type != media
            ):
                raise AuditResultError("audit_result_invalid")
        records = _decode_task_input(task.data, "application/zip")
        owner = AuditInvocationOwner(
            allocation_id,
            "audit-preparation",
            "sha256:" + hashlib.sha256(task.data).hexdigest(),
            tuple(record[0]["item_key"] for record in records),
        )
        inputs = AuditTrustedInputs(owner, task.data, manifest.data)
        # Real encoder validates exact membership/order/digests before the model exists.
        await InvocationAuditCollector(inputs).discard()
        return cls(contract, inputs, client)

    def current(self) -> InvocationAuditCollector:
        if self._collector is None:
            raise AuditCollectionError("invocation", "No active Audit invocation.")
        return self._collector

    def begin(self, invocation_id: str) -> None:
        if self._collector is not None:
            raise AuditResultError("audit_result_invalid")
        inputs = replace(
            self._inputs, owner=replace(self._inputs.owner, invocation_id=invocation_id)
        )
        self._collector = InvocationAuditCollector(inputs)
        self._reminders = 0
        self.reset_diagnostics()

    async def end(self) -> None:
        collector, self._collector = self._collector, None
        if collector is not None:
            # Keep invocation ownership until discard settles, including a
            # cancellation arriving during otherwise successful cleanup.
            cleanup = asyncio.create_task(collector.discard(), name="audit-collector-discard")
            interrupted = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    interrupted = True
            cleanup.result()
            if interrupted:
                raise asyncio.CancelledError

    async def finish(
        self,
        *,
        request: StageContentRequest,
        deadline: float,
        check_active: Callable[[], None],
    ) -> ContinueCompletion | CompleteCompletion | FailCompletion:
        check_active()
        collector = self.current()
        snapshot = await collector.snapshot()
        accepted = {item.value.item_key for item in snapshot.items}
        self.record_progress(len(accepted))
        missing = [key for key in collector.owner.item_keys if key not in accepted]
        if missing:
            if self._reminders >= MAX_REMINDERS:
                return FailCompletion(
                    WorkerFailure(
                        code="audit_result_incomplete",
                        message=(
                            f"Audit results incomplete: {len(accepted)}/"
                            f"{len(collector.owner.item_keys)} recorded."
                        ),
                        retryable=True,
                    )
                )
            self._reminders += 1
            await self.record_phase("collecting")
            return ContinueCompletion(
                "Runtime completion reminder: use read_audit_task and submit_check_result "
                "to record "
                "the missing assigned items: " + ", ".join(missing) + ". "
                "Use only assessments and evidence permitted by each task; "
                "terminal prose does not record results."
            )
        publisher = DeterministicAuditResultPublisher(
            owner=collector.owner,
            result_artifact=self.contract.result_artifact,
            client=self._client,
            check_active=check_active,
        )
        sealed = await collector.seal()
        await self.record_phase("sealed")
        await self.record_phase("publishing")
        receipt = await publisher.publish(sealed, inputs=collector.inputs, deadline=deadline)
        check_active()
        await self.record_phase("published")
        exact = ArtifactRef(
            namespace=receipt.namespace, name=receipt.name, revision=receipt.revision
        )
        # The binding is contract-owned; the model cannot contribute an artifact ref.
        artifacts = {
            slot: exact.model_copy(deep=True)
            for slot, binding in request.result_artifacts.items()
            if binding == self.contract.result_artifact
        }
        return CompleteCompletion(
            WorkerResult(
                subtaskId=request.subtask_id,
                result=f"Recorded and published results for {len(accepted)} assigned Audit items.",
                artifacts=artifacts,
                summarized=False,
                observations=WorkerObservations(
                    profile="lean@1", tools={}, workspace=None, truncated=False
                ),
            )
        )
