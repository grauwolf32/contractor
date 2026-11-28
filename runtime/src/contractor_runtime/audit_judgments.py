"""Durable normalized judgments through the allocation-bound Audit owner."""

import json
from typing import ClassVar, Literal
from uuid import uuid4

import jcs
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import API_VERSION
from contractor_runtime.contracts.audit_documents import validate_audit_material
from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, RESOURCE_ID_PATTERN
from contractor_runtime.core.tool_errors import ToolResultDiagnostic
from contractor_runtime.strict_json import strict_json_loads


class JudgmentGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    itemKey: str
    taskId: str | None = None
    findingId: str | None = None
    generation: int = Field(ge=0, le=MAX_EXACT_INTEGER)


class JudgmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    apiVersion: Literal["contractor/v1alpha1"]
    material: dict
    generations: list[JudgmentGeneration]
    replayed: bool


class JudgmentStale(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    apiVersion: Literal["contractor/v1alpha1"]
    currentGenerations: list[JudgmentGeneration]


class JudgmentStaleError(Exception):
    def __init__(self, generations):
        super().__init__("Audit result changed before acceptance")
        self.generations = generations


class JudgmentPersistenceUnknown(ArtifactTransportError, ToolResultDiagnostic):
    code = "audit_result_persistence_unknown"
    retryable = False
    diagnostic_message = (
        "Judgment acceptance is unconfirmed after bounded receipt reconciliation. "
        "Earlier accepted results remain saved; do not repeat target traffic."
    )
    model_details: ClassVar[dict] = {"persistence": "unknown"}


class AuditJudgmentClient:
    def __init__(self, client, encoder, inputs):
        self.client, self.encoder, self.inputs = client, encoder, inputs
        records = encoder.trusted_inputs(inputs)
        self.scope_kind, self.scope_id = records.scope_kind, records.scope_id
        self.members = records.items

    async def __call__(self, snapshot, bases, mode):
        def prepare():
            document = strict_json_loads(self.encoder.encode(snapshot, inputs=self.inputs).data)
            return jcs.canonicalize(
                {
                    "apiVersion": API_VERSION,
                    "invocationId": snapshot.owner.invocation_id,
                    "submissionId": uuid4().hex,
                    "mode": mode,
                    "items": [
                        {"baseGeneration": bases[item["item_key"]] or 0, "conclusion": item}
                        for item in document["items"]
                    ],
                }
            )

        body = await threads.to_thread_until_done(prepare, name="audit-judgment-request")
        keys = tuple(item.value.item_key for item in snapshot.items)
        try:
            return await self._submit(body, keys)
        except ArtifactTransportError:
            raise JudgmentPersistenceUnknown() from None

    async def _submit(self, body, keys):
        # Both technical sends retain the exact body, base and mutation identity.
        # A new target request or new semantic generation is never recovery.
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Audit judgment request exceeds its size limit")
        for attempt in range(2):
            try:
                return await self._attempt(body, keys)
            except ArtifactTransportError:
                if attempt:
                    raise

    async def _attempt(self, body, keys):
        input_value = await threads.to_thread_until_done(
            strict_json_loads, body, name="audit-judgment-wire"
        )
        try:
            response = await self.client.transport.submit_audit_judgments(
                self.client.allocation_id, input_value
            )
        except ArtifactAPIError as error:
            if error.code in {"invalid_response", "invalid_error_response"}:
                raise ArtifactTransportError("Audit judgment receipt is unconfirmed") from None
            raise
        if not isinstance(response, dict) or set(response) not in ({"receipt"}, {"stale"}):
            raise ArtifactTransportError("Audit result API returned an invalid response")
        if "stale" in response:
            try:
                conflict = JudgmentStale.model_validate(response["stale"])
                generations = self._generations(conflict.currentGenerations, keys)
            except (ValidationError, ValueError, TypeError):
                raise ArtifactTransportError(
                    "Audit result API returned invalid generations"
                ) from None
            raise JudgmentStaleError(generations)
        value = response["receipt"]
        try:
            receipt = JudgmentReceipt.model_validate(value)
            generations = self._generations(receipt.generations, keys)
            if any(value < 1 for value in generations.values()):
                raise ValueError("Nonpositive accepted generation")
            await threads.to_thread_until_done(
                validate_audit_material, receipt.material, name="audit-judgment-receipt"
            )
            if (
                receipt.material["scope_kind"] != self.scope_kind
                or receipt.material["scope_id"] != self.scope_id
                or receipt.material["media_type"] != "application/json"
            ):
                raise ValueError("Another Audit or media type")
        except (ValidationError, ValueError, TypeError, KeyError, SchemaValidationError):
            raise ArtifactTransportError("Audit result API returned an invalid receipt") from None
        return {
            "generations": generations,
            "material": json.loads(receipt.model_dump_json())["material"],
        }

    def _generations(self, values, keys):
        generations = {value.itemKey: value.generation for value in values}
        if len(values) != len(generations) or set(generations) != set(keys):
            raise ValueError("Another judgment membership")
        if any(
            (value.taskId != self.members[value.itemKey].get("task_id"))
            or (value.findingId != self.members[value.itemKey].get("finding_id"))
            or RESOURCE_ID_PATTERN.fullmatch(value.taskId or value.findingId or "") is None
            for value in values
        ):
            raise ValueError("Invalid Task identity")
        return generations
