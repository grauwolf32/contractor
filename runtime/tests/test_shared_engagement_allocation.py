"""Prepare consumes the same private metadata projection as the Server."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.allocation import allocation_snapshot
from fakes.spec import allocation_spec
from fakes.worker_runtime import StubADKWorkerRuntimeFactory

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationSpec,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    ToolsetRef,
    ToolsetSelection,
    decode_private,
)
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import built_in_factories
from contractor_runtime.state import RuntimeState

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "api/testdata/v1alpha1/valid/allocation-spec-engagement-metadata.json"
)


@pytest.mark.parametrize("network", [False, True])
def test_shared_engagement_prepare_reads_metadata_and_rejects_target_transport(
    tmp_path: Path, runtime_capabilities: CapabilitySnapshot, network: bool
) -> None:
    fixture = decode_private(AllocationSpec, FIXTURE.read_bytes())

    async def scenario() -> None:
        state = RuntimeState(instance_id="engagement-test")
        await state.mark_registered()
        factories = replace(
            built_in_factories(tmp_path),
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        )
        service = AllocationService(
            state, factories, runtime_capabilities, a2a_base_url="https://runtime.example"
        )
        spec = allocation_spec()
        spec.engagement = fixture.engagement
        if network:
            spec.agent_template.toolsets = [
                ToolsetSelection(
                    ref=ToolsetRef(toolsetId="http-tools", version="1"), tools=["http_request"]
                )
            ]
            spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
            with pytest.raises(AllocationError) as rejected:
                await service.prepare(spec)
            assert rejected.value.code == "unsupported_toolset"
            assert "engagement metadata" in str(rejected.value)
            assert rejected.value.status_code == 422
            assert not rejected.value.retryable
            assert await allocation_snapshot(service) is None
            assert not list(tmp_path.iterdir())
            return
        first = await service.prepare(spec)
        assert await service.prepare(spec) == first
        changed = spec.model_copy(deep=True)
        assert changed.engagement is not None
        changed.engagement.engagement_digest = "sha256:" + "f" * 64
        with pytest.raises(AllocationError, match="differs from the active allocation"):
            await service.prepare(changed)
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="engagement-final",
                deadline=datetime.now(UTC) + timedelta(seconds=30),
            )
        )
        request = ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        await service.release(request)
        await service.release(request)
        await service.confirm_release(spec.allocation_id)
        assert await allocation_snapshot(service) is None

    asyncio.run(scenario())


def test_shared_engagement_projection_rejects_widening_and_aliases() -> None:
    fixture = json.loads(FIXTURE.read_bytes())
    projection = fixture["engagement"]
    for key, value in list(projection.items()):
        if not any(character.isupper() for character in key):
            continue
        candidate = json.loads(json.dumps(fixture))
        alias = "".join("_" + c.lower() if c.isupper() else c for c in key)
        candidate["engagement"][alias] = value
        del candidate["engagement"][key]
        with pytest.raises(ValueError):
            decode_private(AllocationSpec, json.dumps(candidate).encode())
