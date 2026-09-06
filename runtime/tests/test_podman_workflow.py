from __future__ import annotations

import asyncio

import pytest
from fakes.podman_lifecycle import owner
from fakes.podman_workflow import ArtifactPeer, Commands, exercise_sample, sample_model, sample_spec
from test_projectfs_storage import local_settings

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.state import RuntimeState


@pytest.mark.parametrize("publish", [True, False])
def test_sample_selected_tools_disk_edits_and_only_explicit_artifacts(tmp_path, publish):
    async def scenario():
        async with owner(tmp_path) as fixture:
            peer, model = ArtifactPeer(), sample_model(publish=publish)

            async def verified(root, *, deadline):
                assert fixture.backend.recovered
                return {"available": True, "failure": None}

            # Complete probe behavior is covered separately in V31-006.
            fixture.backend.probe = verified
            fixture.backend.commands = Commands(fixture)
            factories = built_in_factories(
                tmp_path / "scratch",
                artifact_client_factory=lambda *_: peer,
                model_factory=lambda _: model,
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            capabilities = await discover_capabilities(factories)
            state = RuntimeState(instance_id="sample", capabilities=capabilities)
            await state.mark_registered()
            exits = []
            service = AllocationService(
                state,
                factories,
                capabilities,
                a2a_base_url="https://runtime.example",
                force_exit=exits.append,
            )
            await exercise_sample(service, state, peer, model, publish=publish)
            assert not fixture.cli.containers and not exits

    asyncio.run(scenario())


def test_sample_uses_exact_source_and_no_skills_or_overlay_export():
    peer = ArtifactPeer()
    spec, request = sample_spec(peer)
    assert spec.workspace.sources[0].artifact == request.artifacts["source"] == peer.source_ref
    assert peer.source_ref.revision is not None
    assert spec.workspace.mode == "direct" and spec.workspace.export is None
    assert not spec.resolved_skills and not spec.agent_template.skills
    assert request.result_artifacts["report"].model_dump(exclude_none=True) == {
        "namespace": "builder",
        "name": "check_report",
    }


def test_service_template_preserves_independent_cleanup_owners():
    from configparser import ConfigParser
    from pathlib import Path

    unit = ConfigParser(interpolation=None)
    unit.read(
        Path(__file__).resolve().parents[2] / "deploy/podman/contractor-runtime.service.example"
    )
    service = unit["Service"]
    assert service["Type"] == "exec"
    assert service["KillMode"] == "mixed" and service["SendSIGKILL"] == "no"
    assert service["Restart"] == "no" and service["Delegate"] == "yes"
    assert service["UMask"] == "0077"
    assert ".venv/bin/contractor-runtime " in service["ExecStart"]
    assert "uv run" not in service["ExecStart"]
