"""Opt-in real CLI startup, mTLS listener, ADK sample and Podman teardown.

Only remote peers and model responses are stand-ins. No image provisioning,
network model call, optimistic capability snapshot or host execution fallback.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fakes.podman_workflow import ArtifactPeer, exercise_sample, sample_model
from test_cli import generate_pki, unused_tcp_port
from test_control_client import heartbeat_response, registration_response

import contractor_runtime.cli as runtime_cli
from contractor_runtime.allocation import AllocationService
from contractor_runtime.factories import built_in_factories
from contractor_runtime.settings import parse_settings
from contractor_runtime.state import RuntimeState

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_WORKFLOW_GATE") != "1",
    reason="explicit real-rootless deployment/workflow gate",
)


def test_real_cli_startup_registration_sample_artifact_and_release(tmp_path, monkeypatch):
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    assert image, "preinstalled digest-pinned CONTRACTOR_TEST_PODMAN_IMAGE is required"
    pki = generate_pki(tmp_path / "pki")
    port = unused_tcp_port()
    # Same immutable CLI/env surface as the deployment recipe.
    settings = parse_settings(
        ["--listen", f"127.0.0.1:{port}"],
        environ={
            "CONTRACTOR_CONTROL_PLANE_URL": "https://localhost:8443",
            "CONTRACTOR_ADVERTISED_CONTROL_URL": f"https://localhost:{port}",
            "CONTRACTOR_ADVERTISED_A2A_URL": f"https://localhost:{port}",
            "CONTRACTOR_CA_FILE": str(pki["ca"]),
            "CONTRACTOR_CERTIFICATE_FILE": str(pki["agent_certificate"]),
            "CONTRACTOR_PRIVATE_KEY_FILE": str(pki["agent_key"]),
            "CONTRACTOR_WORKSPACE_STORAGE": "local",
            "CONTRACTOR_WORKSPACE_WORK_ROOT": str(tmp_path / "project"),
            "CONTRACTOR_WORK_ROOT": str(tmp_path / "scratch"),
            "CONTRACTOR_PODMAN_ENABLED": "true",
            "CONTRACTOR_PODMAN_IMAGE": image,
            "CONTRACTOR_PODMAN_OWNER": "workflow-gate-" + uuid.uuid4().hex,
            "CONTRACTOR_SHUTDOWN_GRACE_SECONDS": "30",
        },
    )

    async def scenario():
        peer, model = ArtifactPeer(), sample_model()
        state, stop = RuntimeState(instance_id="podman-workflow"), asyncio.Event()
        captured = {}
        exits = []

        def factories(*args, **kwargs):
            kwargs["artifact_client_factory"] = lambda *_: peer
            kwargs["model_factory"] = lambda _: model
            captured["lifecycle"] = kwargs["execution_lifecycle"]
            return built_in_factories(*args, **kwargs)

        def service(*args, **kwargs):
            captured["service"] = AllocationService(*args, **kwargs, force_exit=exits.append)
            return captured["service"]

        class ControlPeer:
            def __init__(self):
                self.ready = asyncio.Event()
                self.registration = None

            async def post_json(self, path, payload):
                if path.endswith("/register"):
                    self.registration = payload
                    assert "podman@1" in payload["supportedSandboxProfiles"]
                    assert {"ref": "code-execution@1", "tools": ["exec_command"]} in payload[
                        "supportedToolsets"
                    ]
                    assert not list(settings.workspace.work_root.glob("workspace-*"))
                    assert captured["lifecycle"].probe_available
                    return registration_response()
                response = heartbeat_response(payload["heartbeatSeq"])
                # Allow the production client to install the confirmed lease.
                asyncio.get_running_loop().call_soon(self.ready.set)
                return response

        control = ControlPeer()
        monkeypatch.setattr(runtime_cli, "built_in_factories", factories)
        monkeypatch.setattr(runtime_cli, "AllocationService", service)
        task = asyncio.create_task(
            runtime_cli.serve(
                settings,
                state=state,
                transport=control,
                stop_requested=stop,
                install_signal_handlers=False,
            )
        )
        ready = asyncio.create_task(control.ready.wait())
        try:
            done, _ = await asyncio.wait(
                {task, ready}, timeout=60, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task
                pytest.fail("Runtime exited before registration/heartbeat")
            assert ready in done, "bounded startup did not register"
            await asyncio.sleep(0)
            await exercise_sample(captured["service"], state, peer, model)
            assert captured["lifecycle"]._entry is None
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
            stop.set()
            await asyncio.wait_for(task, 45)
        assert captured["lifecycle"]._close_confirmed
        assert not exits
        assert not list(settings.workspace.work_root.glob("workspace-*"))
        assert peer.writes and peer.writes[0][0].revision

    asyncio.run(scenario())
