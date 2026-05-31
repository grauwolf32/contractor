"""Unit tests for the podman-backed code-execution sandbox.

Podman is mocked: these assert the container is launched with the expected
isolation flags, that preinit is prepended, timeouts are enforced in-container,
and teardown removes the container. One integration test runs a real script iff
podman + the sandbox image are present.
"""

from __future__ import annotations

import shutil
import subprocess
from unittest import mock

import pytest

from contractor.tools import podman
from contractor.tools.podman import (DEFAULT_SANDBOX_IMAGE, KaliSandbox,
                                     code_exec_tools, get_or_create_sandbox,
                                     teardown_sandbox)


class _FakePodman:
    """Records podman invocations and returns plausible results."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[bytes | None] = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        self.inputs.append(kw.get("input"))
        text = kw.get("text", False)
        rc, out, err = 0, "", ""
        if cmd[:3] == ["podman", "run", "-d"]:
            out = "containerid123\n"
        elif "find" in cmd:
            out = ""  # no created files
        elif "timeout" in cmd:
            out = "SCRIPT-STDOUT"
        elif len(cmd) >= 3 and cmd[2] == "cat":
            return mock.Mock(returncode=0, stdout=b"filedata", stderr=b"")
        if not text and kw.get("input") is not None:
            return mock.Mock(returncode=0, stdout=b"", stderr=b"")
        return mock.Mock(returncode=rc, stdout=out, stderr=err)


@pytest.fixture
def fake_podman(monkeypatch):
    fake = _FakePodman()
    monkeypatch.setattr(podman.subprocess, "run", fake)
    monkeypatch.setattr(podman.shutil, "which", lambda _x: "/usr/bin/podman")
    return fake


def _run_cmd(fake: _FakePodman) -> list[str]:
    return next(c for c in fake.calls if c[:3] == ["podman", "run", "-d"])


def test_container_launched_with_isolation_flags(fake_podman):
    sb = KaliSandbox("inv-1", host_project_path="/host/project")
    sb.ensure_started()
    cmd = _run_cmd(fake_podman)
    assert "--rm" in cmd
    assert ["--network", "host"] == cmd[cmd.index("--network"):cmd.index("--network") + 2]
    assert "/host/project:/project:ro" in cmd          # project mounted read-only
    assert cmd[cmd.index("--workdir") + 1] == "/work"  # writable scratch
    assert "--memory" in cmd and "--cpus" in cmd and "--pids-limit" in cmd
    assert DEFAULT_SANDBOX_IMAGE in cmd
    assert cmd[-2:] == ["sleep", podman._CONTAINER_TTL]


def test_env_is_scrubbed(fake_podman):
    """No host env passed into the sandbox (no -e / --env-host)."""
    KaliSandbox("inv-env").ensure_started()
    cmd = _run_cmd(fake_podman)
    assert "-e" not in cmd and "--env-host" not in cmd


def test_no_project_mount_when_fs_absent(fake_podman):
    KaliSandbox("inv-nofs", host_project_path=None).ensure_started()
    cmd = _run_cmd(fake_podman)
    assert not any(c == "-v" for c in cmd)


def test_run_python_prepends_preinit_and_enforces_timeout(fake_podman):
    sb = KaliSandbox("inv-2")
    result, script = sb.run_python("print(X)", preinit=["X = 41"], timeout_s=30)
    # preinit appears before the main script body
    assert "X = 41" in script
    assert script.index("preinit") < script.index("script")
    # the script was written via `cat >` with that content on stdin
    assert any(b"X = 41" in (i or b"") for i in fake_podman.inputs)
    # python is run under an in-container timeout
    exec_cmd = next(c for c in fake_podman.calls if "timeout" in c and "python3" in c)
    assert "30" in exec_cmd
    assert result.stdout == "SCRIPT-STDOUT"


def test_execute_bash_runs_in_same_container(fake_podman):
    sb = KaliSandbox("inv-3")
    sb.ensure_started()
    res = sb.run_bash("id", timeout_s=10)
    bash_cmd = next(c for c in fake_podman.calls if c[-1] == "id" or "id" in c[-1:])
    assert "timeout" in bash_cmd and "sh" in bash_cmd
    assert res.stdout == "SCRIPT-STDOUT"


def test_teardown_removes_container(fake_podman):
    sb = KaliSandbox("inv-4")
    sb.ensure_started()
    sb.teardown()
    assert ["podman", "rm", "-f", sb.name] in fake_podman.calls


def test_registry_and_teardown_sandbox(fake_podman):
    sb = get_or_create_sandbox("inv-reg")
    assert get_or_create_sandbox("inv-reg") is sb     # reused, not recreated
    sb.ensure_started()
    teardown_sandbox("inv-reg")
    assert ["podman", "rm", "-f", sb.name] in fake_podman.calls
    # gone from registry → a fresh one is created next time
    assert get_or_create_sandbox("inv-reg") is not sb


def test_sandbox_keyed_by_namespace(fake_podman):
    # same namespace → same container reused; different namespace → distinct
    a = get_or_create_sandbox("exploit:case-1")
    assert get_or_create_sandbox("exploit:case-1") is a
    b = get_or_create_sandbox("exploit:case-2")
    assert b is not a and b.name != a.name


def test_teardown_all_sweeps_registry(fake_podman):
    from contractor.tools.podman import _SANDBOXES, teardown_all
    a = get_or_create_sandbox("ns-a")
    a.ensure_started()
    b = get_or_create_sandbox("ns-b")
    b.ensure_started()
    teardown_all()
    assert ["podman", "rm", "-f", a.name] in fake_podman.calls
    assert ["podman", "rm", "-f", b.name] in fake_podman.calls
    assert not _SANDBOXES


@pytest.mark.asyncio
async def test_cleanup_plugin_sweeps_only_on_root(monkeypatch):
    import contractor.runners.plugins.sandbox_cleanup as mod
    swept = []
    monkeypatch.setattr(mod, "teardown_all", lambda: swept.append(1))
    p = mod.SandboxCleanupPlugin()

    class IC:
        def __init__(self, i): self.invocation_id = i

    await p.before_run_callback(invocation_context=IC("P"))  # outer run
    await p.before_run_callback(invocation_context=IC("W"))  # inner AgentTool run
    await p.after_run_callback(invocation_context=IC("W"))   # inner end → no sweep
    assert swept == []
    await p.after_run_callback(invocation_context=IC("P"))   # outer end → sweep once
    assert swept == [1]


@pytest.mark.asyncio
async def test_code_exec_tools_surface_and_result(fake_podman):
    tools = code_exec_tools(namespace="exploit:demo", fs=None)
    names = {t.__name__ for t in tools}
    assert names == {"run_python", "execute_bash"}
    run_python = next(t for t in tools if t.__name__ == "run_python")
    out = await run_python(code="print(1)", tool_context=None)
    assert out["stdout"] == "SCRIPT-STDOUT"
    assert "stderr" in out and "artifacts" in out


@pytest.mark.skipif(
    not shutil.which("podman")
    or subprocess.run(["podman", "image", "exists", DEFAULT_SANDBOX_IMAGE]).returncode != 0,
    reason="podman + contractor-sandbox image required",
)
def test_integration_real_sandbox():
    sb = KaliSandbox("inttest-001")
    try:
        result, _ = sb.run_python(
            "print('hi'); open('o.txt','w').write('z')", preinit=None, timeout_s=30)
        assert "hi" in result.stdout
        assert any(f.name == "o.txt" for f in result.output_files)
    finally:
        sb.teardown()
    listed = subprocess.run(
        ["podman", "ps", "--filter", f"name={sb.name}", "--format", "{{.Names}}"],
        capture_output=True, text=True).stdout.strip()
    assert listed == ""
