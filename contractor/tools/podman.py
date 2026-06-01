"""Podman-backed sandbox for the exploit agents' code-execution tools.

Exposes ``run_python`` / ``execute_bash`` (via :func:`code_exec_tools`) backed
by an ephemeral Kali container, scoped to a single worker **agent-run** (keyed
by the ADK ``invocation_id``):

- one container per agent-run, started lazily on the first call and reused
  across that run's calls so files/state persist;
- target source mounted read-only at ``/project``; a writable scratch
  ``/work`` (container fs) where scripts write — created files are returned as
  ``output_files`` and saved as artifacts by the tools;
- ``--network host`` (reach the live target), ``--rm`` + resource/pid/time caps,
  clean env (podman does not pass host env by default — no host secrets leak);
- torn down by :mod:`contractor.runners.plugins.sandbox_cleanup` at run end,
  with an ``atexit`` sweep + container TTL as backstops.

The engine is an ADK :class:`BaseCodeExecutor` so it stays on ADK rails (uses
``CodeExecutionInput`` / ``CodeExecutionResult`` / ``File`` and could later be
attached as ``LlmAgent.code_executor``); the agent-facing surface is the two
function tools.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import logging
import shutil
import subprocess
import threading
from typing import Any, Callable, Optional

from fsspec import AbstractFileSystem
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import (
    CodeExecutionInput, CodeExecutionResult, File)
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = "contractor-sandbox:latest"
_CONTAINER_PREFIX = "contractor-sbx-"
_PROJECT_MOUNT = "/project"
_WORKDIR = "/work"
_CONTAINER_TTL = "2h"          # self-expiry backstop if teardown is missed
_DEFAULT_TIMEOUT_S = 120
_MAX_OUTPUT_CHARS = 60_000     # truncate stdout/stderr returned to the model
_MAX_OUTPUT_FILES = 20
_MAX_FILE_BYTES = 1_000_000


class SandboxError(RuntimeError):
    """Raised when the sandbox container cannot be created or exec'd."""


def _name_token(key: str) -> str:
    """Collision-free, valid container-name token for a sandbox key."""
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _safe(key: str) -> str:
    """Sanitise a key for use inside an artifact path."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in key) or "default"


class KaliSandbox:
    """One ephemeral podman container, keyed by a stable scope key.

    The key is the worker's ``namespace`` (per-case / per-finding), so a single
    container is reused across all of that run's tool calls and torn down once.
    """

    def __init__(
        self,
        key: str,
        *,
        image: str = DEFAULT_SANDBOX_IMAGE,
        host_project_path: Optional[str] = None,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 512,
    ) -> None:
        self.key = key
        self.name = f"{_CONTAINER_PREFIX}{_name_token(key)}"
        self.image = image
        self.host_project_path = host_project_path
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self._started = False
        self._lock = threading.Lock()
        self._seq = 0

    # ── lifecycle ────────────────────────────────────────────────────────
    def ensure_started(self) -> None:
        with self._lock:
            if self._started:
                return
            if not shutil.which("podman"):
                raise SandboxError("podman not found on PATH")
            # Drop any stale container with the same name.
            subprocess.run(["podman", "rm", "-f", self.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cmd = [
                "podman", "run", "-d", "--rm", "--name", self.name,
                "--network", "host",
                "--memory", self.memory, "--cpus", self.cpus,
                "--pids-limit", str(self.pids_limit),
                "--workdir", _WORKDIR,
            ]
            if self.host_project_path:
                cmd += ["-v", f"{self.host_project_path}:{_PROJECT_MOUNT}:ro"]
            # No -e / --env-host: container gets a clean env (no host secrets).
            cmd += [self.image, "sleep", _CONTAINER_TTL]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise SandboxError(
                    f"failed to start sandbox ({self.image}): {res.stderr.strip()}"
                )
            self._started = True
            logger.info("sandbox %s started (image=%s)", self.name, self.image)

    def teardown(self) -> None:
        with self._lock:
            if not self._started:
                return
            subprocess.run(["podman", "rm", "-f", self.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._started = False
            logger.info("sandbox %s removed", self.name)

    # ── exec helpers ─────────────────────────────────────────────────────
    def _write_file(self, path: str, content: str) -> None:
        res = subprocess.run(
            ["podman", "exec", "-i", self.name, "sh", "-c", f"cat > {path}"],
            input=content.encode(), capture_output=True,
        )
        if res.returncode != 0:
            raise SandboxError(f"failed to write {path}: {res.stderr.decode().strip()}")

    def _list_workdir(self) -> set[str]:
        res = subprocess.run(
            ["podman", "exec", self.name, "sh", "-c",
             f"find {_WORKDIR} -maxdepth 2 -type f"],
            capture_output=True, text=True,
        )
        return {ln.strip() for ln in res.stdout.splitlines() if ln.strip()}

    def _read_file(self, path: str) -> bytes:
        res = subprocess.run(["podman", "exec", self.name, "cat", path],
                             capture_output=True)
        return res.stdout[:_MAX_FILE_BYTES]

    def _collect_new_files(self, before: set[str], skip: set[str]) -> list[File]:
        files: list[File] = []
        for path in sorted(self._list_workdir() - before - skip):
            if len(files) >= _MAX_OUTPUT_FILES:
                break
            data = self._read_file(path)
            if not data:
                continue
            name = path[len(_WORKDIR) + 1:]
            files.append(File(name=name, content=data,
                              mime_type="application/octet-stream"))
        return files

    def _exec(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        """Run argv in the container under an in-container `timeout`."""
        full = ["podman", "exec", self.name,
                "timeout", "--signal=KILL", str(timeout_s), *argv]
        try:
            res = subprocess.run(full, capture_output=True, text=True,
                                 timeout=timeout_s + 15)
        except subprocess.TimeoutExpired:
            return 124, "", f"sandbox exec exceeded {timeout_s}s wall-clock"
        out = res.stdout[:_MAX_OUTPUT_CHARS]
        err = res.stderr[:_MAX_OUTPUT_CHARS]
        if res.returncode == 124:
            err = (err + f"\n[timed out after {timeout_s}s]").strip()
        return res.returncode, out, err

    # ── code execution ───────────────────────────────────────────────────
    def run_python(self, code: str, preinit: Optional[list[str]],
                   timeout_s: int) -> tuple[CodeExecutionResult, str]:
        self.ensure_started()
        self._seq += 1
        script_name = f"{_WORKDIR}/script_{self._seq}.py"
        body = ""
        if preinit:
            body += "# --- preinit ---\n" + "\n".join(preinit) + "\n\n"
        body += "# --- script ---\n" + code + "\n"
        self._write_file(script_name, body)
        before = self._list_workdir()
        rc, out, err = self._exec(["python3", script_name], timeout_s)
        output_files = self._collect_new_files(before, skip={script_name})
        result = CodeExecutionResult(stdout=out, stderr=err, output_files=output_files)
        return result, body

    def run_bash(self, command: str, timeout_s: int) -> CodeExecutionResult:
        self.ensure_started()
        before = self._list_workdir()
        rc, out, err = self._exec(["sh", "-c", command], timeout_s)
        output_files = self._collect_new_files(before, skip=set())
        return CodeExecutionResult(stdout=out, stderr=err, output_files=output_files)


# ── sandbox registry (keyed by scope key = worker namespace) ──────────────
_SANDBOXES: dict[str, KaliSandbox] = {}
_REGISTRY_LOCK = threading.Lock()


def get_or_create_sandbox(key: str, **kwargs: Any) -> KaliSandbox:
    with _REGISTRY_LOCK:
        sb = _SANDBOXES.get(key)
        if sb is None:
            sb = KaliSandbox(key, **kwargs)
            _SANDBOXES[key] = sb
        return sb


def teardown_sandbox(key: str) -> None:
    """Remove the container for one scope key."""
    with _REGISTRY_LOCK:
        sb = _SANDBOXES.pop(key, None)
    if sb is not None:
        sb.teardown()


def teardown_all() -> None:
    """Remove every live sandbox. Called at run end (cleanup plugin) + atexit.

    Safe because code-exec runs are sequential (the only parallel workflow,
    trace_graph_pathpar, does not use code-exec), so no other run's sandbox is
    active when one finishes.
    """
    with _REGISTRY_LOCK:
        sandboxes = list(_SANDBOXES.values())
        _SANDBOXES.clear()
    for sb in sandboxes:
        try:
            sb.teardown()
        except Exception:  # best-effort
            logger.exception("sandbox teardown failed for %s", sb.name)


atexit.register(teardown_all)


class KaliCodeExecutor(BaseCodeExecutor):
    """ADK code executor backed by :class:`KaliSandbox` (one per agent-run)."""

    image: str = DEFAULT_SANDBOX_IMAGE
    host_project_path: Optional[str] = None

    def _sandbox(self, invocation_id: str) -> KaliSandbox:
        return get_or_create_sandbox(
            invocation_id, image=self.image,
            host_project_path=self.host_project_path,
        )

    def execute_code(self, invocation_context: Any,
                     code_execution_input: CodeExecutionInput) -> CodeExecutionResult:
        inv = getattr(invocation_context, "invocation_id", None) or "default"
        result, _ = self._sandbox(inv).run_python(
            code_execution_input.code, preinit=None, timeout_s=_DEFAULT_TIMEOUT_S)
        return result


async def _save_artifacts(
    tool_context: Optional[ToolContext], namespace: str,
    script: Optional[str], output_files: list[File],
) -> list[str]:
    """Persist the executed script + any created files as artifacts."""
    if tool_context is None:
        return []
    saved: list[str] = []
    base = f"code-exec/{_safe(namespace)}"
    try:
        if script is not None:
            key = f"{base}/script_{abs(hash(script)) % 10000:04d}.py"
            await tool_context.save_artifact(
                filename=key, artifact=types.Part.from_text(text=script))
            saved.append(key)
        for f in output_files:
            key = f"{base}/files/{f.name}"
            data = f.content if isinstance(f.content, bytes) else f.content.encode()
            await tool_context.save_artifact(
                filename=key, artifact=types.Part.from_bytes(
                    data=data, mime_type=f.mime_type))
            saved.append(key)
    except Exception:  # artifact persistence must not break the tool
        logger.exception("failed to persist code-exec artifacts")
    return saved


def code_exec_tools(
    namespace: str,
    fs: Optional[AbstractFileSystem] = None,
    *,
    image: str = DEFAULT_SANDBOX_IMAGE,
    default_timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> list[Callable[..., Any]]:
    """Build the ``run_python`` / ``execute_bash`` tools for an exploit agent.

    The target source (``fs.root_path`` when ``fs`` is a rooted local fs) is
    bind-mounted read-only into the per-run sandbox; pass ``fs=None`` for
    black-box agents (no project mount).
    """
    host_project_path = getattr(fs, "root_path", None)

    def _sandbox() -> KaliSandbox:
        # Keyed by the per-case namespace → one container per run, reused across
        # this run's tool calls and torn down once at run end.
        return get_or_create_sandbox(
            namespace, image=image, host_project_path=host_project_path)

    async def run_python(
        code: str,
        preinit: Optional[list[str]] = None,
        timeout_s: int = default_timeout_s,
        tool_context: Optional[ToolContext] = None,
    ) -> dict[str, Any]:
        """Execute a Python 3 script in the exploitation sandbox and return its output.

        The sandbox is a Kali container (requests/httpx/pwntools/bs4/pyjwt
        preinstalled, host network) that persists for this agent run — files you
        write to the working directory survive across calls and are saved as
        artifacts. Use this to script repetitive work in ONE call (e.g. a blind
        SQL-injection extraction via a binary-search loop) instead of issuing the
        same request hundreds of times by hand.

        Args:
            code: Python source to run. The target source tree is mounted
                read-only at /project; write any output files to the current dir.
            preinit: Optional setup snippets executed before `code` in the same
                interpreter (seed variables, imports, helpers, captured data).
            timeout_s: Hard wall-clock limit for the script in seconds.

        Returns:
            stdout, stderr, exit_code, and the artifact keys of the saved script
            and any files the script produced.
        """
        result, script = await asyncio.to_thread(
            _sandbox().run_python, code, preinit, int(timeout_s))
        saved = await _save_artifacts(
            tool_context, namespace, script, result.output_files)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "artifacts": saved,
        }

    async def execute_bash(
        command: str,
        timeout_s: int = default_timeout_s,
        tool_context: Optional[ToolContext] = None,
    ) -> dict[str, Any]:
        """Run a shell command in the same exploitation sandbox as run_python.

        Kali CLI tooling is available (curl, jq, nmap, sqlmap, netcat,
        gobuster). The container persists for this agent run and shares its
        filesystem with run_python.

        Args:
            command: Shell command line to execute via `sh -c`.
            timeout_s: Hard wall-clock limit in seconds.

        Returns:
            stdout, stderr, and the artifact keys of any files produced.
        """
        result = await asyncio.to_thread(
            _sandbox().run_bash, command, int(timeout_s))
        saved = await _save_artifacts(
            tool_context, namespace, None, result.output_files)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "artifacts": saved,
        }

    return [run_python, execute_bash]


__all__ = [
    "KaliSandbox",
    "KaliCodeExecutor",
    "SandboxError",
    "code_exec_tools",
    "get_or_create_sandbox",
    "teardown_sandbox",
    "teardown_all",
    "DEFAULT_SANDBOX_IMAGE",
]
