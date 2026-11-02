"""Opt-in local fault boundaries for the real mTLS/ADK process harness.

Artifact-named tools below are synchronization fixtures, not ArtifactStore
implementations. The normal lifecycle_server never installs this module.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from google.adk.tools import ToolContext

import contractor_runtime.allocation.cleanup as allocation_cleanup
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.worker.runtime import INVOCATION_CLEANUP_TIMEOUT_SECONDS

from .clock import advance_clock

MODES = (
    "native_tool_cancel_join",
    "expired_deadline_no_admission",
    "native_deadline_stops_next_dispatch",
    "native_parallel_child_cancel_keeps_completed_sibling",
    "native_cleanup_failure_fences_until_release",
    "native_lease_expiry_joins_active_adk",
    "native_terminal_cleanup_budget_expires",
    "native_failed_prepare_rollback",
    "native_successful_release_replay",
)


def _publish(path: Path, payload: str) -> None:
    staging = path.with_suffix(".staging")
    staging.write_text(payload)
    os.replace(staging, path)


async def _joined_even_if_cancelled(task: asyncio.Task) -> None:
    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            if task.done():
                task.result()
                return


class NativeFaults:
    def __init__(self, mode: str, directory: Path) -> None:
        self.mode = mode
        self.directory = directory
        directory.mkdir(parents=True)
        self.events: list[str] = []
        self.dispatch: dict[str, int] = {}
        self.created: dict[str, int] = {}
        self.closed: dict[str, int] = {}
        self.active = 0
        self.workspace = ""
        self.callback_entries = {"model": 0, "tool": 0}
        self.admitted = {"model": 0, "tool": 0}
        self._record_lock = asyncio.Lock()
        self.join = asyncio.Event()
        self.cleanup = asyncio.Event()
        self.completed = asyncio.Event()
        self.watchdog: LeaseWatchdog | None = None
        self._poller: asyncio.Task | None = None
        self._watcher: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def record(self, event: str) -> None:
        async with self._record_lock:
            self.events.append(event)
            payload = json.dumps(
                {
                    "events": self.events,
                    "dispatch": self.dispatch,
                    "created": self.created,
                    "closed": self.closed,
                    "active": self.active,
                    "callbackEntries": self.callback_entries,
                    "admitted": self.admitted,
                    "workspace": self.workspace,
                },
                separators=(",", ":"),
            )
            await to_thread_until_done(
                _publish,
                self.directory / "snapshot.json",
                payload,
                name="native-process-marker",
            )

    def factories(self, original):
        controller = self
        delegate = original.worker_runtimes["adk@1"]

        class WorkerFactory:
            ref = "adk@1"

            async def probe(self):
                return await delegate.probe()

            async def create(self, context):
                controller.workspace = str(context.workspace.path)
                if (
                    controller.mode == "native_failed_prepare_rollback"
                    and "prepare_failure_injected" not in controller.events
                ):
                    if set(context.tools) != FaultTools.exported_tools:
                        raise RuntimeError("Prepare fault did not follow exact local tool creation")
                    await controller.record("prepare_failure_injected")
                    raise RuntimeError("controlled process Worker creation failure")
                worker = await delegate.create(context)
                before_model = worker._plugin.before_model_callback
                before_tool = worker._plugin.before_tool_callback
                after_tool = worker._plugin.after_tool_callback

                async def model_callback(**kwargs):
                    controller.callback_entries["model"] += 1
                    result = await before_model(**kwargs)
                    controller.admitted["model"] += 1
                    await controller.record("model_admitted")
                    return result

                async def tool_callback(**kwargs):
                    controller.callback_entries["tool"] += 1
                    result = await before_tool(**kwargs)
                    if result is None:
                        controller.admitted["tool"] += 1
                    await controller.record("tool_callback")
                    return result

                async def tool_completed(**kwargs):
                    result = await after_tool(**kwargs)
                    if kwargs["tool"].name == "read_artifact" and controller.mode == MODES[3]:
                        await controller.record("completed_sibling_settled")
                        controller.completed.set()
                    return result

                worker._plugin.before_model_callback = model_callback
                worker._plugin.before_tool_callback = tool_callback
                worker._plugin.after_tool_callback = tool_completed
                if controller.mode in {MODES[2], MODES[6]}:
                    # This hook is inside production's independent bounded
                    # invocation-session cleanup, after semantic State closes.
                    finish = worker._session_lifecycle.finish_invocation

                    async def finish_after_expiry():
                        await controller.record("terminal_cleanup_started")
                        try:
                            await controller.cleanup.wait()
                        except asyncio.CancelledError:
                            await controller.record("terminal_cleanup_cancelled")
                            raise
                        result = await finish()
                        await controller.record("terminal_cleanup_done")
                        return result

                    worker._session_lifecycle.finish_invocation = finish_after_expiry
                await controller.record("adk_prepared")
                return worker

        return replace(
            original,
            worker_runtimes={**original.worker_runtimes, "adk@1": WorkerFactory()},
            toolsets={**original.toolsets, "run-artifacts@1": FaultTools(controller)},
        )

    async def start(self, service) -> None:
        abort = service.abort
        release = service.release

        async def observed_abort(request):
            await self.record("abort_requested")
            result = await abort(request)
            await self.record("abort_finished")
            return result

        async def observed_release(request):
            await self.record("release_requested")
            try:
                result = await release(request)
            except Exception:
                await self.record("release_failed")
                raise
            await self.record("release_finished")
            return result

        service.abort = observed_abort
        service.release = observed_release
        if self.mode == "native_cleanup_failure_fences_until_release":
            original = allocation_cleanup.release_adk_tool_caches
            attempts = 0

            def release_once_failing():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("controlled process cache cleanup failure")
                original()

            allocation_cleanup.release_adk_tool_caches = release_once_failing
        if self.mode == "native_lease_expiry_joins_active_adk":

            async def expired():
                await self.record("lease_expiry_started")
                await service.expire_control_lease(10)
                await self.record("lease_expiry_finished")

            self.watchdog = LeaseWatchdog(expired, monotonic=asyncio.get_running_loop().time)
            await self.watchdog.arm(60)
            self._watcher = asyncio.create_task(self.watchdog.run(self._stop))
        self._poller = asyncio.create_task(self._signals())
        await self.record("fault_harness_ready")

    def observe_transport(self, app):
        async def observed(scope, receive, send):
            body = bytearray()

            async def incoming():
                message = await receive()
                if message["type"] == "http.request":
                    remaining = max(0, (1 << 20) + 1 - len(body))
                    body.extend(message.get("body", b"")[:remaining])
                    if not message.get("more_body", False) and len(body) <= 1 << 20:
                        try:
                            method = json.loads(body).get("method")
                        except (ValueError, AttributeError):
                            method = None
                        if method == "GetTask" and "a2a_poll_seen" not in self.events:
                            await self.record("a2a_poll_seen")
                return message

            await app(scope, incoming, send)

        return observed

    async def _signals(self) -> None:
        handled = set()
        while not self._stop.is_set():
            for name in ("join", "cleanup", "lease-expire", "cleanup-expire"):
                if name in handled or not (self.directory / name).exists():
                    continue
                handled.add(name)
                if name == "join":
                    self.join.set()
                elif name == "cleanup":
                    self.cleanup.set()
                elif name == "cleanup-expire":
                    advance_clock(INVOCATION_CLEANUP_TIMEOUT_SECONDS + 1)
                else:
                    await self.record("lease_expiry_requested")
                    advance_clock(61)
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._poller, self._watcher):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._poller, self._watcher) if task is not None),
            return_exceptions=True,
        )


class FaultTools:
    ref = "run-artifacts@1"
    exported_tools = frozenset({"list_artifacts", "read_artifact", "write_artifact"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, controller: NativeFaults) -> None:
        self.controller = controller

    async def probe(self):
        return self.exported_tools

    async def create_selected(self, *, selected, **_kwargs):
        controller = self.controller

        async def enter(name):
            controller.active += 1
            controller.dispatch[name] = controller.dispatch.get(name, 0) + 1
            await controller.record(name + "_started")

        async def finish(name):
            controller.active -= 1
            await controller.record(name + "_joined")

        async def read_artifact(namespace: str, name: str, tool_context: ToolContext) -> dict:
            """Synchronize one local native ADK dispatch; no ArtifactStore effect."""
            del namespace, name
            await enter("read")
            if controller.mode in {MODES[0], MODES[5]}:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as error:

                    async def join_cancelled():
                        await controller.record("native_cancel_received")
                        await controller.join.wait()
                        await finish("read")

                    await _joined_even_if_cancelled(asyncio.create_task(join_cancelled()))
                    raise error
            await finish("read")
            if controller.mode in {MODES[2], MODES[6]}:
                advance_clock(7200)
            return {"completed": "sibling-once", "logicalInvocation": tool_context.invocation_id}

        async def write_artifact(namespace: str, name: str) -> dict:
            """Raise child-originated cancellation after a completed local sibling."""
            del namespace, name
            await enter("write")
            await controller.completed.wait()
            await finish("write")
            raise asyncio.CancelledError()

        async def list_artifacts() -> dict:
            """Return an empty test-only listing without a business-store read."""
            return {"artifacts": []}

        tools = {tool.__name__: tool for tool in (list_artifacts, read_artifact, write_artifact)}
        result = {}
        for name in selected:

            async def close(tool_name=name):
                if controller.active:
                    raise RuntimeError("Tool close preceded native dispatch join")
                controller.closed[tool_name] = controller.closed.get(tool_name, 0) + 1
                await controller.record("tool_closed")

            tool = tools[name]
            tool.name = name
            tool.close = close
            result[name] = tool
            controller.created[name] = controller.created.get(name, 0) + 1
        await controller.record("tools_created")
        return result
