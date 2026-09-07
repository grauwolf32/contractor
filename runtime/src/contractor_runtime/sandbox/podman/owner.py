"""Private host owner service; no public socket, mount or model RPC surface.

Runtime owns a framed stream for RPC and a seqpacket health endpoint. Health is read independently
of slow lifecycle RPCs. EOF revokes the guardian immediately, but this process
retains the engine flock until pending operations and exact removal settle.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import struct
import sys
import time
from dataclasses import asdict
from pathlib import Path

from contractor_runtime.sandbox.contracts import ExecutionRequest, SandboxContractError
from contractor_runtime.sandbox.podman.lifecycle_backend import LifecycleBackend
from contractor_runtime.sandbox.podman.settings import PodmanSettings

MAX_PACKET = 8192
MAX_RPC_BYTES = 16 << 20


async def send_rpc(channel: socket.socket, message: dict) -> None:
    raw = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(raw) > MAX_RPC_BYTES:
        raise ValueError("owner RPC exceeds bound")
    await asyncio.get_running_loop().sock_sendall(channel, struct.pack("!I", len(raw)) + raw)


async def receive_rpc(channel: socket.socket) -> dict:
    async def exact(size):
        raw = bytearray()
        while len(raw) < size:
            chunk = await asyncio.get_running_loop().sock_recv(channel, min(65536, size - len(raw)))
            if not chunk:
                raise ValueError("owner channel unavailable")
            raw.extend(chunk)
        return raw

    size = struct.unpack("!I", await exact(4))[0]
    if not 0 < size <= MAX_RPC_BYTES:
        raise ValueError("owner RPC exceeds bound")
    message = json.loads(await exact(size))
    if not isinstance(message, dict):
        raise ValueError("invalid owner RPC")
    return message


def encode(message: dict) -> bytes:
    raw = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(raw) > MAX_PACKET:
        raise ValueError("invalid owner packet")
    return raw


async def receive(channel: socket.socket) -> dict:
    raw = await asyncio.get_running_loop().sock_recv(channel, MAX_PACKET + 1)
    if not raw or len(raw) > MAX_PACKET:
        raise ValueError("owner channel unavailable")
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("invalid owner packet")
    return message


def deadline_value(value: object) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("invalid owner deadline")
    return float(value)


async def serve_owner(
    control: socket.socket, health: socket.socket, backend: LifecycleBackend
) -> None:
    lost = asyncio.Event()

    async def watch_health() -> None:
        try:
            while True:
                message = await receive(health)
                if set(message) == {"lease"}:
                    value = message["lease"]
                    backend.pulse(None if value is None else deadline_value(value))
                elif set(message) == {"reject"} and isinstance(message["reject"], str):
                    backend.reject(message["reject"])
                else:
                    raise ValueError("invalid health packet")
        finally:
            backend.disconnect()
            lost.set()

    async def renew() -> None:
        while not lost.is_set():
            await backend.renew()
            await asyncio.sleep(0.25)

    async def dispatch(message: dict) -> dict:
        op = message.get("op")
        fields = {"op", "deadline"}
        if op in {"prepare", "stop", "remove", "execute"}:
            fields.add("allocation")
            if not isinstance(message.get("allocation"), str):
                raise ValueError("invalid allocation")
        if op == "prepare":
            fields.update({"root", "lease"})
        if op == "execute":
            fields.update({"request", "lease"})
        if op == "probe":
            fields.add("root")
        if set(message) != fields:
            raise ValueError("invalid owner operation")
        deadline = deadline_value(message["deadline"])
        result = None
        if op == "recover":
            await backend.recover(deadline=deadline)
        elif op == "prepare":
            if not isinstance(message["root"], str):
                raise ValueError("invalid content root")
            # Cross-channel scheduling must not make the initial pulse race
            # preparation. This is still a Runtime-confirmed absolute lease.
            backend.pulse(deadline_value(message["lease"]))
            result = asdict(
                await backend.prepare(
                    message["allocation"], Path(message["root"]), deadline=deadline
                )
            )
        elif op == "stop":
            await backend.stop(message["allocation"], deadline=deadline)
        elif op == "remove":
            await backend.remove(message["allocation"], deadline=deadline)
        elif op == "execute":
            if not isinstance(message["request"], dict):
                raise ValueError("invalid execution request")
            request = ExecutionRequest(**message["request"])
            backend.pulse(deadline_value(message["lease"]))
            result = asdict(
                await backend.execute(message["allocation"], request, deadline=deadline)
            )
        elif op == "probe":
            if not isinstance(message["root"], str) or len(message["root"]) > 4096:
                raise ValueError("invalid probe root")
            result = await backend.probe(Path(message["root"]), deadline=deadline)
        elif op == "close":
            await backend.close(deadline=deadline)
        else:
            raise ValueError("invalid owner operation")
        return {"result": result}

    watcher = asyncio.create_task(watch_health(), name="podman-owner-health")
    renewer = asyncio.create_task(renew(), name="podman-owner-renew")
    reader: asyncio.Task | None = None
    loss = asyncio.create_task(lost.wait())
    try:
        while not lost.is_set() and not backend.closed:
            reader = asyncio.create_task(receive_rpc(control))
            done, _ = await asyncio.wait({reader, loss}, return_when=asyncio.FIRST_COMPLETED)
            if loss in done:
                break
            message = await reader
            try:
                response = await dispatch(message)
            except SandboxContractError as error:
                response = {"error": error.code.value}
            except Exception:
                # An unexpected backend error invalidates this channel. Never
                # send exception text, paths, image names or CLI diagnostics.
                break
            await send_rpc(control, response)
    except (OSError, ValueError):
        pass
    finally:
        backend.disconnect()
        lost.set()
        for task in (watcher, renewer, reader, loss):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (watcher, renewer, reader, loss) if task is not None),
            return_exceptions=True,
        )
        control.close()
        health.close()
        # No mutation is issued before a successful open. Otherwise keep the
        # owner alive even for an indefinitely uncertain create; a successor
        # must fail startup, never infer that bind deletion is safe from EOF.
        while backend.opened and not backend.closed:
            try:
                await backend.close(deadline=time.monotonic() + 30)
            except Exception:
                await asyncio.sleep(1)


async def _main(control_fd: int, health_fd: int) -> None:
    control = socket.socket(fileno=control_fd)
    health = socket.socket(fileno=health_fd)
    control.setblocking(False)
    health.setblocking(False)
    control.set_inheritable(False)
    health.set_inheritable(False)
    try:
        async with asyncio.timeout(3):
            initial = await receive_rpc(control)
        if set(initial) != {"settings"} or not isinstance(initial["settings"], dict):
            raise ValueError("invalid owner configuration")
        backend = LifecycleBackend(PodmanSettings(**initial["settings"]))
        await serve_owner(control, health, backend)
    finally:
        control.close()
        health.close()


def main() -> None:
    try:
        asyncio.run(_main(int(sys.argv[1]), int(sys.argv[2])))
    except Exception:
        raise SystemExit(70) from None


if __name__ == "__main__":
    main()
