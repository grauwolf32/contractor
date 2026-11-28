"""Real TLS admission and cleanup of the production native-gRPC boundary."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import grpc
import pytest
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc
from google.protobuf.wrappers_pb2 import BytesValue
from test_grpc_interop import AllocationHandler, ProbeWorker, principal, runtime_context
from test_grpc_interop import go_client as go_client
from test_grpc_interop import pki as pki

from contractor_runtime._grpc_tls import CONNECTION_LIMIT, TLSForwarder
from contractor_runtime.a2a_server import ContractorAgentExecutor, build_agent_card


@pytest.mark.parametrize("negative", [None, "wrong_role", "wrong_principal"])
def test_incoming_native_a2a_authenticates_actual_connection(pki, go_client, negative):
    async def scenario():
        with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as local:
            path = Path(local) / "a2a.sock"
            worker = ProbeWorker()
            card = build_agent_card(
                allocation_id=worker.allocation_id,
                endpoint="localhost:1",
                logical_agent_name="probe",
                description="test",
                version="1",
            )
            card.supported_interfaces[0].protocol_binding = "GRPC"
            handler = AllocationHandler(
                DefaultRequestHandler(
                    agent_executor=ContractorAgentExecutor(worker),
                    task_store=InMemoryTaskStore(owner_resolver=lambda _: worker.allocation_id),
                    agent_card=card,
                )
            )
            server = grpc.aio.server()
            a2a_pb2_grpc.add_A2AServiceServicer_to_server(handler, server)
            server.add_insecure_port(f"unix:{path}")
            await server.start()
            os.chmod(path, 0o600)
            forwarder = TLSForwarder()
            process = None
            try:
                port = await forwarder.listen(
                    "127.0.0.1", 0, runtime_context(pki, incoming=True), {"h2": path}
                )
                config = {
                    "Address": f"localhost:{port}",
                    "CA": str(pki.ca),
                    "Certificate": str(pki.control_plane_certificate),
                    "Key": str(pki.control_plane_key),
                    "Principal": principal(pki),
                    "Mode": "reject" if negative else "accept",
                }
                if negative == "wrong_role":
                    config.update(Certificate=str(pki.agent_certificate), Key=str(pki.agent_key))
                if negative == "wrong_principal":
                    config["Principal"] = "0" * 64
                process = await asyncio.create_subprocess_exec(
                    str(go_client),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                async with asyncio.timeout(20):
                    out, error = await process.communicate(json.dumps(config).encode())
                assert process.returncode == 0, error.decode()
                if negative:
                    assert json.loads(out) == {"rejected": True}
                    assert worker.invocations == 0
                else:
                    assert json.loads(out)["completed"]
                    assert worker.invocations == 2 and worker.cancelled
            finally:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                await forwarder.close()
                await server.stop(0)
                await server.wait_for_termination()
            assert not forwarder._connections
            assert worker.active is None

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["success", "wrong_role", "lose_reply"])
def test_outgoing_native_grpc_authenticates_actual_connection(pki, go_client, mode):
    async def scenario():
        config = {
            "CA": str(pki.ca),
            "Certificate": str(pki.control_plane_certificate),
            "Key": str(pki.control_plane_key),
            "Principal": principal(pki),
            "Mode": "serve",
        }
        if mode == "wrong_role":
            config.update(Certificate=str(pki.agent_certificate), Key=str(pki.agent_key))
        process = await asyncio.create_subprocess_exec(
            str(go_client),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        forwarder = TLSForwarder()
        try:
            process.stdin.write(json.dumps(config).encode() + b"\n")
            await process.stdin.drain()
            async with asyncio.timeout(10):
                port = json.loads(await process.stdout.readline())["port"]
            with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as local:
                path = Path(local) / "control.sock"
                await forwarder.connect(path, f"https://localhost:{port}", runtime_context(pki))
                async with grpc.aio.insecure_channel(
                    f"unix:{path}", options=(("grpc.enable_retries", 0),)
                ) as channel:
                    rpc = channel.unary_unary(
                        "/probe.Control/Register",
                        request_serializer=BytesValue.SerializeToString,
                        response_deserializer=BytesValue.FromString,
                    )
                    value = b"lose-reply" if mode == "lose_reply" else b"protected-probe"
                    if mode == "success":
                        assert (await rpc(BytesValue(value=value), timeout=3)).value == b"accepted"
                    else:
                        with pytest.raises(grpc.aio.AioRpcError) as failure:
                            await rpc(BytesValue(value=value), timeout=3)
                        assert failure.value.code() == grpc.StatusCode.UNAVAILABLE
                await forwarder.close()
            process.stdin.close()
            async with asyncio.timeout(10):
                out, error = await process.communicate()
            assert process.returncode == 0, error.decode()
            assert json.loads(out)["calls"] == (0 if mode == "wrong_role" else 1)
        finally:
            await forwarder.close()
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert not forwarder._connections

    asyncio.run(scenario())


def test_admission_precedes_tls_and_shutdown_joins_stalled_handshakes(pki):
    async def scenario():
        with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as local:
            forwarder = TLSForwarder()
            port = await forwarder.listen(
                "127.0.0.1", 0, runtime_context(pki, incoming=True), {"h2": Path(local) / "absent"}
            )
            clients = []
            try:
                for _ in range(CONNECTION_LIMIT):
                    clients.append(await asyncio.open_connection("127.0.0.1", port))
                async with asyncio.timeout(3):
                    while len(forwarder._connections) != CONNECTION_LIMIT:
                        await asyncio.sleep(0)
                extra = await asyncio.open_connection("127.0.0.1", port)
                clients.append(extra)
                assert await asyncio.wait_for(extra[0].read(1), 3) == b""
                assert len(forwarder._connections) == CONNECTION_LIMIT
                await asyncio.wait_for(forwarder.close(), 3)
                assert not forwarder._connections
                for reader, _ in clients:
                    assert await asyncio.wait_for(reader.read(1), 3) == b""
            finally:
                for _, writer in clients:
                    writer.close()
                await asyncio.gather(*(writer.wait_closed() for _, writer in clients))
                await forwarder.close()

    asyncio.run(scenario())
