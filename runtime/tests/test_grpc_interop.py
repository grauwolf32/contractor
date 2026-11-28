"""Real TLS and official A2A SDK interop prerequisite for the gRPC migration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import grpc
import pytest
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.grpc_handler import GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from fakes.grpc_tls import TLSAdapter
from google.protobuf.wrappers_pb2 import BytesValue
from test_mtls import PKI, generate_pki

from contractor_runtime.a2a_server import ContractorAgentExecutor, build_agent_card
from contractor_runtime.contracts import API_VERSION, WorkerCompletion, WorkerFailure
from contractor_runtime.mtls import runtime_agent_client_context, runtime_agent_server_context

ROOT = Path(__file__).resolve().parents[2]


def test_adapter_bounds_forwarding_and_joins_open_connections():
    async def scenario():
        adapter = TLSAdapter(maximum=8)
        admitted = asyncio.Event()
        entered = 0

        async def connect(_writer):
            nonlocal entered
            entered += 1
            if entered == adapter.maximum:
                admitted.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as local:
            path = Path(local) / "admission.sock"
            adapter.server = await asyncio.start_unix_server(adapter.accept(connect), path=path)
            clients = []
            try:
                for _ in range(adapter.maximum):
                    clients.append(await asyncio.open_unix_connection(path))
                await asyncio.wait_for(admitted.wait(), 3)
                reader, writer = await asyncio.open_unix_connection(path)
                clients.append((reader, writer))
                assert await asyncio.wait_for(reader.read(1), 3) == b""
                assert adapter.rejections == 1
                assert len(adapter.tasks) == adapter.maximum
                # Leave admitted connections open: close must cancel and join
                # forwarding before waiting for the listening server to drain.
                await asyncio.wait_for(adapter.close(), 3)
                assert not adapter.tasks
                for reader, _ in clients:
                    assert await asyncio.wait_for(reader.read(1), 3) == b""
            finally:
                for _, writer in clients:
                    writer.close()
                await asyncio.gather(*(writer.wait_closed() for _, writer in clients))
                await adapter.close()

    asyncio.run(scenario())


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    return generate_pki(tmp_path_factory.mktemp("grpc-pki"))


@pytest.fixture(scope="module")
def go_client(tmp_path_factory):
    binary = tmp_path_factory.mktemp("grpc-go") / "probe"
    subprocess.run(
        ["go", "build", "-o", str(binary), "./runtime/tests/testdata/grpcinterop"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return binary


class ProbeWorker:
    allocation_id = "allocation-probe"

    def __init__(self):
        self.invocations = 0
        self.cancelled = False
        self.active = None

    async def invoke(self, request):
        self.invocations += 1
        self.active = asyncio.current_task()
        try:
            if request.objective == "block":
                await asyncio.Event().wait()
            return WorkerCompletion.model_validate(
                {
                    "apiVersion": API_VERSION,
                    "invocationId": "invocation-probe",
                    "stateRevision": 9007199254740991,
                    "result": {
                        "subtaskId": "0",
                        "result": "done",
                        "summarized": False,
                        "observations": {"profile": "lean@1", "tools": {}, "truncated": False},
                        "artifacts": {},
                    },
                }
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.active = None

    async def failure_completion(self, code, message, *, retryable=False):
        return WorkerCompletion(
            apiVersion=API_VERSION,
            invocationId="invocation-probe",
            stateRevision=1,
            failure=WorkerFailure(code=code, message=message, retryable=retryable),
        )

    def cancel_active(self, owner):
        if self.active is owner:
            owner.cancel()


class AllocationHandler(GrpcHandler):
    def _build_call_context(self, context, request):
        # Production dispatch will select only an active allocation before
        # delegating to its SDK handler. The prototype has exactly one.
        from a2a.types import TaskNotFoundError

        if getattr(request, "tenant", "") != "allocation-probe":
            raise TaskNotFoundError
        return super()._build_call_context(context, request)


def runtime_context(pki: PKI, *, incoming=False):
    factory = runtime_agent_server_context if incoming else runtime_agent_client_context
    return factory(
        ca_file=pki.ca, certificate_file=pki.agent_certificate, private_key_file=pki.agent_key
    )


def principal(pki):
    cert = x509.load_pem_x509_certificate(pki.agent_certificate.read_bytes())
    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki).hexdigest()


@pytest.mark.parametrize("negative", [None, "wrong_role", "wrong_principal"])
def test_official_go_python_a2a_grpc_via_verified_tls(pki, go_client, negative):
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
            server = grpc.aio.server(options=(("grpc.max_receive_message_length", 4 << 20),))
            a2a_pb2_grpc.add_A2AServiceServicer_to_server(handler, server)
            server.add_insecure_port(f"unix:{path}")
            await server.start()
            os.chmod(path, 0o600)
            adapter = TLSAdapter()
            try:
                port = await adapter.inbound(path, runtime_context(pki, incoming=True))
                process = await asyncio.create_subprocess_exec(
                    str(go_client),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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
                try:
                    async with asyncio.timeout(20):
                        out, error = await process.communicate(json.dumps(config).encode())
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                assert process.returncode == 0, error.decode()
                result = json.loads(out)
                if negative:
                    assert result == {"rejected": True}
                    assert worker.invocations == 0
                else:
                    assert result == {
                        "completed": True,
                        "cancellation_delivered": True,
                        "tenant_rejected": True,
                    }
                    assert worker.invocations == 2
                    assert worker.cancelled
            finally:
                await adapter.close()
                await server.stop(0)
                await server.wait_for_termination()
            assert not adapter.tasks
            assert worker.active is None

    asyncio.run(scenario())


@pytest.mark.parametrize("scenario", ["success", "wrong_role", "lose_reply"])
def test_python_outbound_tls_adapter_to_go_preserves_role(pki, go_client, scenario):
    wrong_role = scenario == "wrong_role"
    lose_reply = scenario == "lose_reply"

    async def scenario():
        config = {
            "CA": str(pki.ca),
            "Certificate": str(pki.control_plane_certificate),
            "Key": str(pki.control_plane_key),
            "Principal": principal(pki),
            "Mode": "serve",
        }
        if wrong_role:
            config.update(Certificate=str(pki.agent_certificate), Key=str(pki.agent_key))
        process = await asyncio.create_subprocess_exec(
            str(go_client),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        adapter = TLSAdapter()
        try:
            process.stdin.write(json.dumps(config).encode() + b"\n")
            await process.stdin.drain()
            async with asyncio.timeout(10):
                ready = await process.stdout.readline()
            port = json.loads(ready)["port"]
            with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as local:
                path = Path(local) / "control.sock"
                await adapter.outbound(path, port, runtime_context(pki))
                async with grpc.aio.insecure_channel(
                    f"unix:{path}",
                    options=(("grpc.enable_retries", 0),),
                ) as channel:
                    rpc = channel.unary_unary(
                        "/probe.Control/Register",
                        request_serializer=BytesValue.SerializeToString,
                        response_deserializer=BytesValue.FromString,
                    )
                    if wrong_role:
                        with pytest.raises(grpc.aio.AioRpcError):
                            await rpc(BytesValue(value=b"protected-probe"), timeout=3)
                        assert adapter.verified == 0
                        assert adapter.rejections >= 1
                    elif lose_reply:
                        with pytest.raises(grpc.aio.AioRpcError) as failure:
                            await rpc(BytesValue(value=b"lose-reply"), timeout=3)
                        assert failure.value.code() == grpc.StatusCode.UNAVAILABLE
                        assert adapter.verified == 1
                    else:
                        assert (
                            await rpc(BytesValue(value=b"protected-probe"), timeout=3)
                        ).value == b"accepted"
                        assert adapter.verified == 1
                await adapter.close()
            process.stdin.close()
            async with asyncio.timeout(10):
                out, error = await process.communicate()
            assert process.returncode == 0, error.decode()
            assert json.loads(out)["calls"] == (0 if wrong_role else 1)
        finally:
            await adapter.close()
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert not adapter.tasks

    asyncio.run(scenario())
