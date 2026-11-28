"""Private protobuf preserves the existing shared domain contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contractor_runtime._grpc_client import _decode_response
from contractor_runtime.contracts.base import WireModel
from contractor_runtime.contracts.grpc_wire import codec, limits
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.grpc_server import _decode_request

FIXTURES = Path(__file__).parents[2] / "api/testdata/v1alpha1"
INDEX = json.loads((FIXTURES / "index.json").read_text())
TYPES = {
    "SharedConfig": pb.SharedConfig,
    "AgentHeartbeat": pb.AgentHeartbeat,
    "HeartbeatResponse": pb.HeartbeatResponse,
    "AgentRegistration": pb.AgentRegistration,
    "AgentRegistrationResponse": pb.AgentRegistrationResponse,
    "AllocationSpec": pb.AllocationSpec,
    "AllocationWorkspaceSpec": pb.AllocationWorkspace,
    "FinalizeAllocationRequest": pb.FinalizeAllocationRequest,
    "AbortAllocationRequest": pb.AbortAllocationRequest,
    "ReleaseAllocationRequest": pb.ReleaseAllocationRequest,
    "AllocationFinalResponse": pb.AllocationFinalResponse,
    "WorkspaceCapabilities": pb.WorkspaceCapabilities,
}


@pytest.mark.parametrize(
    ("name", "model"),
    [(name, TYPES[item["type"]]) for name, item in INDEX["valid"].items() if item["type"] in TYPES],
)
def test_shared_protobuf_domain_round_trip(name, model):
    value = json.loads((FIXTURES / "valid" / name).read_bytes())
    request = codec.encode(value, model())
    parsed = model.FromString(request.SerializeToString())
    assert codec.decode(parsed) == value


def test_shared_protobuf_presence_and_unknown_fields():
    for value in ({}, {"allocationSlots": 0}, {"allocations": []}, {"capabilities": None}):
        assert codec.decode(codec.encode(value, pb.AgentRegistration())) == value
    message = pb.AgentRegistration.FromString(b"\xf8\x07\x01")
    with pytest.raises(codec.InvalidWireValue):
        codec.decode(message)
    message = pb.AgentRegistration()
    message.supported_runtimes.ParseFromString(b"\xf8\x07\x01")
    with pytest.raises(codec.InvalidWireValue):
        codec.decode(message)
    message = pb.AgentRegistration(capabilities=pb.NullableCompletionCapabilities())
    with pytest.raises(codec.InvalidWireValue):
        codec.decode(message)


@pytest.mark.parametrize(
    "raw",
    [b'{"a":1,"a":2}', b'{"a":NaN}', b"{} {}", b"null", b'{"a":' * 10000 + b"0" + b"}" * 10000],
)
def test_shared_domain_document_requires_strict_json(raw):
    message = pb.AllocationSpec(agent_template=pb.AgentTemplateDocument(json=raw))
    with pytest.raises(codec.InvalidWireValue):
        codec.decode(message)


def test_shared_grpc_bounds():
    assert json.loads((FIXTURES / "grpc-limits.json").read_bytes()) == {
        "controlMessageBytes": limits.MAX_CONTROL_BYTES,
        "a2aMessageBytes": limits.MAX_A2A_BYTES,
        "stateMessageBytes": limits.MAX_STATE_MESSAGE_BYTES,
        "metadataBytes": limits.MAX_METADATA_BYTES,
        "concurrentRPCs": limits.MAX_CONCURRENT_RPCS,
        "bulkChunkBytes": limits.BULK_CHUNK_BYTES,
        "bulkHeaderBytes": limits.BULK_HEADER_BYTES,
        "bulkFrameBytes": limits.BULK_FRAME_BYTES,
        "bulkServerWindowBytes": 1 << 20,  # Server-owned receive window.
        "bulkClientLookaheadBytes": limits.BULK_LOOKAHEAD_BYTES,
    }


def test_shared_protobuf_sequence_precision():
    cases = json.loads((FIXTURES / "grpc-sequences.json").read_bytes())
    for raw in cases["valid"]:
        value = {"heartbeatSeq": json.loads(raw)}
        message = codec.encode(value, pb.AgentHeartbeat())
        parsed = pb.AgentHeartbeat.FromString(message.SerializeToString())
        assert codec.decode(parsed) == value
    for raw in cases["invalid"]:
        with pytest.raises(codec.InvalidWireValue):
            codec.encode({"heartbeatSeq": json.loads(raw)}, pb.AgentHeartbeat())


def test_shared_allocation_protobuf_shapes():
    cases = json.loads((FIXTURES / "grpc-allocation-cases.json").read_bytes())
    assert len(cases["valid"]) >= 18 and len(cases["invalid"]) >= 6
    for case in cases["valid"]:
        model = getattr(pb, case["type"])
        request = codec.encode(case["value"], model())
        response = model.FromString(request.SerializeToString())
        assert codec.decode(response) == case["value"]
        response.MergeFromString(b"\xf8\x07\x01")
        with pytest.raises(codec.InvalidWireValue):
            codec.decode(response)
    for case in cases["invalid"]:
        with pytest.raises(codec.InvalidWireValue):
            codec.encode(case["value"], getattr(pb, case["type"])())


class LifecycleText(WireModel):
    text: str


def test_shared_lifecycle_compact_utf8_size():
    cases = json.loads((FIXTURES / "grpc-json-size-cases.json").read_bytes())["cases"]
    for case in cases:
        assert len(case["unit"].encode()) == case["utf8Bytes"]
        for extra in (0, 1):
            size = limits.MAX_CONTROL_BYTES - len(b'{"text":""}') + extra
            value = {
                "text": case["unit"] * (size // case["utf8Bytes"])
                + "a" * (size % case["utf8Bytes"])
            }
            raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            assert len(raw) == limits.MAX_CONTROL_BYTES + extra
            message = codec.encode(value, pb.ResolvedInstructions())
            for decode in (_decode_request, _decode_response):
                if extra:
                    with pytest.raises(codec.InvalidWireValue):
                        decode(message, LifecycleText)
                else:
                    decoded = decode(message, LifecycleText)
                    assert (
                        decoded.text if isinstance(decoded, LifecycleText) else decoded["text"]
                    ) == value["text"]
    with pytest.raises(codec.InvalidWireValue):
        _decode_request(pb.ResolvedInstructions(text="a" * limits.MAX_CONTROL_BYTES), LifecycleText)
