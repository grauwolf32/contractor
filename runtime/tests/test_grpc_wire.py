"""Private protobuf preserves the existing shared domain contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contractor_runtime.contracts.grpc_wire import codec, limits
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb

FIXTURES = Path(__file__).parents[2] / "api/testdata/v1alpha1"
INDEX = json.loads((FIXTURES / "index.json").read_text())
TYPES = {
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
        "metadataBytes": limits.MAX_METADATA_BYTES,
        "concurrentRPCs": limits.MAX_CONCURRENT_RPCS,
    }
