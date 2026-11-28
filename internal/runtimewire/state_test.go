package runtimewire

import (
	"bytes"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"google.golang.org/protobuf/proto"
)

func TestSharedStateResponseRequiresExactRevisionAndOneResult(t *testing.T) {
	document := contractstest.ReadFixture(t, "valid", "agent-state-snapshot.json")
	for _, tc := range []struct {
		name        string
		conditional uint64
		mutate      func(*pb.StateReadResponse)
		valid       bool
	}{
		{"snapshot", 0, func(*pb.StateReadResponse) {}, true},
		{"changed", 6, func(*pb.StateReadResponse) {}, true},
		{"unchanged snapshot", 7, func(*pb.StateReadResponse) {}, false},
		{"missing revision", 0, func(v *pb.StateReadResponse) { v.Revision = nil }, false},
		{"zero revision", 0, func(v *pb.StateReadResponse) { v.Revision = proto.Uint64(0) }, false},
		{"wrong revision", 0, func(v *pb.StateReadResponse) { v.Revision = proto.Uint64(8) }, false},
		{"missing result", 0, func(v *pb.StateReadResponse) { v.Result = nil }, false},
		{"unknown field", 0, func(v *pb.StateReadResponse) { v.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 1}) }, false},
		{"missing document", 0, func(v *pb.StateReadResponse) { v.GetSnapshot().Json = nil }, false},
		{"unknown document", 0, func(v *pb.StateReadResponse) { v.GetSnapshot().ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 1}) }, false},
		{"domain byte ceiling", 0, func(v *pb.StateReadResponse) {
			v.GetSnapshot().Json = bytes.Repeat([]byte(" "), reporting.MaxAgentStateSnapshotBytes+1)
		}, false},
		{"not modified", 7, func(v *pb.StateReadResponse) {
			v.Result = &pb.StateReadResponse_NotModified{NotModified: &pb.StateNotModified{}}
		}, true},
		{"unsolicited not modified", 0, func(v *pb.StateReadResponse) {
			v.Result = &pb.StateReadResponse_NotModified{NotModified: &pb.StateNotModified{}}
		}, false},
		{"wrong conditional", 6, func(v *pb.StateReadResponse) {
			v.Result = &pb.StateReadResponse_NotModified{NotModified: &pb.StateNotModified{}}
		}, false},
		{"unknown conditional result", 7, func(v *pb.StateReadResponse) {
			m := &pb.StateNotModified{}
			m.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 1})
			v.Result = &pb.StateReadResponse_NotModified{NotModified: m}
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			message := &pb.StateReadResponse{Revision: proto.Uint64(7), Result: &pb.StateReadResponse_Snapshot{Snapshot: &pb.AgentStateDocument{Json: bytes.Clone(document)}}}
			tc.mutate(message)
			result, err := WorkerStateResult(message, tc.conditional)
			if (err == nil) != tc.valid {
				t.Fatalf("valid=%v, result=%+v, error=%v", tc.valid, result, err)
			}
			if tc.valid && result.ETag != `"contractor-agent-state-v1-7"` {
				t.Fatal("lost revision")
			}
		})
	}
}
