package runtimewire

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts"
	"reflect"
	"strings"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"google.golang.org/protobuf/proto"
)

func TestSharedProtobufDomainRoundTrip(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "index.json")
	var index struct {
		Valid map[string]struct {
			Type string `json:"type"`
		} `json:"valid"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatal(err)
	}
	types := map[string]proto.Message{
		"SharedConfig": &pb.SharedConfig{}, "AgentHeartbeat": &pb.AgentHeartbeat{}, "HeartbeatResponse": &pb.HeartbeatResponse{},
		"AgentRegistration": &pb.AgentRegistration{}, "AgentRegistrationResponse": &pb.AgentRegistrationResponse{},
		"AllocationSpec": &pb.AllocationSpec{}, "AllocationWorkspaceSpec": &pb.AllocationWorkspace{},
		"FinalizeAllocationRequest": &pb.FinalizeAllocationRequest{}, "AbortAllocationRequest": &pb.AbortAllocationRequest{},
		"ReleaseAllocationRequest": &pb.ReleaseAllocationRequest{}, "AllocationFinalResponse": &pb.AllocationFinalResponse{},
		"WorkspaceCapabilities": &pb.WorkspaceCapabilities{},
	}
	count := 0
	for name, item := range index.Valid {
		model, ok := types[item.Type]
		if !ok {
			continue
		}
		count++
		t.Run(name, func(t *testing.T) {
			raw := contractstest.ReadFixture(t, "valid", name)
			message := model.ProtoReflect().New().Interface()
			if err := encodeMessage(raw, message.ProtoReflect()); err != nil {
				t.Fatal(err)
			}
			binary, err := proto.Marshal(message)
			if err != nil {
				t.Fatal(err)
			}
			parsed := model.ProtoReflect().New().Interface()
			if err := proto.Unmarshal(binary, parsed); err != nil {
				t.Fatal(err)
			}
			result, err := decodeMessage(parsed.ProtoReflect())
			if err != nil {
				t.Fatal(err)
			}
			var a, b any
			if err := json.Unmarshal(raw, &a); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(result, &b); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(a, b) {
				t.Fatal("protobuf changed domain values")
			}
		})
	}
	if count < 20 {
		t.Fatalf("only %d fixtures selected", count)
	}
}

func TestProtobufPresenceAndUnknownFields(t *testing.T) {
	for _, raw := range []string{`{}`, `{"allocationSlots":0}`, `{"allocations":[]}`, `{"capabilities":null}`} {
		message := &pb.AgentRegistration{}
		if err := encodeMessage([]byte(raw), message.ProtoReflect()); err != nil {
			t.Fatal(err)
		}
		result, err := decodeMessage(message.ProtoReflect())
		if err != nil || string(result) != raw {
			t.Fatalf("presence changed: %s %v", result, err)
		}
	}
	for _, nested := range []bool{false, true} {
		message := &pb.AgentRegistration{}
		if nested {
			message.SupportedRuntimes = &pb.ListOfString{}
			message.SupportedRuntimes.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
		} else {
			message.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
		}
		if _, err := decodeMessage(message.ProtoReflect()); err == nil {
			t.Fatal("accepted unknown wire field")
		}
	}
	for _, raw := range []string{`{"a":1,"a":2}`, `{"a":NaN}`, `{} {}`, `null`} {
		message := &pb.AllocationSpec{AgentTemplate: &pb.AgentTemplateDocument{Json: []byte(raw)}}
		if _, err := decodeMessage(message.ProtoReflect()); err == nil {
			t.Fatal("accepted invalid domain document")
		}
	}
}

func TestSharedGRPCBounds(t *testing.T) {
	var bounds map[string]int
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-limits.json"), &bounds); err != nil {
		t.Fatal(err)
	}
	actual := map[string]int{"controlMessageBytes": MaxControlBytes, "a2aMessageBytes": MaxA2ABytes, "stateMessageBytes": MaxStateMessageBytes, "metadataBytes": MaxMetadataBytes, "concurrentRPCs": MaxConcurrentRPCs, "bulkChunkBytes": BulkChunkBytes, "bulkHeaderBytes": BulkHeaderBytes, "bulkFrameBytes": BulkFrameBytes, "bulkServerWindowBytes": BulkWindowBytes, "bulkClientLookaheadBytes": 64 << 10}
	if !reflect.DeepEqual(bounds, actual) {
		t.Fatalf("gRPC bounds drift: %v != %v", actual, bounds)
	}
}

func TestSharedProtobufSequencePrecision(t *testing.T) {
	var cases struct{ Valid, Invalid []string }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-sequences.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, value := range cases.Valid {
		raw := []byte(`{"heartbeatSeq":` + value + `}`)
		request := &pb.AgentHeartbeat{}
		if err := encodeMessage(raw, request.ProtoReflect()); err != nil {
			t.Fatal(err)
		}
		binary, err := proto.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		parsed := &pb.AgentHeartbeat{}
		if err := proto.Unmarshal(binary, parsed); err != nil {
			t.Fatal(err)
		}
		result, err := decodeMessage(parsed.ProtoReflect())
		if err != nil || string(result) != string(raw) {
			t.Fatalf("sequence %s changed: %s, %v", value, result, err)
		}
	}
	for _, value := range cases.Invalid {
		if err := encodeMessage([]byte(`{"heartbeatSeq":`+value+`}`), (&pb.AgentHeartbeat{}).ProtoReflect()); err == nil {
			t.Fatalf("invalid sequence %s accepted", value)
		}
	}
}

type lifecycleText struct {
	Text string `json:"text"`
}

func (v lifecycleText) Validate() error {
	if v.Text == "" {
		return ErrInvalid
	}
	return nil
}
func TestSharedLifecycleCompactUTF8Size(t *testing.T) {
	var cases struct {
		Cases []struct {
			Name, Unit string
			UTF8Bytes  int `json:"utf8Bytes"`
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-json-size-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases.Cases {
		t.Run(test.Name, func(t *testing.T) {
			if len(test.Unit) != test.UTF8Bytes {
				t.Fatal("incorrect UTF-8 fixture")
			}
			const overhead = len(`{"text":""}`)
			for _, extra := range []int{0, 1} {
				size := MaxControlBytes - overhead + extra
				value := lifecycleText{Text: strings.Repeat(test.Unit, size/test.UTF8Bytes) + strings.Repeat("a", size%test.UTF8Bytes)}
				raw, err := contracts.EncodeCompactJSON(value)
				if err != nil || len(raw) != MaxControlBytes+extra {
					t.Fatal("incorrect boundary fixture")
				}
				message := &pb.ResolvedInstructions{}
				err = Encode(value, message)
				if extra == 1 {
					if err == nil {
						t.Fatal("accepted oversized domain")
					}
					continue
				}
				if err != nil {
					t.Fatalf("compact UTF-8 request rejected: %v", err)
				}
				var decoded lifecycleText
				if err := Decode(message, &decoded); err != nil || decoded != value {
					t.Fatal("domain round trip failed", err)
				}
			}
			// Independently reject an oversized protobuf message even before projection.
			if Decode(&pb.ResolvedInstructions{Text: proto.String(strings.Repeat("a", MaxControlBytes))}, &lifecycleText{}) == nil {
				t.Fatal("accepted oversized protobuf")
			}
		})
	}
}
