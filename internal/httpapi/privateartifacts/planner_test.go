package privateartifacts

import (
	"context"
	"encoding/json"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type plannerOperationsFunc func(context.Context, string, string, contracts.PlannerOperation) (contracts.PlannerOperationResult, error)

func (f plannerOperationsFunc) ExecuteOperation(ctx context.Context, stage, allocation string, request contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
	return f(ctx, stage, allocation, request)
}

func TestPrivatePlannerOperationsUseAllocationAuthorityAndRejectMalformedRequests(t *testing.T) {
	for _, name := range []string{"admitted", "missing-sequence", "unknown-field", "unknown-nested", "nul-argument", "missing-arguments", "ending", "lost"} {
		t.Run(name, func(t *testing.T) {
			grant := testGrant("run-a")
			grant.WriteFenced = name == "ending"
			grant.Lost = name == "lost"
			registry := &fakeRegistry{grant: grant}
			calls := 0
			operations := plannerOperationsFunc(func(ctx context.Context, stage, allocation string, input contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
				calls++
				if stage != grant.StageExecutionID || allocation != grant.AllocationID || input.Sequence != 1 {
					t.Fatal("request changed authority")
				}
				if _, err := registry.GetGrant(allocation); err != nil {
					t.Fatal(err)
				}
				return contracts.PlannerOperationResult{APIVersion: contracts.APIVersion, Result: json.RawMessage(`{}`)}, nil
			})
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: registry, PlannerOperations: operations}})
			request := &pb.PlannerOperationRequest{ApiVersion: proto.String(contracts.APIVersion), Sequence: proto.Int64(1), Name: proto.String("before_model"), Arguments: &pb.PlannerArgumentsDocument{Json: []byte(`{}`)}}
			switch name {
			case "missing-sequence":
				request.Sequence = nil
			case "unknown-field":
				request.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "unknown-nested":
				request.Arguments.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "nul-argument":
				request.Arguments.Json = []byte(`{"value":"\u0000"}`)
			case "missing-arguments":
				request.Arguments = nil
			}
			response, err := client.PlannerOperation(ctx, request)
			want := codes.InvalidArgument
			if name == "ending" || name == "lost" {
				want = codes.FailedPrecondition
			}
			if name == "admitted" {
				want = codes.OK
			}
			if status.Code(err) != want || (calls == 1) != (want == codes.OK) {
				t.Fatalf("response=%v error=%v calls=%d want=%v", response, err, calls, want)
			}
		})
	}
}
