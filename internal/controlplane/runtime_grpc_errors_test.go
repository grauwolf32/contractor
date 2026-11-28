package controlplane

import (
	"context"
	"encoding/json"
	"net"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

type oversizeLifecycle struct {
	pb.UnimplementedRuntimeLifecycleServer
}

func (*oversizeLifecycle) Finalize(context.Context, *pb.FinalizeAllocationRequest) (*pb.AllocationFinalResponse, error) {
	payload := make([]byte, runtimewire.MaxControlBytes+1)
	return &pb.AllocationFinalResponse{Report: &pb.AllocationFinalReport{Worker: &pb.ExecutionReportDocument{Json: payload}}}, nil
}
func TestOversizedLifecycleResponseIsPermanent(t *testing.T) {
	listener := bufconn.Listen(2 << 20)
	server := grpc.NewServer()
	pb.RegisterRuntimeLifecycleServer(server, &oversizeLifecycle{})
	done := make(chan struct{})
	go func() { defer close(done); _ = server.Serve(listener) }()
	defer func() { server.Stop(); _ = listener.Close(); <-done }()
	channel, e := grpc.NewClient("passthrough:///review", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }), grpc.WithDisableRetry())
	if e != nil {
		t.Fatal(e)
	}
	defer channel.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client := pb.NewRuntimeLifecycleClient(channel)
	for _, test := range []struct {
		name string
		call func() error
	}{
		{"finalize", func() error {
			_, err := client.Finalize(ctx, &pb.FinalizeAllocationRequest{}, grpc.MaxCallRecvMsgSize(runtimewire.MaxControlBytes))
			return err
		}},
		{"state", func() error {
			_, err := client.ReadState(ctx, &pb.StateReadRequest{}, grpc.MaxCallRecvMsgSize(runtimewire.MaxStateMessageBytes))
			return err
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			wireError := test.call()
			if status.Code(wireError) != codes.ResourceExhausted {
				t.Fatalf("expected size error, got %v", wireError)
			}
			value, ok := decodeLifecycleError(wireError).(*RuntimeAPIError)
			if !ok || value.Retryable || value.Code != "invalid_response" {
				t.Fatalf("oversized reply must be permanent: %v", value)
			}
		})
	}
}

func TestSharedLifecycleErrorClassification(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name          string
			Status        uint32
			Code          string
			Retryable     bool
			Detail        json.RawMessage
			UnknownDetail bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-error-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			st := status.New(codes.Code(test.Status), "untrusted diagnostics")
			if string(test.Detail) != "null" {
				var detail struct {
					Code      *string
					Retryable *bool
					RequestID *string `json:"requestId"`
				}
				if err := json.Unmarshal(test.Detail, &detail); err != nil {
					t.Fatal(err)
				}
				message := &pb.ErrorDetail{Code: detail.Code, Retryable: detail.Retryable, RequestId: detail.RequestID}
				if test.UnknownDetail {
					message.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 1})
				}
				var err error
				st, err = st.WithDetails(message)
				if err != nil {
					t.Fatal(err)
				}
			}
			failure, ok := decodeLifecycleError(st.Err()).(*RuntimeAPIError)
			if !ok || failure.Code != test.Code || failure.Retryable != test.Retryable {
				t.Fatalf("classification=%v", failure)
			}
		})
	}
}

func (*oversizeLifecycle) ReadState(context.Context, *pb.StateReadRequest) (*pb.StateReadResponse, error) {
	return &pb.StateReadResponse{Revision: proto.Uint64(1), Result: &pb.StateReadResponse_Snapshot{Snapshot: &pb.AgentStateDocument{Json: make([]byte, runtimewire.MaxStateMessageBytes+1)}}}, nil
}
