package privateartifacts

import (
	"context"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"testing"
)

type recordingRecovery struct{ requests []gatewayrecovery.Request }

func (r *recordingRecovery) Update(_ context.Context, _ string, input gatewayrecovery.Request) (gatewayrecovery.Decision, error) {
	r.requests = append(r.requests, input)
	return gatewayrecovery.Decision{Allowed: true}, nil
}

func TestGatewayRecoveryFencedAllocationDeliversOnlyTerminalUpdates(t *testing.T) {
	for _, action := range []string{"succeeded", "finished", "released", "acquire", "failed"} {
		t.Run(action, func(t *testing.T) {
			grant := testGrant("run-a")
			grant.WriteFenced = true
			recovery := &recordingRecovery{}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: grant}}, recovery: recovery})
			request := &pb.GatewayRecoveryRequest{Model: proto.String("model"), RequestId: proto.String("request-1"), Action: proto.String(action), RetryAfterSeconds: proto.Float64(0)}
			if action == "failed" {
				request.Code = proto.String("model_unavailable")
			}
			response, err := client.GatewayRecovery(ctx, request)
			want := codes.OK
			if action == "acquire" || action == "failed" {
				want = codes.FailedPrecondition
			}
			if status.Code(err) != want || (len(recovery.requests) == 1) != (want == codes.OK) {
				t.Fatalf("response=%v error=%v calls=%d", response, err, len(recovery.requests))
			}
		})
	}
}
