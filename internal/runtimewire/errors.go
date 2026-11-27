package runtimewire

import (
	"context"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Error carries only a static domain code, retry class and bounded correlation
// identity. Raw errors and request/protobuf formatting never reach the peer.
func Error(ctx context.Context, statusCode codes.Code, code string, retryable bool) error {
	result := status.New(statusCode, code)
	withDetails, err := result.WithDetails(&pb.ErrorDetail{
		Code: proto.String(code), Retryable: proto.Bool(retryable),
		RequestId: proto.String(requestid.Ensure(ctx)),
	})
	if err != nil {
		return result.Err()
	}
	return withDetails.Err()
}
