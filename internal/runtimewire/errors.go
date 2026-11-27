package runtimewire

import (
	"context"
	"encoding/base64"
	"fmt"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/protobuf/types/known/durationpb"
	"net/http"
	"time"

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

// WriteError rejects a request before gRPC decodes its message, for example
// when allocation authority or bounded admission is unavailable.
func WriteError(w http.ResponseWriter, err error) {
	st := status.Convert(err)
	w.Header().Set("Content-Type", "application/grpc")
	w.Header().Set("Grpc-Status", fmt.Sprint(int(st.Code())))
	w.Header().Set("Grpc-Message", st.Message())
	if details, marshalErr := proto.Marshal(st.Proto()); marshalErr == nil {
		w.Header().Set("Grpc-Status-Details-Bin", base64.RawStdEncoding.EncodeToString(details))
	}
	w.WriteHeader(http.StatusOK)
}

// ErrorWithRetry preserves a domain owner's existing retry delay as typed status metadata.
func ErrorWithRetry(ctx context.Context, statusCode codes.Code, code string, retryable bool, delay time.Duration) error {
	result := status.Convert(Error(ctx, statusCode, code, retryable))
	detailed, err := result.WithDetails(&errdetails.RetryInfo{RetryDelay: durationpb.New(delay)})
	if err != nil {
		return result.Err()
	}
	return detailed.Err()
}
