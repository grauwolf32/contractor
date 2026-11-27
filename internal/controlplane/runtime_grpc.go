package controlplane

import (
	"context"
	"errors"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func (c *RuntimeControlClient) callLifecycle(ctx context.Context, origin, allocationID, principal, operation string, request contracts.Validatable, response contracts.Validatable, validate bool) error {
	if c.channels == nil {
		return errors.New("Runtime gRPC channels are unavailable")
	}
	var input, output proto.Message
	var method string
	switch operation {
	case "prepare":
		input = &pb.PrepareAllocationRequest{}
		output = &pb.PrepareAllocationResponse{}
		method = pb.RuntimeLifecycle_Prepare_FullMethodName
	case "finalize":
		input = &pb.FinalizeAllocationRequest{}
		output = &pb.AllocationFinalResponse{}
		method = pb.RuntimeLifecycle_Finalize_FullMethodName
	case "abort":
		input = &pb.AbortAllocationRequest{}
		output = &pb.AllocationFinalResponse{}
		method = pb.RuntimeLifecycle_Abort_FullMethodName
	case "release":
		input = &pb.ReleaseAllocationRequest{}
		output = &pb.ReleaseAllocationResponse{}
		method = pb.RuntimeLifecycle_Release_FullMethodName
	default:
		return errors.New("unknown Runtime lifecycle operation")
	}
	defer runtimewire.Clear(input)
	if err := runtimewire.Encode(request, input); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestid.Ensure(ctx))
	channel, release, err := c.channels.Borrow(ctx, origin, principal)
	if err != nil {
		return err
	}
	defer release()
	if err := channel.Invoke(ctx, method, input, output, grpc.WaitForReady(false)); err != nil {
		rejected := c.channels.PeerRejected(channel)
		c.channels.Evict(principal)
		if rejected {
			return &RuntimeAPIError{RPCStatus: codes.Unauthenticated, Code: "invalid_worker_handle", Retryable: false}
		}
		return decodeLifecycleError(err)
	}
	if operation == "release" {
		if len(output.ProtoReflect().GetUnknown()) != 0 {
			return runtimewire.ErrInvalid
		}
		return nil
	}
	if err := runtimewire.Decode(output, response); err != nil {
		c.channels.Evict(principal)
		return err
	}
	if validate {
		return response.Validate()
	}
	return nil
}

func decodeLifecycleError(err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return errors.New("Runtime RPC failed")
	}
	failure := &RuntimeAPIError{RPCStatus: st.Code(), Code: "transport_unavailable", Retryable: st.Code() == codes.Unavailable || st.Code() == codes.DeadlineExceeded || st.Code() == codes.ResourceExhausted}
	details := st.Details()
	if len(details) == 0 {
		return failure
	}
	failure.Code = "invalid_error_response"
	failure.Retryable = false
	if len(details) != 1 {
		return failure
	}
	detail, ok := details[0].(*pb.ErrorDetail)
	if !ok || detail.Code == nil || detail.Retryable == nil || detail.RequestId == nil || !requestid.Valid(detail.GetRequestId()) || len(detail.ProtoReflect().GetUnknown()) != 0 {
		return failure
	}
	if !validPrivateErrorCode(detail.GetCode()) {
		return failure
	}
	failure.Code = detail.GetCode()
	failure.Retryable = detail.GetRetryable()
	return failure
}
func validPrivateErrorCode(value string) bool {
	if len(value) == 0 || len(value) > 128 {
		return false
	}
	for _, c := range value {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
			return false
		}
	}
	return true
}
