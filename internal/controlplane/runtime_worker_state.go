package controlplane

import (
	"context"
	"errors"
	"regexp"
	"strconv"
	"strings"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

var agentStateETagPattern = regexp.MustCompile(`^"contractor-agent-state-v1-[1-9][0-9]*"$`)

func (c *RuntimeControlClient) ReadWorkerState(ctx context.Context, handle contracts.WorkerHandle, conditional string) (reporting.WorkerStateReadResult, error) {
	origin, err := c.workerStateEndpoint(handle)
	if err != nil {
		return reporting.WorkerStateReadResult{}, workerStateReadError("worker_state_endpoint_invalid", false)
	}
	return c.readWorkerState(ctx, origin, handle.AllocationID, handle.RuntimeAgentID, conditional)
}

func (c *RuntimeControlClient) readWorkerState(ctx context.Context, origin, allocationID, principal, conditional string) (reporting.WorkerStateReadResult, error) {
	empty := reporting.WorkerStateReadResult{}
	request := &pb.StateReadRequest{AllocationId: proto.String(allocationID)}
	if conditional != "" {
		if !agentStateETagPattern.MatchString(conditional) {
			return empty, workerStateReadError("worker_state_etag_invalid", false)
		}
		revision, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(conditional, `"contractor-agent-state-v1-`), `"`), 10, 64)
		if err != nil {
			return empty, workerStateReadError("worker_state_etag_invalid", false)
		}
		request.IfRevision = proto.Uint64(revision)
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", requestid.Ensure(ctx))
	channel, release, err := c.channels.Borrow(ctx, origin, principal)
	if err != nil {
		if ctx.Err() != nil {
			return empty, workerStateReadError("worker_state_cancelled", true)
		}
		return empty, workerStateReadError("worker_state_transport_failed", true)
	}
	defer release()
	response, err := pb.NewRuntimeLifecycleClient(channel).ReadState(ctx, request,
		grpc.MaxCallRecvMsgSize(runtimewire.MaxStateMessageBytes), grpc.WaitForReady(false))
	if err != nil {
		rejected := c.channels.PeerRejected(channel)
		c.channels.Evict(principal)
		if rejected {
			return empty, workerStateReadError("invalid_worker_handle", false)
		}
		if ctx.Err() != nil {
			return empty, workerStateReadError("worker_state_cancelled", true)
		}
		var typed *RuntimeAPIError
		if errors.As(decodeLifecycleError(err), &typed) {
			if typed.Code == "transport_unavailable" {
				return empty, workerStateReadError("worker_state_transport_failed", typed.Retryable)
			}
			return empty, workerStateReadError(typed.Code, typed.Retryable)
		}
		return empty, workerStateReadError("worker_state_transport_failed", true)
	}
	result, err := runtimewire.WorkerStateResult(response, request.GetIfRevision())
	if err != nil {
		c.channels.Evict(principal)
		return empty, workerStateReadError("worker_state_response_invalid", false)
	}
	return result, nil
}

func (c *RuntimeControlClient) workerStateEndpoint(handle contracts.WorkerHandle) (string, error) {
	if contracts.ValidateResourceID("allocationId", handle.AllocationID) != nil {
		return "", errors.New("allocation ID is invalid")
	}
	interfaces, ok := handle.AgentCard["supportedInterfaces"].([]any)
	if !ok || len(interfaces) != 1 {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	current, ok := interfaces[0].(map[string]any)
	if !ok {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	rawEndpoint, ok := current["url"].(string)
	if !ok {
		return "", errors.New("WorkerHandle has no exact A2A endpoint")
	}
	if current["protocolBinding"] != "GRPC" || current["tenant"] != handle.AllocationID {
		return "", errors.New("WorkerHandle A2A interface is invalid")
	}
	authority, err := mtls.GRPCAuthority("https://" + rawEndpoint)
	if err != nil || authority != rawEndpoint {
		return "", errors.New("WorkerHandle A2A endpoint is invalid")
	}
	return "https://" + authority, nil
}

func workerStateReadError(code string, retryable bool) error {
	return &reporting.WorkerStateReadError{Code: code, Retryable: retryable}
}
