package privateartifacts

import (
	"context"
	"errors"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
)

func (h *allocationRPCServer) GatewayRecovery(ctx context.Context, request *pb.GatewayRecoveryRequest) (*pb.GatewayRecoveryResponse, error) {
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	grant := scope.grant
	if grant.Lost {
		return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "allocation_unavailable", false)
	}
	if h.owner.recovery == nil {
		return nil, runtimewire.Error(ctx, codes.Unavailable, "recovery_unavailable", true)
	}
	var input gatewayrecovery.Request
	if runtimewire.DecodeDomain(request, &input, maximumRecoveryRequestBytes, maximumRecoveryRequestBytes+rpcEnvelopeBytes) != nil || input.Validate() != nil {
		return nil, runtimewire.Error(ctx, codes.InvalidArgument, "invalid_recovery_request", false)
	}
	activeOnly := !input.Terminal()
	if grant.WriteFenced && activeOnly {
		return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "allocation_unavailable", false)
	}
	var result gatewayrecovery.Decision
	err := h.owner.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.owner.runStoreForGrant(grant.AllocationID, current, scope.identity, false); err != nil ||
			current.WritePolicy != controlplane.WriteInputsAndIntermediates || (current.WriteFenced && activeOnly) {
			return gatewayrecovery.ErrUnavailable
		}
		var err error
		result, err = h.owner.recovery.Update(ctx, current.AllocationID, input)
		return err
	})
	if err != nil {
		if gatewayrecovery.IsUnavailable(err) || errors.Is(err, controlplane.ErrAllocationNotFound) {
			return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "recovery_unavailable", false)
		}
		return nil, runtimewire.Error(ctx, codes.Unavailable, "recovery_unavailable", true)
	}
	response := new(pb.GatewayRecoveryResponse)
	if runtimewire.EncodeDomain(result, response, maximumRecoveryRequestBytes, maximumRecoveryRequestBytes+rpcEnvelopeBytes) != nil {
		return nil, runtimewire.Error(ctx, codes.Internal, "invalid_response", false)
	}
	return response, nil
}
