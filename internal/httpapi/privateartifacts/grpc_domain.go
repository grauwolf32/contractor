package privateartifacts

import (
	"context"
	"errors"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
	"net/http"
)

func (f privateFailure) GRPCStatus() codes.Code {
	switch f.status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return codes.InvalidArgument
	case http.StatusNotFound:
		return codes.NotFound
	case http.StatusForbidden:
		return codes.PermissionDenied
	case http.StatusUnauthorized:
		return codes.Unauthenticated
	case http.StatusConflict:
		return codes.FailedPrecondition
	case http.StatusRequestEntityTooLarge:
		return codes.ResourceExhausted
	case http.StatusServiceUnavailable:
		if f.code == "artifact_transfer_capacity" {
			return codes.ResourceExhausted
		}
		if f.code == "storage_transaction_conflict" {
			return codes.Aborted
		}
		return codes.Unavailable
	}
	return codes.Internal
}
func allocationDomainError(ctx context.Context, err error) error {
	if errors.Is(err, context.Canceled) {
		return runtimewire.Error(ctx, codes.Canceled, "cancelled", false)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return runtimewire.Error(ctx, codes.DeadlineExceeded, "deadline_exceeded", true)
	}
	failure := classifyPrivateError(err)
	if failure.code == "artifact_history_maintenance_pending" {
		return runtimewire.ErrorWithRetry(ctx, failure.GRPCStatus(), failure.Code(), failure.Retryable(), artifactpostgres.HistoryRetryAfter)
	}
	return runtimewire.Error(ctx, failure.GRPCStatus(), failure.Code(), failure.Retryable())
}

func encodeAllocationResponse[T proto.Message](ctx context.Context, value any, response T, limit int) (T, error) {
	if err := runtimewire.EncodeDomain(value, response, limit, limit+rpcEnvelopeBytes); err != nil {
		return response, runtimewire.Error(ctx, codes.Internal, "invalid_response", false)
	}
	return response, nil
}

func (h *allocationRPCServer) withGrant(ctx context.Context, operation func(context.Context, controlplane.AllocationGrant) error) error {
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	bounded, cancel := artifacttransfer.StorageContext(ctx)
	defer cancel()
	return h.owner.dependencies.Registry.WithWriteGrant(scope.grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.owner.runStoreForGrant(scope.grant.AllocationID, current, scope.identity, false); err != nil {
			return err
		}
		return operation(bounded, current)
	})
}

func isAuditRPC(method string) bool {
	return method == pb.Allocation_SubmitAuditEvidence_FullMethodName || method == pb.Allocation_SubmitAuditJudgments_FullMethodName
}
