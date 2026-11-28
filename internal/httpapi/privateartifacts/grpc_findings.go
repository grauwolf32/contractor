package privateartifacts

import (
	"context"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runtimewire"
)

func (h *allocationRPCServer) withFindings(ctx context.Context, operation func(context.Context, controlplane.AllocationGrant) error) error {
	if h.owner.dependencies.ProjectFindings == nil {
		return errArtifactAccessDenied
	}
	return h.withGrant(ctx, operation)
}
func (h *allocationRPCServer) ListFindings(ctx context.Context, request *pb.FindingListRequest) (*pb.FindingListResponse, error) {
	var input struct {
		Limit  *int   `json:"limit,omitempty"`
		Cursor string `json:"cursor,omitempty"`
	}
	if runtimewire.DecodeDomain(request, &input, 1<<20, (1<<20)+rpcEnvelopeBytes) != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	limit := 20
	if input.Limit != nil {
		limit = *input.Limit
	}
	if limit < 1 || limit > 100 {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result findingintake.AllocationFindingPage
	err := h.withFindings(ctx, func(ctx context.Context, grant controlplane.AllocationGrant) error {
		var err error
		result, err = h.owner.dependencies.ProjectFindings.ListAllocationFindings(ctx, grant, limit, input.Cursor)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, result, new(pb.FindingListResponse), 256<<10)
}
func (h *allocationRPCServer) ReadFinding(ctx context.Context, request *pb.FindingReadRequest) (*pb.ProjectFinding, error) {
	var input struct {
		FindingID string `json:"findingId" strictjson:"required"`
		Revision  uint64 `json:"revision" strictjson:"required"`
	}
	if runtimewire.DecodeDomain(request, &input, 1<<20, (1<<20)+rpcEnvelopeBytes) != nil || !contracts.ValidResourceID(input.FindingID) || input.Revision == 0 {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result auditstore.ProjectFinding
	err := h.withFindings(ctx, func(ctx context.Context, grant controlplane.AllocationGrant) error {
		var err error
		result, err = h.owner.dependencies.ProjectFindings.ReadAllocationFinding(ctx, grant, input.FindingID, input.Revision)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, result, new(pb.ProjectFinding), 256<<10)
}
func (h *allocationRPCServer) SelectFindings(ctx context.Context, request *pb.FindingSelectionRequest) (*pb.FindingSelectionResponse, error) {
	var input findingintake.FindingSelection
	if runtimewire.DecodeDomain(request, &input, 1<<20, (1<<20)+rpcEnvelopeBytes) != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result findingintake.PublishedCollection
	err := h.withFindings(ctx, func(ctx context.Context, grant controlplane.AllocationGrant) error {
		if grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
			return errArtifactAccessDenied
		}
		var err error
		result, err = h.owner.dependencies.ProjectFindings.SelectAllocationFindings(ctx, grant, input)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, result, new(pb.FindingSelectionResponse), 256<<10)
}
