package privateartifacts

import (
	"context"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"slices"
)

type sitemapMutationRPC struct {
	sitemapMutationRequest
	IdempotencyKey string `json:"idempotencyKey" strictjson:"required"`
}
type sitemapSnapshotRPC struct {
	sitemapSnapshotRequest
	IdempotencyKey string `json:"idempotencyKey" strictjson:"required"`
}
type sitemapCollectionRPC struct {
	sitemapCollectionRequest
	IdempotencyKey string `json:"idempotencyKey" strictjson:"required"`
}

func (h *allocationRPCServer) withSitemap(ctx context.Context, allowed func(controlplane.AllocationGrant) bool, operation func(context.Context, projectsitemap.AllocationAccess) error) error {
	bounded, cancel := context.WithTimeout(ctx, projectsitemap.OperationTimeout)
	defer cancel()
	return h.withGrant(bounded, func(ctx context.Context, grant controlplane.AllocationGrant) error {
		if grant.Lost || grant.WriteFenced {
			return errAllocationWriteFenced
		}
		if grant.ReadPolicy != controlplane.ReadCurrentRun || grant.SitemapPolicy == nil || grant.SitemapPolicy.Validate() != nil || !allowed(grant) {
			return errSitemapAccessDenied
		}
		if h.owner.dependencies.Sitemap == nil {
			return errSitemapUnavailable
		}
		access, err := h.owner.dependencies.Sitemap.AllocationAccess(ctx, grant.RunID, grant.Namespace, *grant.SitemapPolicy)
		if err != nil {
			return err
		}
		return operation(ctx, access)
	})
}
func (h *allocationRPCServer) MutateSitemap(ctx context.Context, request *pb.SitemapMutationRequest) (*pb.SitemapMutationResponse, error) {
	var input sitemapMutationRPC
	if runtimewire.DecodeDomain(request, &input, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes) != nil || input.APIVersion != contracts.APIVersion || !contracts.ValidIdempotencyKey(input.IdempotencyKey) || !sitemapDocumentBound(input.sitemapMutationRequest) {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	tool := ""
	switch input.Mutation.Operation {
	case "declare", "annotate":
		tool = contracts.SitemapUpsertTool
	case "remove":
		tool = contracts.SitemapRemoveTool
	case "restore":
		tool = contracts.SitemapRestoreTool
	case "replace-source":
		tool = contracts.SitemapGroupingTool
	default:
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result projectsitemap.Result
	err := h.withSitemap(ctx, func(grant controlplane.AllocationGrant) bool {
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, tool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		var err error
		result, err = h.owner.dependencies.Sitemap.MutateAllocation(ctx, access, input.IdempotencyKey, input.Mutation)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, struct {
		APIVersion string                `json:"apiVersion"`
		Result     projectsitemap.Result `json:"result"`
	}{contracts.APIVersion, result}, new(pb.SitemapMutationResponse), 16<<10)
}
func (h *allocationRPCServer) SnapshotSitemap(ctx context.Context, request *pb.SitemapSnapshotRequest) (*pb.SitemapSnapshotResponse, error) {
	var input sitemapSnapshotRPC
	if runtimewire.DecodeDomain(request, &input, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes) != nil || input.APIVersion != contracts.APIVersion || !contracts.ValidIdempotencyKey(input.IdempotencyKey) || !sitemapDocumentBound(input.sitemapSnapshotRequest) || input.Selection.Validate() != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result projectsitemap.SnapshotResult
	err := h.withSitemap(ctx, func(grant controlplane.AllocationGrant) bool {
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, contracts.SitemapSnapshotTool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		var err error
		result, err = h.owner.dependencies.Sitemap.SnapshotAllocation(ctx, access, input.IdempotencyKey, input.Selection)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, struct {
		APIVersion string                        `json:"apiVersion"`
		Result     projectsitemap.SnapshotResult `json:"result"`
	}{contracts.APIVersion, result}, new(pb.SitemapSnapshotResponse), 16<<10)
}
func (h *allocationRPCServer) CollectSitemap(ctx context.Context, request *pb.SitemapCollectionRequest) (*pb.SitemapMutationResponse, error) {
	var input sitemapCollectionRPC
	if runtimewire.DecodeDomain(request, &input, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes, projectsitemap.MaxBatchBytes+rpcEnvelopeBytes) != nil || input.APIVersion != contracts.APIVersion || !contracts.ValidIdempotencyKey(input.IdempotencyKey) || !sitemapDocumentBound(input.sitemapCollectionRequest) || input.Input.Validate() != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	var result projectsitemap.Result
	err := h.withSitemap(ctx, func(grant controlplane.AllocationGrant) bool {
		if input.Automatic {
			return sitemapCollector(grant) && (input.Input.Artifact == nil || input.Input.Artifact.Namespace == grant.Namespace)
		}
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, contracts.SitemapImportTool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		var err error
		result, err = h.owner.dependencies.Sitemap.CollectAllocation(ctx, access, input.IdempotencyKey, input.Input)
		return err
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, struct {
		APIVersion string                `json:"apiVersion"`
		Result     projectsitemap.Result `json:"result"`
	}{contracts.APIVersion, result}, new(pb.SitemapMutationResponse), 16<<10)
}

func sitemapDocumentBound(value any) bool {
	size, err := contracts.ResultJSONSize(value)
	return err == nil && size <= projectsitemap.MaxBatchBytes
}
