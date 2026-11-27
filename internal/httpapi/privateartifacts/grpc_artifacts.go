package privateartifacts

import (
	"context"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runtimewire"
)

type artifactListRPC struct {
	Namespace  *string `json:"namespace,omitempty"`
	NamePrefix *string `json:"namePrefix,omitempty"`
	Limit      int     `json:"limit" strictjson:"required"`
	Cursor     *string `json:"cursor,omitempty"`
}
type artifactTargetRPC struct {
	Artifact  contracts.ArtifactRef `json:"artifact" strictjson:"required"`
	Scope     *string               `json:"scope,omitempty"`
	Operation *string               `json:"operation,omitempty"`
}

func (h *allocationRPCServer) ListArtifacts(ctx context.Context, request *pb.ArtifactListRequest) (*pb.ArtifactListResponse, error) {
	var input artifactListRPC
	if runtimewire.DecodeDomain(request, &input, 1<<20, (1<<20)+rpcEnvelopeBytes) != nil || input.Namespace != nil && *input.Namespace == "" || input.NamePrefix != nil && (input.Namespace == nil || input.Cursor != nil) {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	query := artifacts.RefPageQuery{Namespace: input.Namespace, ModelVisible: true, Limit: input.Limit}
	if input.Cursor != nil {
		cursor, err := decodeListCursor(*input.Cursor)
		if err != nil {
			return nil, allocationDomainError(ctx, err)
		}
		query.AfterNamespace, query.AfterName = cursor.Namespace, cursor.Name
	}
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	store, err := h.owner.runStoreForGrant(scope.grant.AllocationID, scope.grant, scope.identity, false)
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	bounded, cancel := artifacttransfer.StorageContext(ctx)
	defer cancel()
	result := contracts.ArtifactListResult{APIVersion: contracts.APIVersion}
	if input.NamePrefix != nil {
		result.Artifacts, err = store.ListPrefix(bounded, *input.Namespace, *input.NamePrefix, input.Limit)
	} else {
		var page artifacts.RefPage
		page, err = store.ListRefs(bounded, query)
		result.Artifacts = page.Refs
		if err == nil && page.More {
			var cursor string
			cursor, err = encodeListCursor(page.Refs[len(page.Refs)-1])
			result.NextCursor = &cursor
		}
	}
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	if err = result.Validate(); err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, result, new(pb.ArtifactListResponse), 1<<20)
}
func (h *allocationRPCServer) withReadTarget(ctx context.Context, request *pb.ArtifactReadTarget, access bool, read func(context.Context, artifacts.ScopedStore, contracts.ArtifactRef) error) error {
	var input artifactTargetRPC
	if runtimewire.DecodeDomain(request, &input, 1<<20, (1<<20)+rpcEnvelopeBytes) != nil {
		return errInvalidRequest
	}
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	return h.owner.withArtifactTarget(ctx, scope.grant.AllocationID, scope.identity, input.Artifact, input.Scope, input.Operation, access, read)
}
func (h *allocationRPCServer) ReadArtifactMetadata(ctx context.Context, request *pb.ArtifactReadTarget) (*pb.ArtifactMetadataResponse, error) {
	var result contracts.ArtifactMetadataResult
	err := h.withReadTarget(ctx, request, false, func(ctx context.Context, store artifacts.ScopedStore, ref contracts.ArtifactRef) error {
		bounded, cancel := artifacttransfer.StorageContext(ctx)
		defer cancel()
		metadata, err := store.Metadata(bounded, ref)
		if err != nil {
			return err
		}
		result = contracts.ArtifactMetadataResult{APIVersion: contracts.APIVersion, Artifact: metadata.Ref, MediaType: metadata.MediaType, Size: metadata.Size, ContentCacheKey: metadata.ContentCacheKey}
		return result.Validate()
	})
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return encodeAllocationResponse(ctx, result, new(pb.ArtifactMetadataResponse), 1<<20)
}
func (h *allocationRPCServer) CheckArtifactAccess(ctx context.Context, request *pb.ArtifactReadTarget) (*pb.ArtifactAccessResponse, error) {
	err := h.withReadTarget(ctx, request, true, func(context.Context, artifacts.ScopedStore, contracts.ArtifactRef) error { return nil })
	if err != nil {
		return nil, allocationDomainError(ctx, err)
	}
	return new(pb.ArtifactAccessResponse), nil
}
