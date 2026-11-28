package privateartifacts

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type payloadMetadata struct {
	APIVersion        string                `json:"apiVersion"`
	Artifact          contracts.ArtifactRef `json:"artifact"`
	MediaType         string                `json:"mediaType"`
	Size              int64                 `json:"size"`
	BindingCreatedAt  string                `json:"bindingCreatedAt"`
	RevisionCreatedAt string                `json:"revisionCreatedAt"`
}

func payloadRPCMetadata(ctx context.Context, ref contracts.ArtifactRef, mediaType string, size int64, binding, revision time.Time) (*pb.ArtifactPayloadMetadata, error) {
	if (contracts.ArtifactReadResult{APIVersion: contracts.APIVersion, Artifact: ref, MediaType: mediaType, Size: size}).Validate() != nil || size > artifacts.MaxPayloadSize {
		return nil, runtimewire.Error(ctx, codes.Internal, "invalid_response", false)
	}
	return encodeAllocationResponse(ctx, payloadMetadata{contracts.APIVersion, ref, mediaType, size, binding.UTC().Format(time.RFC3339Nano), revision.UTC().Format(time.RFC3339Nano)}, new(pb.ArtifactPayloadMetadata), runtimewire.BulkHeaderBytes)
}

func (scope allocationRPCContext) beginPayload(size int64) {
	deadline := time.Now().Add(artifacttransfer.Duration(size))
	_ = scope.controller.SetReadDeadline(deadline)
	_ = scope.controller.SetWriteDeadline(deadline)
}
func (scope allocationRPCContext) beginStorage() {
	_ = scope.controller.SetReadDeadline(time.Time{})
	_ = scope.controller.SetWriteDeadline(time.Time{})
}
func (scope allocationRPCContext) beforeReceipt() {
	scope.releaseTransfer()
	_ = scope.controller.SetWriteDeadline(time.Now().Add(artifacttransfer.Duration(0)))
}

func bulkReceiveError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if errors.Is(err, io.EOF) || status.Code(err) == codes.ResourceExhausted {
		return errInvalidRequest
	}
	return err
}

func isBulkRPC(method string) bool {
	return method == pb.Artifact_Read_FullMethodName || method == pb.Artifact_Write_FullMethodName || method == pb.Artifact_SubmitFinding_FullMethodName
}

func (h *allocationRPCServer) Read(request *pb.ArtifactReadTarget, stream grpc.ServerStreamingServer[pb.ArtifactReadFrame]) error {
	ctx := stream.Context()
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	scope.beginStorage()
	err := h.withReadTarget(ctx, request, false, func(ctx context.Context, store artifacts.ScopedStore, ref contracts.ArtifactRef) error {
		bounded, cancel := artifacttransfer.StorageContext(ctx)
		result, err := store.Read(bounded, ref)
		cancel()
		if err != nil {
			return err
		}
		metadata, err := payloadRPCMetadata(ctx, result.Ref, result.Payload.MediaType, int64(len(result.Payload.Data)), result.BindingCreatedAt, result.RevisionCreatedAt)
		if err != nil {
			return err
		}
		_ = scope.controller.SetWriteDeadline(time.Now().Add(artifacttransfer.Duration(int64(len(result.Payload.Data)))))
		if err := stream.Send(&pb.ArtifactReadFrame{Frame: &pb.ArtifactReadFrame_Header{Header: metadata}}); err != nil {
			return err
		}
		for offset := 0; offset < len(result.Payload.Data); {
			length := runtimewire.BulkChunkSize(len(result.Payload.Data) - offset)
			if err := stream.Send(&pb.ArtifactReadFrame{Frame: &pb.ArtifactReadFrame_Chunk{Chunk: result.Payload.Data[offset : offset+length]}}); err != nil {
				return err
			}
			offset += length
		}
		return nil
	})
	scope.beforeReceipt()
	if err != nil {
		return allocationDomainError(ctx, err)
	}
	return nil
}

type payloadWriteHeader struct {
	Artifact         contracts.ArtifactRef `json:"artifact" strictjson:"required"`
	MediaType        string                `json:"mediaType" strictjson:"required"`
	Size             *uint64               `json:"size" strictjson:"required"`
	CreateOnly       *bool                 `json:"createOnly,omitempty"`
	ExpectedRevision *string               `json:"expectedRevision,omitempty"`
}

func (h *allocationRPCServer) Write(stream grpc.ClientStreamingServer[pb.ArtifactWriteFrame, pb.ArtifactPayloadMetadata]) error {
	ctx := stream.Context()
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	finish := sync.OnceFunc(scope.beforeReceipt)
	defer finish()
	first, err := stream.Recv()
	if err != nil {
		return allocationDomainError(ctx, bulkReceiveError(ctx, err))
	}
	var input payloadWriteHeader
	if len(first.ProtoReflect().GetUnknown()) != 0 || first.GetHeader() == nil || runtimewire.DecodeDomain(first.GetHeader(), &input, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes) != nil || input.Size == nil || input.Artifact.Validate() != nil || input.Artifact.Revision != nil || !contracts.ValidMediaType(input.MediaType) || (input.CreateOnly == nil) == (input.ExpectedRevision == nil) || input.CreateOnly != nil && !*input.CreateOnly || input.ExpectedRevision != nil && contracts.ValidateOpaqueID("expected revision", *input.ExpectedRevision) != nil {
		return allocationDomainError(ctx, errInvalidRequest)
	}
	if *input.Size > artifacts.MaxPayloadSize {
		return allocationDomainError(ctx, blob.ErrTooLarge)
	}
	if _, err := h.owner.runStoreForGrant(scope.grant.AllocationID, scope.grant, scope.identity, true); err != nil {
		return allocationDomainError(ctx, err)
	}
	scope.beginPayload(int64(*input.Size))
	payload := make([]byte, int(*input.Size))
	for offset := 0; offset < len(payload); {
		frame, err := stream.Recv()
		if err != nil {
			return allocationDomainError(ctx, bulkReceiveError(ctx, err))
		}
		length := runtimewire.BulkChunkSize(len(payload) - offset)
		if len(frame.ProtoReflect().GetUnknown()) != 0 || frame.GetHeader() != nil || len(frame.GetChunk()) != length {
			return allocationDomainError(ctx, errInvalidRequest)
		}
		copy(payload[offset:offset+length], frame.GetChunk())
		offset += length
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errInvalidRequest
		} else {
			err = bulkReceiveError(ctx, err)
		}
		return allocationDomainError(ctx, err)
	}
	scope.beginStorage()
	bounded, cancel := artifacttransfer.StorageContext(ctx)
	defer cancel()
	result, err := h.owner.writeArtifactPayload(bounded, scope.grant.AllocationID, scope.identity, input.Artifact, input.MediaType, payload, input.ExpectedRevision)
	finish()
	if err != nil {
		return allocationDomainError(ctx, err)
	}
	metadata, err := payloadRPCMetadata(ctx, result.Ref, result.MediaType, result.Size, result.BindingCreatedAt, result.RevisionCreatedAt)
	if err != nil {
		return err
	}
	return stream.SendAndClose(metadata)
}
