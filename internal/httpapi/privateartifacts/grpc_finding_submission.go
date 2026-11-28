package privateartifacts

import (
	"context"
	"errors"
	"io"
	"sync"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
)

type findingStreamHeader struct {
	APIVersion               string                  `json:"apiVersion" strictjson:"required"`
	InvocationID             string                  `json:"invocationId" strictjson:"required"`
	SubmissionID             string                  `json:"submissionId" strictjson:"required"`
	Incidental               *bool                   `json:"incidental,omitempty"`
	ItemKey                  *string                 `json:"itemKey,omitempty"`
	EvidenceRefs             []contracts.ArtifactRef `json:"evidenceRefs" strictjson:"required"`
	HTTPEvidenceInvocationID *string                 `json:"httpEvidenceInvocationId,omitempty"`
	ProposalSize             *uint64                 `json:"proposalSize" strictjson:"required"`
	DerivedAnchorsSize       *uint64                 `json:"derivedAnchorsSize,omitempty"`
	HTTPEvidenceSize         *uint64                 `json:"httpEvidenceSize,omitempty"`
}

func (input findingStreamHeader) submission() (findingintake.Submission, uint64, error) {
	result := findingintake.Submission{APIVersion: input.APIVersion, InvocationID: input.InvocationID, SubmissionID: input.SubmissionID, ItemKey: input.ItemKey, EvidenceRefs: input.EvidenceRefs}
	if input.Incidental != nil {
		result.Incidental = *input.Incidental
	}
	if input.ProposalSize == nil || *input.ProposalSize < 2 || *input.ProposalSize > auditdomain.MaximumDocumentBytes || len(input.EvidenceRefs) > findingintake.MaxEvidenceRefs || (input.HTTPEvidenceSize == nil) != (input.HTTPEvidenceInvocationID == nil) {
		return result, 0, findingintake.ErrInvalid
	}
	if input.DerivedAnchorsSize != nil && *input.DerivedAnchorsSize > auditdomain.MaximumFindingAnchorDerivationBytes {
		return result, 0, findingintake.ErrInvalid
	}
	projection := map[string]any{"apiVersion": input.APIVersion, "invocationId": input.InvocationID, "submissionId": input.SubmissionID, "evidenceRefs": input.EvidenceRefs}
	if input.Incidental != nil {
		projection["incidental"] = *input.Incidental
	}
	if input.ItemKey != nil {
		projection["itemKey"] = *input.ItemKey
	}
	if input.HTTPEvidenceInvocationID != nil {
		if *input.HTTPEvidenceInvocationID == "" {
			return result, 0, findingintake.ErrInvalid
		}
		result.HTTPEvidenceInvocationID = *input.HTTPEvidenceInvocationID
		projection["httpEvidenceInvocationId"] = *input.HTTPEvidenceInvocationID
	}
	for _, ref := range input.EvidenceRefs {
		if ref.ValidateExact() != nil {
			return result, 0, findingintake.ErrInvalid
		}
	}
	header, err := contracts.EncodeCompactJSON(projection)
	if err != nil {
		return result, 0, findingintake.ErrInvalid
	}
	// Count the complete logical submission without constructing a second
	// complete JSON transport copy. Every declared length is bounded before sum.
	total := *input.ProposalSize
	logical := uint64(len(header)+len(`,"proposal":`)) + total
	for _, document := range []struct {
		length *uint64
		member string
	}{{input.DerivedAnchorsSize, `,"derived_anchors":`}, {input.HTTPEvidenceSize, `,"httpEvidence":`}} {
		if document.length != nil {
			if *document.length == 0 || *document.length > findingintake.MaxRequestBytes {
				return result, 0, findingintake.ErrInvalid
			}
			total += *document.length
			logical += uint64(len(document.member)) + *document.length
		}
	}
	if logical > findingintake.MaxRequestBytes {
		return result, 0, findingintake.ErrInvalid
	}
	return result, total, nil
}

func (h *allocationRPCServer) SubmitFinding(stream grpc.ClientStreamingServer[pb.FindingSubmissionFrame, pb.FindingSubmissionReceipt]) error {
	ctx := stream.Context()
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	finish := sync.OnceFunc(scope.beforeReceipt)
	defer finish()
	if h.owner.dependencies.Findings == nil {
		return allocationDomainError(ctx, errors.New("finding intake is not configured"))
	}
	first, err := stream.Recv()
	if err != nil {
		return allocationDomainError(ctx, bulkReceiveError(ctx, err))
	}
	var header findingStreamHeader
	if len(first.ProtoReflect().GetUnknown()) != 0 || first.GetHeader() == nil || runtimewire.DecodeDomain(first.GetHeader(), &header, runtimewire.BulkHeaderBytes, runtimewire.BulkFrameBytes) != nil {
		return allocationDomainError(ctx, errInvalidRequest)
	}
	input, total, err := header.submission()
	if err != nil {
		return allocationDomainError(ctx, err)
	}
	scope.beginPayload(int64(total))
	buffer := make([]byte, int(total))
	position := 0
	for _, document := range []struct {
		kind   pb.FindingDocumentKind
		length *uint64
	}{{pb.FindingDocumentKind_FINDING_DOCUMENT_PROPOSAL, header.ProposalSize}, {pb.FindingDocumentKind_FINDING_DOCUMENT_DERIVED_ANCHORS, header.DerivedAnchorsSize}, {pb.FindingDocumentKind_FINDING_DOCUMENT_HTTP_EVIDENCE, header.HTTPEvidenceSize}} {
		if document.length == nil {
			continue
		}
		data := buffer[position : position+int(*document.length)]
		for offset := 0; offset < len(data); {
			frame, err := stream.Recv()
			if err != nil {
				return allocationDomainError(ctx, bulkReceiveError(ctx, err))
			}
			chunk := frame.GetChunk()
			length := runtimewire.BulkChunkSize(len(data) - offset)
			if len(frame.ProtoReflect().GetUnknown()) != 0 || chunk == nil || len(chunk.ProtoReflect().GetUnknown()) != 0 || chunk.Document == nil || chunk.GetDocument() != document.kind || len(chunk.GetData()) != length {
				return allocationDomainError(ctx, errInvalidRequest)
			}
			copy(data[offset:offset+length], chunk.GetData())
			offset += length
		}
		position += len(data)
		switch document.kind {
		case pb.FindingDocumentKind_FINDING_DOCUMENT_PROPOSAL:
			if decodeRequestJSON(data, &input.Proposal) != nil {
				return allocationDomainError(ctx, findingintake.ErrInvalid)
			}
		case pb.FindingDocumentKind_FINDING_DOCUMENT_DERIVED_ANCHORS:
			// The intake's existing derivation decoder owns its optional rules.
			if strictjson.HasNULEscape(data) || strictjson.RejectDuplicateKeys(data) != nil {
				return allocationDomainError(ctx, findingintake.ErrInvalid)
			}
			input.DerivedAnchors = data
		case pb.FindingDocumentKind_FINDING_DOCUMENT_HTTP_EVIDENCE:
			input.HTTPEvidence = new(auditdomain.CapturedHTTPExchange)
			if decodeRequestJSON(data, input.HTTPEvidence) != nil {
				return allocationDomainError(ctx, findingintake.ErrInvalid)
			}
		}
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
	receipt, replayed, err := h.owner.submitFinding(bounded, scope.grant, scope.identity, input)
	finish()
	if err != nil {
		return allocationDomainError(ctx, err)
	}
	response, err := encodeAllocationResponse(ctx, findingSubmissionResponse(receipt, replayed), new(pb.FindingSubmissionReceipt), runtimewire.BulkHeaderBytes)
	if err != nil {
		return err
	}
	return stream.SendAndClose(response)
}

func (h *handler) submitFinding(ctx context.Context, grant controlplane.AllocationGrant, identity authenticatedRuntime, input findingintake.Submission) (findingintake.Receipt, bool, error) {
	var receipt findingintake.Receipt
	var replayed bool
	err := h.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, false); err != nil {
			return err
		}
		var found bool
		var err error
		receipt, found, err = h.dependencies.Findings.FindReplay(ctx, current, input)
		if err != nil || found {
			replayed = found
			return err
		}
		if _, err := h.runStoreForGrant(grant.AllocationID, current, identity, true); err != nil {
			return err
		}
		receipt, replayed, err = h.dependencies.Findings.Submit(ctx, current, input)
		return err
	})
	return receipt, replayed, err
}
