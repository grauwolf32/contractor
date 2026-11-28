package privateartifacts

import (
	"context"
	"errors"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
)

func auditRPCError(ctx context.Context, err error, kind string) error {
	var state *runstore.StateConflictError
	switch {
	case errors.Is(err, auditstore.ErrEvidenceBudgetExhausted):
		if kind == "evidence" {
			return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_evidence_limit_exceeded", false)
		}
		return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_result_limit_exceeded", false)
	case errors.Is(err, auditstore.ErrConflict):
		if kind == "evidence" {
			return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_evidence_conflict", false)
		}
		return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_result_conflict", false)
	case errors.Is(err, auditstore.ErrPrecondition), errors.Is(err, runstore.ErrNotFound), errors.Is(err, runstore.ErrPrecondition), errors.Is(err, runstore.ErrProjectDeleting), errors.Is(err, auditstore.ErrProjectDeleting), errors.As(err, &state):
		if kind == "evidence" {
			return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_evidence_closed", false)
		}
		return runtimewire.Error(ctx, codes.FailedPrecondition, "audit_result_closed", false)
	case errors.Is(err, auditstore.ErrInvalid), errors.Is(err, auditservice.ErrInvalid), errors.Is(err, auditstore.ErrNotFound):
		return runtimewire.Error(ctx, codes.InvalidArgument, "invalid_request", false)
	default:
		return allocationDomainError(ctx, err)
	}
}

func (h *allocationRPCServer) SubmitAuditEvidence(ctx context.Context, request *pb.AuditEvidenceRequest) (*pb.AuditEvidenceReceipt, error) {
	var input auditcontract.EvidenceSubmission
	if runtimewire.DecodeDomain(request, &input, auditcontract.MaximumDocumentBytes, auditcontract.MaximumDocumentBytes+rpcEnvelopeBytes) != nil || input.Validate() != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	if h.owner.dependencies.AuditEvidence == nil {
		return nil, runtimewire.Error(ctx, codes.Unavailable, "internal_error", true)
	}
	var receipt auditcontract.EvidenceReceipt
	err := h.withGrant(ctx, func(bounded context.Context, grant controlplane.AllocationGrant) error {
		var err error
		receipt, err = h.owner.dependencies.AuditEvidence.RecordEvidence(bounded, grant, input)
		return err
	})
	if err != nil {
		return nil, auditRPCError(ctx, err, "evidence")
	}
	return encodeAllocationResponse(ctx, receipt, new(pb.AuditEvidenceReceipt), 1<<20)
}
func (h *allocationRPCServer) SubmitAuditJudgments(ctx context.Context, request *pb.AuditJudgmentsRequest) (*pb.AuditJudgmentsResponse, error) {
	var input auditcontract.JudgmentSubmission
	if runtimewire.DecodeDomain(request, &input, auditcontract.MaximumDocumentBytes, auditcontract.MaximumDocumentBytes+rpcEnvelopeBytes) != nil || input.Validate() != nil {
		return nil, allocationDomainError(ctx, errInvalidRequest)
	}
	if h.owner.dependencies.AuditJudgments == nil {
		return nil, runtimewire.Error(ctx, codes.Unavailable, "internal_error", true)
	}
	var receipt auditcontract.JudgmentReceipt
	err := h.withGrant(ctx, func(bounded context.Context, grant controlplane.AllocationGrant) error {
		var err error
		receipt, err = h.owner.dependencies.AuditJudgments.RecordJudgments(bounded, grant, input)
		return err
	})
	var stale *auditservice.JudgmentStaleError
	if errors.As(err, &stale) {
		result, err := encodeAllocationResponse(ctx, struct {
			APIVersion  string                             `json:"apiVersion"`
			Generations []auditcontract.JudgmentGeneration `json:"currentGenerations"`
		}{contracts.APIVersion, stale.Generations}, new(pb.AuditJudgmentStale), 1<<20)
		if err != nil {
			return nil, err
		}
		return &pb.AuditJudgmentsResponse{Result: &pb.AuditJudgmentsResponse_Stale{Stale: result}}, nil
	}
	if err != nil {
		return nil, auditRPCError(ctx, err, "result")
	}
	result, err := encodeAllocationResponse(ctx, receipt, new(pb.AuditJudgmentReceipt), 1<<20)
	if err != nil {
		return nil, err
	}
	return &pb.AuditJudgmentsResponse{Result: &pb.AuditJudgmentsResponse_Receipt{Receipt: result}}, nil
}
