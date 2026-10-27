package auditservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// A changing slice handles one complete candidate closure. The containing
// population can have thousands of receipts; only its indexed next header and
// selected exact source are read, with a durable receipt/ordinal cursor.
func (s *Service) selectTypedCandidate(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot, selection DraftSelection, routes *RoutedCheckContext, budget *auditdomain.FindingRoundBudget, capacity int) (auditstore.AdmissionMutation, error) {
	state := session.State
	mutation := auditstore.AdmissionMutation{Next: state}
	page, err := session.Reads.ReadCandidates(ctx, state.Cursor, 1)
	if err != nil {
		return mutation, err
	}
	if len(page.Candidates) == 0 {
		mutation.Next.Exhausted = page.Exhausted
		return typedSelectionCheckpoint(mutation, budget)
	}
	candidate := page.Candidates[0]
	if candidate.MigrationPending {
		mutation.Next.BlockedReceiptID = candidate.ReceiptID
		mutation.Next.BlockedCode = "source_projection_pending"
		return typedSelectionCheckpoint(mutation, budget)
	}
	mutation.Next.BlockedReceiptID, mutation.Next.BlockedCode = "", ""
	if candidate.NoChecks || candidate.LateSource || candidate.AlreadyConsumed || candidate.AlreadyRejected {
		mutation.Next.Cursor = candidate.Cursor
		mutation.Next.Exhausted = page.Exhausted
		return typedSelectionCheckpoint(mutation, budget)
	}
	decision := auditstore.AdmissionDecision{ReceiptID: candidate.ReceiptID, ProposedCheckOrdinal: candidate.Ordinal, Proposal: candidate.Proposal.Clone(), SourceCheckDigest: candidate.Header.Fragment.Digest, PlanDigest: candidate.Header.PlanDigest}
	reject := func(code string) (auditstore.AdmissionMutation, error) {
		decision.Disposition, decision.Code = "rejected", code
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.Cursor = candidate.Cursor
		mutation.Next.Exhausted = page.Exhausted
		return typedSelectionCheckpoint(mutation, budget)
	}
	if candidate.Header.Status == "unsupported" {
		return reject("unsupported_method")
	}
	if candidate.Header.Status == "plan-too-large" {
		return reject("plan_limit_exceeded")
	}
	planHeader := auditdomain.CheckPlan{Schema: candidate.Header.PlanSchema, Method: candidate.Header.SourceMethod, Phase: candidate.Header.Phase}
	route, ok := routes.selectBinding(planHeader)
	if !ok {
		return reject("unmapped_method")
	}
	decision.WorkflowRole, decision.RouteDigest = route.route.WorkflowRole, route.route.MappingDigest
	if state.RoundOrdinal > audit.Limits.MaxRounds || capacity < 1 {
		code := "round_budget_exhausted"
		if capacity < 1 {
			code = "item_budget_exhausted"
		}
		decision.Disposition, decision.Code = "deferred", code
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.StopCode = code
		mutation.Next.Exhausted = true
		return typedSelectionCheckpoint(mutation, budget)
	}
	if audit.ReservedRunCount >= audit.Limits.MaxSubmittedRuns {
		decision.Disposition, decision.Code = "deferred", "submission_budget_exhausted"
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.StopCode = decision.Code
		mutation.Next.Exhausted = true
		return typedSelectionCheckpoint(mutation, budget)
	}
	if budget.Full() {
		decision.Disposition, decision.Code = "deferred", "round_metadata_or_item_limit"
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.Exhausted = true
		return typedSelectionCheckpoint(mutation, budget)
	}
	var prepared typedCandidateResult
	kept, err := session.WithCandidate(ctx, func(ctx context.Context, service *artifacts.Service, reads auditstore.AdmissionReader) (bool, error) {
		check, err := reads.ReadFindingCheck(ctx, candidate.ReceiptID, candidate.Ordinal)
		if err != nil {
			return false, err
		}
		prepared, err = prepareTypedCandidate(ctx, service, reads, audit, profile, baseline, selection, state, routes, budget, check, candidate.Ordinal)
		return prepared.admitted, err
	})
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		decision.Disposition, decision.Code = "deferred", "evidence_budget_exhausted"
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.StopCode = decision.Code
		mutation.Next.Exhausted = true
		return typedSelectionCheckpoint(mutation, budget)
	}
	if err != nil {
		if code, permanent := typedPermanentCandidateFailure(err); permanent {
			return reject(code)
		}
		return mutation, err
	}
	if !kept {
		if prepared.unschedulable != "" {
			return reject("candidate_metadata_limit")
		}
		decision.Disposition, decision.Code = "deferred", "round_metadata_or_item_limit"
		mutation.Decisions = []auditstore.AdmissionDecision{decision}
		mutation.Next.Exhausted = true
		return typedSelectionCheckpoint(mutation, budget)
	}
	writer := auditArtifactWriter{service: session.Artifacts, projectID: audit.ProjectID}
	digest := auditdomain.DigestBytes(prepared.pageContent)
	artifact, err := writer.write(ctx, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: typedAdmissionArtifactName("prepared-round-page", state, digest)}, artifacts.Payload{MediaType: auditdomain.PreparedRoundPageMediaType, Data: prepared.pageContent})
	if err != nil {
		return mutation, err
	}
	decision.Disposition, decision.Code, decision.ItemID = "admitted", "selected", prepared.item.ItemID
	mutation.Page = &artifact
	mutation.PageContent = prepared.pageContent
	mutation.Retained = prepared.retained
	mutation.Decisions = []auditstore.AdmissionDecision{decision}
	mutation.Next.Cursor = candidate.Cursor
	mutation.Next.Exhausted = page.Exhausted || budget.Full()
	return typedSelectionCheckpoint(mutation, budget)
}

func typedSelectionCheckpoint(mutation auditstore.AdmissionMutation, budget *auditdomain.FindingRoundBudget) (auditstore.AdmissionMutation, error) {
	checkpoint, err := budget.Checkpoint()
	if err != nil {
		return mutation, err
	}
	mutation.Next.Budget, err = auditdomain.EncodeFindingRoundBudgetState(checkpoint)
	return mutation, err
}
