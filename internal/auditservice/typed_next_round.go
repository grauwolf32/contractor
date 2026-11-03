package auditservice

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func (s *Service) prepareTypedNextRound(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot, selection DraftSelection) (auditstore.AcceptRoundParams, *auditstore.StopReason, error) {
	routes, err := NewRoutedCheckContext(snapshot.Audit, profile, baseline)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	total, err := s.reads().CountItems(ctx, snapshot.Audit.AuditID)
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	capacity := roundItemCapacity(snapshot.Audit.Limits.MaxItemsPerRound, snapshot.Audit.Limits.MaxItemsTotal, total)
	progress, err := hookstore.NewClaimedAdmission(s.pool, s.storage).WithStep(ctx, auditstore.AdmissionStepParams{Claim: claim, PreviousRoundID: snapshot.Round.RoundID, Now: s.now()}, func(ctx context.Context, session *hookstore.AdmissionSession) (auditstore.AdmissionMutation, error) {
		state := session.State
		if state.StopCode != "" {
			return auditstore.AdmissionMutation{Next: state}, nil
		}
		if state.Root != nil {
			if state.StagedPages < state.PublishedPages {
				ordinal := state.StagedPages
				return auditstore.AdmissionMutation{Next: state, StagePageOrdinal: &ordinal}, nil
			}
			return auditstore.AdmissionMutation{Next: state}, nil
		}
		if state.Exhausted {
			if state.SelectedCount == 0 {
				page, err := session.Reads.ReadCandidates(ctx, state.Cursor, 1)
				if err != nil {
					return auditstore.AdmissionMutation{}, err
				}
				if page.LateSources {
					return auditstore.AdmissionMutation{Next: state, Restart: true}, nil
				}
				return auditstore.AdmissionMutation{Next: state}, nil
			}
			return sealTypedRoot(ctx, session, snapshot.Audit)
		}
		budget, err := typedRoundBudget(state, capacity, snapshot.Audit, profile, baseline)
		if err != nil {
			return auditstore.AdmissionMutation{}, err
		}
		return s.selectTypedCandidate(ctx, session, snapshot.Audit, profile, baseline, selection, routes, budget, capacity)
	})
	if errors.Is(err, auditstore.ErrAdmissionDeadline) {
		return auditstore.AcceptRoundParams{}, typedAdmissionStopReason("next_round_validation_deadline"), nil
	}
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		return auditstore.AcceptRoundParams{}, typedAdmissionStopReason("evidence_budget_exhausted"), nil
	}
	if err != nil {
		return auditstore.AcceptRoundParams{}, nil, err
	}
	if progress.StopCode != "" {
		return auditstore.AcceptRoundParams{}, typedAdmissionStopReason(progress.StopCode), nil
	}
	if progress.Root != nil && progress.StagedPages == progress.PublishedPages && progress.StagedCount == progress.SelectedCount {
		return auditstore.AcceptRoundParams{Claim: claim, ExpectedAuditRevision: snapshot.Audit.Revision, PreviousRoundID: snapshot.Round.RoundID, RoundID: progress.RoundID, RoundOrdinal: progress.RoundOrdinal, Manifest: progress.Root.Clone(), PreparedRoot: progress.Root, Generation: progress.Generation}, nil, nil
	}
	if progress.Exhausted && progress.SelectedCount == 0 && !progress.Changed {
		return auditstore.AcceptRoundParams{}, typedAdmissionStopReason("round_complete"), nil
	}
	return auditstore.AcceptRoundParams{}, nil, ErrNextRoundAdmissionPending
}

func typedRoundBudget(state auditstore.AdmissionProgress, capacity int, audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot) (*auditdomain.FindingRoundBudget, error) {
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	source := contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("finding-check-source", audit.AuditID, placeholderAdmissionDigest()), Revision: &provisionalRevision}
	reserve, err := typedRootMetadataReserve(state)
	if err != nil {
		return nil, err
	}
	shape := auditdomain.FindingRoundShape{Inventory: nextRoundInventoryOptions(state.RoundOrdinal, profile, baseline, auditdomain.ApprovalNone, source), TaskNamespace: namespace, TaskRevision: provisionalRevision, ExecutionInputs: []auditdomain.ExactInput{}, Paged: true, PreparedRootMetadataBytes: int64(reserve)}
	if string(state.Budget) == "{}" && state.SelectedCount == 0 && state.PublishedPages == 0 {
		return auditdomain.NewFindingRoundBudget(capacity, shape)
	}
	checkpoint, err := auditdomain.DecodeFindingRoundBudgetState(state.Budget)
	if err != nil {
		return nil, err
	}
	if checkpoint.Checks != state.SelectedCount {
		return nil, inconsistentRound("typed admission budget differs from selected membership", nil)
	}
	return auditdomain.ResumeFindingRoundBudget(capacity, shape, checkpoint)
}

func typedRootMetadataReserve(state auditstore.AdmissionProgress) (int, error) {
	root := auditdomain.PreparedRoundRoot{Schema: auditdomain.PreparedRoundRootSchema, AuditID: state.AuditID, RoundID: state.RoundID, Generation: state.Generation, ProfileDigest: state.ProfileDigest, BaselineDigest: state.BaselineDigest, PageCount: auditdomain.MaximumItems, ItemCount: auditdomain.MaximumItems, PagesDigest: placeholderAdmissionDigest()}
	data, err := auditdomain.EncodePreparedRoundRoot(root)
	return len(data), err
}

func placeholderAdmissionDigest() string { return auditdomain.DigestBytes(nil) }

func typedAdmissionArtifactName(kind string, state auditstore.AdmissionProgress, digest string) string {
	return auditdomain.DeterministicID(kind, state.AuditID, state.Generation, digest)
}

func typedAdmissionStopReason(code string) *auditstore.StopReason {
	message := map[string]string{
		"round_complete":                 "The immutable Audit Round has no further admissible typed finding checks; retained decisions describe rejected or blocked candidates.",
		"round_budget_exhausted":         "Typed finding checks remain after the configured Round budget was exhausted.",
		"item_budget_exhausted":          "Typed finding checks remain after the cumulative item budget was exhausted.",
		"submission_budget_exhausted":    "Typed finding checks remain after the cumulative child Run submission budget was exhausted.",
		"evidence_budget_exhausted":      "The complete retained typed-check context exceeds the remaining evidence budget.",
		"next_round_validation_deadline": "The durable next-Round validation deadline expired before complete immutable membership was accepted.",
	}[code]
	if message == "" {
		message = fmt.Sprintf("Typed next-Round admission stopped (%s).", code)
	}
	return &auditstore.StopReason{Code: code, Message: message}
}

func typedPermanentCandidateFailure(err error) (string, bool) {
	if errors.Is(err, artifacts.ErrHistoryMaintenancePending) || errors.Is(err, blob.ErrIO) {
		return "", false
	}
	if errors.Is(err, auditstore.ErrAdmissionSliceFull) {
		return "candidate_work_limit", true
	}
	if errors.Is(err, artifacts.ErrHistoryProtected) {
		return "artifact_history_protected", true
	}
	if errors.Is(err, auditstore.ErrDependencyClosureLimit) {
		return "dependency_closure_limit", true
	}
	if errors.Is(err, auditstore.ErrDependencySourceAmbiguous) {
		return "dependency_source_ambiguous", true
	}
	for _, permanent := range []error{contracts.ErrValidation, artifacts.ErrInvalidScope, artifacts.ErrInvalidName, artifacts.ErrExactRevisionRequired, artifacts.ErrArtifactConflict, artifacts.ErrArtifactFrozen, artifacts.ErrReservedNamespace, artifacts.ErrRunRevisionLimit, artifacts.ErrInvalidMediaType, blob.ErrTooLarge, artifacts.ErrVersionedWriteTarget} {
		if errors.Is(err, permanent) {
			return "dependency_or_source_invalid", true
		}
	}
	var invalid *auditdomain.ValidationError
	if errors.As(err, &invalid) || errors.Is(err, ErrInvalid) || errors.Is(err, auditstore.ErrInvalid) || errors.Is(err, auditstore.ErrNotFound) || errors.Is(err, contracts.ErrExactArtifactMismatch) || errors.Is(err, blob.ErrIntegrity) || artifactpostgres.IsExactRevisionMissing(err) {
		return "dependency_or_source_invalid", true
	}
	return "", false
}
