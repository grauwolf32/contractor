package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// ProposeReport freezes exact report links and opens the existing review
// ledger instead of publishing them. It is replay-safe across Controller
// crashes and leaves the Audit without a runnable claim while a person reads
// the immutable candidate.
func (s *ClaimStore) ProposeReport(
	ctx context.Context, params auditstore.ProposeReportParams,
) (auditstore.Audit, bool, error) {
	if err := validateCommitReport(auditstore.CommitReportParams(params)); err != nil {
		return auditstore.Audit{}, false, err
	}
	machine, err := json.Marshal(params.Machine)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	summary, err := json.Marshal(params.Summary)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	reviewID := "review-report-" + strings.TrimPrefix(params.RequestDigest, "sha256:")
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, proposeReportSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.RoundID, params.ExpectedRoundRevision,
		reviewID, params.RequestDigest, machine, summary,
	))
	if err == nil {
		return audit, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) && persistencepostgres.Classify(err) != persistencepostgres.ClassDuplicate {
		return auditstore.Audit{}, false, fmt.Errorf("propose Audit report: %w", err)
	}
	candidate, replayErr := s.getReportCandidate(ctx, params.Claim.AuditID)
	if replayErr == nil {
		if candidate.SubjectRevision != params.ExpectedAuditRevision || candidate.SubjectDigest != params.RequestDigest {
			return auditstore.Audit{}, false, auditstore.ErrConflict
		}
		existing, getErr := s.getAuditTrusted(ctx, params.Claim.AuditID)
		return existing, false, getErr
	} else if !errors.Is(replayErr, auditstore.ErrNotFound) {
		return auditstore.Audit{}, false, replayErr
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Audit{}, false, liveErr
	} else if !live {
		return auditstore.Audit{}, false, auditstore.ErrClaimLost
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
}

func (s *store) getReportCandidate(
	ctx context.Context, auditID string,
) (auditstore.ReportCandidate, error) {
	if err := validateID("auditID", auditID); err != nil {
		return auditstore.ReportCandidate{}, err
	}
	var result auditstore.ReportCandidate
	var revision int64
	var machine, summary []byte
	err := s.db.QueryRow(ctx, `
SELECT request_id, report_round_id, subject_revision, subject_digest,
       report_machine_link, report_summary_link, created_at
  FROM audit_review_requests
 WHERE audit_id = $1 AND subject_kind = 'audit-report'`, auditID).Scan(
		&result.RequestID, &result.RoundID, &revision, &result.SubjectDigest,
		&machine, &summary, &result.CreatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ReportCandidate{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ReportCandidate{}, fmt.Errorf("read Audit report candidate: %w", err)
	}
	if revision < 1 || validateDigest("report candidate digest", result.SubjectDigest) != nil ||
		json.Unmarshal(machine, &result.Machine) != nil ||
		json.Unmarshal(summary, &result.Summary) != nil ||
		validateReportCandidateLinks(result.Machine, result.Summary) != nil {
		return auditstore.ReportCandidate{}, errors.New("stored Audit report candidate is invalid")
	}
	result.SubjectRevision = uint64(revision)
	return result, nil
}

// ExpireReportReview deterministically closes an Audit whose immutable report
// candidate was not accepted within its bounded review window. The Controller
// calls it only under a live claim; the SQL rechecks database time so process
// clock skew cannot expire human authority early.
func (s *ClaimStore) ExpireReportReview(
	ctx context.Context, claim auditstore.ControllerClaim, expectedAuditRevision uint64,
) (bool, error) {
	if err := validateClaimIdentity(claim); err != nil || expectedAuditRevision == 0 {
		if err != nil {
			return false, err
		}
		return false, auditstore.ErrInvalid
	}
	var changed bool
	err := s.db.QueryRow(ctx, expireReportReviewSQL, claim.AuditID, claim.HolderID,
		claim.Epoch, int64(expectedAuditRevision)).Scan(&changed)
	if err != nil {
		return false, fmt.Errorf("expire Audit report review: %w", err)
	}
	if changed {
		return true, nil
	}
	live, err := s.claimLive(ctx, claim)
	if err != nil {
		return false, err
	}
	if !live {
		return false, auditstore.ErrClaimLost
	}
	return false, nil
}

// AcceptReportCandidate publishes the exact links of the frozen report
// candidate an owner accepted and completes the Audit. It records the same
// audit.report_committed event as CommitReport, carrying the candidate's
// report request digest.
func (s *ReviewStore) AcceptReportCandidate(
	ctx context.Context, params auditstore.ReportDecisionParams,
) (auditstore.Audit, error) {
	if err := validateReportDecision(params); err != nil {
		return auditstore.Audit{}, err
	}
	candidate, err := s.getReportCandidate(ctx, params.AuditID)
	if err != nil {
		return auditstore.Audit{}, err
	}
	if candidate.RequestID != params.RequestID || candidate.SubjectRevision != params.SubjectRevision ||
		candidate.SubjectDigest != params.SubjectDigest {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, acceptReportCandidateSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.AuditID, int64(params.ExpectedAuditRevision), params.RequestID,
		params.SubjectDigest, encodeReportLinks(candidate.Machine, candidate.Summary),
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	if err != nil {
		return auditstore.Audit{}, fmt.Errorf("accept Audit report candidate: %w", err)
	}
	return audit, nil
}

// RejectReportCandidate fails the Audit whose frozen report candidate an owner
// rejected, recording the audit.state_changed event of a terminal transition.
func (s *ReviewStore) RejectReportCandidate(
	ctx context.Context, params auditstore.ReportDecisionParams,
) (auditstore.Audit, error) {
	if err := validateReportDecision(params); err != nil {
		return auditstore.Audit{}, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, rejectReportCandidateSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.AuditID, int64(params.ExpectedAuditRevision), params.RequestID,
		int64(params.SubjectRevision), params.SubjectDigest,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		if _, candidateErr := s.getReportCandidate(ctx, params.AuditID); candidateErr != nil {
			return auditstore.Audit{}, candidateErr
		}
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	if err != nil {
		return auditstore.Audit{}, fmt.Errorf("reject Audit report candidate: %w", err)
	}
	return audit, nil
}

// validateReportCandidateLinks requires current exact report descriptors before
// reading or approving a retained candidate.
func validateReportCandidateLinks(machine, summary auditstore.ArtifactLink) error {
	for _, pair := range []struct {
		link  auditstore.ArtifactLink
		key   string
		media string
	}{
		{machine, auditstore.ReportMachineLogicalKey, "application/json"},
		{summary, auditstore.ReportSummaryLogicalKey, "text/markdown"},
	} {
		if pair.link.LogicalKey != pair.key || pair.link.Artifact.MediaType != pair.media ||
			validateExactArtifact("report candidate", pair.link.Artifact, false) != nil ||
			len(pair.link.SourceProvenance) == 0 || !json.Valid(pair.link.SourceProvenance) {
			return auditstore.ErrInvalid
		}
	}
	return nil
}
