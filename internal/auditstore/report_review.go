package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// ProposeReport freezes exact report links and opens the existing review
// ledger instead of publishing them. It is replay-safe across Controller
// crashes and leaves the Audit without a runnable claim while a person reads
// the immutable candidate.
func (s *PostgresStore) ProposeReport(
	ctx context.Context, params ProposeReportParams,
) (Audit, bool, error) {
	if err := validateCommitReport(CommitReportParams(params)); err != nil {
		return Audit{}, false, err
	}
	machine, err := json.Marshal(params.Machine)
	if err != nil {
		return Audit{}, false, err
	}
	summary, err := json.Marshal(params.Summary)
	if err != nil {
		return Audit{}, false, err
	}
	reviewID := "review-report-" + strings.TrimPrefix(params.RequestDigest, "sha256:")
	audit, err := scanAudit(s.db.QueryRow(ctx, proposeReportSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.RoundID, params.ExpectedRoundRevision,
		reviewID, params.RequestDigest, machine, summary,
	))
	if err == nil {
		return audit, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) && persistencepostgres.SQLState(err) != persistencepostgres.SQLStateUniqueViolation {
		return Audit{}, false, fmt.Errorf("propose Audit report: %w", err)
	}
	candidate, replayErr := s.GetReportCandidate(ctx, params.Claim.AuditID)
	if replayErr == nil {
		if candidate.SubjectRevision != params.ExpectedAuditRevision || candidate.SubjectDigest != params.RequestDigest {
			return Audit{}, false, ErrConflict
		}
		existing, getErr := s.getAuditTrusted(ctx, params.Claim.AuditID)
		return existing, false, getErr
	} else if !errors.Is(replayErr, ErrNotFound) {
		return Audit{}, false, replayErr
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Audit{}, false, liveErr
	} else if !live {
		return Audit{}, false, ErrClaimLost
	}
	return Audit{}, false, ErrPrecondition
}

func (s *PostgresStore) GetReportCandidate(
	ctx context.Context, auditID string,
) (ReportCandidate, error) {
	if err := validateID("auditID", auditID); err != nil {
		return ReportCandidate{}, err
	}
	var result ReportCandidate
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
		return ReportCandidate{}, ErrNotFound
	}
	if err != nil {
		return ReportCandidate{}, fmt.Errorf("read Audit report candidate: %w", err)
	}
	if revision < 1 || validateDigest("report candidate digest", result.SubjectDigest) != nil ||
		json.Unmarshal(machine, &result.Machine) != nil ||
		json.Unmarshal(summary, &result.Summary) != nil ||
		ValidateReportCandidateLinks(result.Machine, result.Summary) != nil {
		return ReportCandidate{}, errors.New("stored Audit report candidate is invalid")
	}
	result.SubjectRevision = uint64(revision)
	return result, nil
}

// ExpireReportReview deterministically closes an Audit whose immutable report
// candidate was not accepted within its bounded review window. The Controller
// calls it only under a live claim; the SQL rechecks database time so process
// clock skew cannot expire human authority early.
func (s *PostgresStore) ExpireReportReview(
	ctx context.Context, claim ControllerClaim, expectedAuditRevision uint64,
) (bool, error) {
	if err := validateClaimIdentity(claim); err != nil || expectedAuditRevision == 0 {
		if err != nil {
			return false, err
		}
		return false, ErrInvalid
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
		return false, ErrClaimLost
	}
	return false, nil
}

// AcceptReportCandidate publishes the exact links of the frozen report
// candidate an owner accepted and completes the Audit. It records the same
// audit.report_committed event as CommitReport, carrying the candidate's
// report request digest.
func (s *PostgresStore) AcceptReportCandidate(
	ctx context.Context, params ReportDecisionParams,
) (Audit, error) {
	if err := validateReportDecision(params); err != nil {
		return Audit{}, err
	}
	candidate, err := s.GetReportCandidate(ctx, params.AuditID)
	if err != nil {
		return Audit{}, err
	}
	if candidate.RequestID != params.RequestID || candidate.SubjectRevision != params.SubjectRevision ||
		candidate.SubjectDigest != params.SubjectDigest {
		return Audit{}, ErrPrecondition
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, acceptReportCandidateSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.AuditID, int64(params.ExpectedAuditRevision), params.RequestID,
		params.SubjectDigest, encodeReportLinks(candidate.Machine, candidate.Summary),
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, ErrPrecondition
	}
	if err != nil {
		return Audit{}, fmt.Errorf("accept Audit report candidate: %w", err)
	}
	return audit, nil
}

// RejectReportCandidate fails the Audit whose frozen report candidate an owner
// rejected, recording the audit.state_changed event of a terminal transition.
func (s *PostgresStore) RejectReportCandidate(
	ctx context.Context, params ReportDecisionParams,
) (Audit, error) {
	if err := validateReportDecision(params); err != nil {
		return Audit{}, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, rejectReportCandidateSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.AuditID, int64(params.ExpectedAuditRevision), params.RequestID,
		int64(params.SubjectRevision), params.SubjectDigest,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		if _, candidateErr := s.GetReportCandidate(ctx, params.AuditID); candidateErr != nil {
			return Audit{}, candidateErr
		}
		return Audit{}, ErrPrecondition
	}
	if err != nil {
		return Audit{}, fmt.Errorf("reject Audit report candidate: %w", err)
	}
	return audit, nil
}

// ValidateReportCandidateLinks requires current exact report descriptors before
// reading or approving a retained candidate.
func ValidateReportCandidateLinks(machine, summary ArtifactLink) error {
	for _, pair := range []struct {
		link  ArtifactLink
		key   string
		media string
	}{
		{machine, ReportMachineLogicalKey, "application/json"},
		{summary, ReportSummaryLogicalKey, "text/markdown"},
	} {
		if pair.link.LogicalKey != pair.key || pair.link.Artifact.MediaType != pair.media ||
			validateExactArtifact("report candidate", pair.link.Artifact, false) != nil ||
			len(pair.link.SourceProvenance) == 0 || !json.Valid(pair.link.SourceProvenance) {
			return ErrInvalid
		}
	}
	return nil
}
