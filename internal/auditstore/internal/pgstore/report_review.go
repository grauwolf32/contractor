package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

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
