package auditservice

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// WorkspaceSummary is one database statement's owner-scoped snapshot. Work is
// scoped to the current round; findings and reviews span the entire Audit.
type WorkspaceSummary struct {
	AuditID            string    `json:"auditId"`
	AuditRevision      uint64    `json:"auditRevision"`
	AsOf               time.Time `json:"asOf"`
	RoundID            *string   `json:"roundId,omitempty"`
	ExecutionState     string    `json:"executionState"`
	OutstandingRuns    int       `json:"outstandingRuns"`
	TotalChecks        int       `json:"totalChecks"`
	CompletedChecks    int       `json:"completedChecks"`
	Issues             int       `json:"issues"`
	Gaps               int       `json:"gaps"`
	Unchecked          int       `json:"unchecked"`
	Findings           int       `json:"findings"`
	UnreviewedFindings int       `json:"unreviewedFindings"`
	PendingReviews     int       `json:"pendingReviews"`
}

type PageBasis struct {
	AuditRevision uint64    `json:"auditRevision"`
	AsOf          time.Time `json:"asOf"`
	Total         int       `json:"total"`
}

type FindingPage struct {
	Items []Finding
	PageBasis
}
type ReviewPage struct {
	Items []ReviewRequest
	PageBasis
}

func (s *Service) GetWorkspace(ctx context.Context, ownerID, auditID string) (WorkspaceSummary, error) {
	if !validReviewIdentity(ownerID, 256) || !validReviewIdentity(auditID, 256) {
		return WorkspaceSummary{}, auditstore.ErrInvalid
	}
	var result WorkspaceSummary
	err := s.pool.QueryRow(ctx, `
SELECT audit.audit_id, audit.revision, statement_timestamp(), audit.current_round_id,
       audit.state, audit.outstanding_run_count,
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status IN ('satisfied', 'violated', 'traced-complete', 'not-applicable', 'excluded')),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status = 'violated'),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status IN ('inconclusive', 'blocked', 'traced-partial', 'unmapped')),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status = 'not-tested'),
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id),
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id AND current_decision_id IS NULL),
       (SELECT count(*) FROM audit_review_requests WHERE audit_id = audit.audit_id AND state = 'pending')
FROM audits audit WHERE owner_id = $1 AND audit_id = $2`, ownerID, auditID).Scan(
		&result.AuditID, &result.AuditRevision, &result.AsOf, &result.RoundID, &result.ExecutionState, &result.OutstandingRuns,
		&result.TotalChecks, &result.CompletedChecks, &result.Issues, &result.Gaps, &result.Unchecked, &result.Findings, &result.UnreviewedFindings, &result.PendingReviews)
	if errors.Is(err, pgx.ErrNoRows) {
		return WorkspaceSummary{}, auditstore.ErrNotFound
	}
	return result, err
}

// A revision fence includes hydration and the whole-filter count. Audit mutations
// increment its revision atomically, so a changing view is rejected rather than
// returning rows/counts/evidence from different revisions. No transaction is held
// while hydration acquires a pool connection (including single-connection pools).
func (s *Service) checkPageRevision(ctx context.Context, ownerID, auditID string, revision uint64) error {
	audit, err := s.Get(ctx, ownerID, auditID)
	if err != nil {
		return err
	}
	if audit.Revision != revision {
		return auditstore.ErrConflict
	}
	return nil
}

func (s *Service) ListFindingsPage(ctx context.Context, params FindingListParams) (FindingPage, error) {
	var result FindingPage
	if err := validateFindingList(params); err != nil {
		return result, err
	}
	err := s.pool.QueryRow(ctx, `SELECT audit.revision, statement_timestamp(),
 (SELECT count(*) `+findingFilterSQL+`) FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`,
		params.OwnerID, params.AuditID, params.State, params.Verdict, params.Unreviewed, params.Severity).Scan(&result.AuditRevision, &result.AsOf, &result.Total)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	if err != nil {
		return result, err
	}
	if params.AuditRevision != nil && *params.AuditRevision != result.AuditRevision {
		return result, auditstore.ErrConflict
	}
	result.Items, err = s.ListFindings(ctx, params)
	if err != nil {
		return result, err
	}
	return result, s.checkPageRevision(ctx, params.OwnerID, params.AuditID, result.AuditRevision)
}

func (s *Service) ListReviewsPage(ctx context.Context, params ReviewListParams) (ReviewPage, error) {
	var result ReviewPage
	if err := validateReviewList(params); err != nil {
		return result, err
	}
	err := s.pool.QueryRow(ctx, `SELECT audit.revision, statement_timestamp(),
 (SELECT count(*) FROM audit_review_requests request WHERE request.audit_id=audit.audit_id
 AND ($3::text IS NULL OR request.finding_id=$3) AND ($4::text IS NULL OR request.state=$4))
 FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`, params.OwnerID, params.AuditID, params.FindingID, params.State).Scan(&result.AuditRevision, &result.AsOf, &result.Total)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	if err != nil {
		return result, err
	}
	if params.AuditRevision != nil && *params.AuditRevision != result.AuditRevision {
		return result, auditstore.ErrConflict
	}
	result.Items, err = s.ListReviews(ctx, params)
	if err != nil {
		return result, err
	}
	return result, s.checkPageRevision(ctx, params.OwnerID, params.AuditID, result.AuditRevision)
}
