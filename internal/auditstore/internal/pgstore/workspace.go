package pgstore

import (
	"context"
	"errors"
	"math"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// GetWorkspace reads the workspace summary of an owner's Audit in one
// statement.
func (s *QueryStore) GetWorkspace(ctx context.Context, ownerID, auditID string) (auditstore.WorkspaceSummary, error) {
	if !auditstore.ValidReviewIdentity(ownerID, 256) || !auditstore.ValidReviewIdentity(auditID, 256) {
		return auditstore.WorkspaceSummary{}, auditstore.ErrInvalid
	}
	var result auditstore.WorkspaceSummary
	err := s.db.QueryRow(ctx, getWorkspaceSQL, ownerID, auditID,
		auditdomain.CoverageStatusesIn(auditdomain.CoverageBucketCompleted), auditdomain.CoverageIssueStatuses(),
		auditdomain.CoverageStatusesIn(auditdomain.CoverageBucketGap), auditdomain.CoverageStatusesIn(auditdomain.CoverageBucketUnchecked),
	).Scan(
		&result.AuditID, &result.AuditRevision, &result.AsOf, &result.RoundID, &result.ExecutionState, &result.OutstandingRuns,
		&result.TotalChecks, &result.CompletedChecks, &result.Issues, &result.Gaps, &result.Unchecked, &result.Findings, &result.UnreviewedFindings, &result.PendingReviews)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.WorkspaceSummary{}, auditstore.ErrNotFound
	}
	return result, err
}

// ListEventsPage reads one page of an owner's Audit events, newest first,
// in one read-only snapshot. It returns the sequence the page is frozen
// through: the caller's continuation pin, or the Audit's last event. Events
// are numbered without gaps (AUD-R223), so that sequence is also the number
// of events in the prefix, and a page reads at most p.Limit events and no
// count.
func (s *QueryStore) ListEventsPage(
	ctx context.Context, p auditstore.EventListParams,
) (uint64, []auditstore.Event, error) {
	if !auditstore.ValidReviewIdentity(p.OwnerID, 256) || !auditstore.ValidReviewIdentity(p.AuditID, 256) ||
		!pagelimit.ValidFetch(p.Limit) ||
		(p.ThroughSequence == nil) != (p.BeforeSequence == nil) ||
		(p.ThroughSequence != nil && (*p.ThroughSequence > math.MaxInt64 ||
			*p.BeforeSequence == 0 || *p.BeforeSequence > *p.ThroughSequence)) {
		return 0, nil, auditstore.ErrInvalid
	}
	var through uint64
	var events []auditstore.Event
	err := persistencepostgres.Within(ctx, s.db, snapshotTransaction, func(tx pgx.Tx) error {
		audit, err := (&store{db: tx}).get(ctx, p.OwnerID, p.AuditID)
		if err != nil {
			return err
		}
		through = audit.EventSequence
		if p.ThroughSequence != nil {
			if *p.ThroughSequence > audit.EventSequence {
				return auditstore.ErrInvalid
			}
			through = *p.ThroughSequence
		}
		var before *int64
		if p.BeforeSequence != nil {
			value := int64(*p.BeforeSequence)
			before = &value
		}
		rows, err := tx.Query(ctx, listEventsBeforeSQL, p.AuditID, int64(through), before, p.Limit)
		if err != nil {
			return err
		}
		defer rows.Close()
		events = make([]auditstore.Event, 0, p.Limit)
		for rows.Next() {
			var event auditstore.Event
			if err := rows.Scan(&event.AuditID, &event.Sequence, &event.Kind, &event.EntityID,
				&event.EntityRevision, &event.Summary, &event.CreatedAt); err != nil {
				return err
			}
			events = append(events, event)
		}
		return rows.Err()
	})
	if err != nil {
		return 0, nil, err
	}
	return through, events, nil
}
