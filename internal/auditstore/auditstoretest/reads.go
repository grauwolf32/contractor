package auditstoretest

import (
	"context"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

// EventsAfter reads up to limit events of the owner's Audit numbered above
// after, oldest first. It pages through the owner-fenced event pages that the
// public API serves, newest first, back to after; events are numbered without
// gaps (AUD-R223).
func (s *Store) EventsAfter(
	ctx context.Context, ownerID, auditID string, after uint64, limit int,
) ([]auditstore.Event, error) {
	params := auditstore.EventListParams{OwnerID: ownerID, AuditID: auditID, Limit: pagelimit.MaxRows}
	newestFirst := make([]auditstore.Event, 0)
	for {
		through, page, err := s.ListEventsPage(ctx, params)
		if err != nil {
			return nil, err
		}
		done := len(page) < params.Limit
		for _, event := range page {
			if event.Sequence <= after {
				done = true
				break
			}
			newestFirst = append(newestFirst, event)
		}
		if done {
			break
		}
		before := page[len(page)-1].Sequence
		params.ThroughSequence, params.BeforeSequence = &through, &before
	}
	slices.Reverse(newestFirst)
	return newestFirst[:min(limit, len(newestFirst))], nil
}

// listExecutionsSQL reads every execution of Audit $1 in creation order.
var listExecutionsSQL = `
SELECT ` + pgstore.ExecutionColumns("execution") + `
  FROM audit_executions AS execution WHERE execution.audit_id = $1
 ORDER BY execution.created_at, execution.execution_id`

// ListExecutions reads every execution of an Audit in creation order, so a
// test can inspect all of them. Production reads executions only through
// bounded windows, roles and selections; this read decodes and validates
// each execution as they do.
func (s *Store) ListExecutions(ctx context.Context, auditID string) ([]auditstore.Execution, error) {
	rows, err := s.db.Query(ctx, listExecutionsSQL, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit executions: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Execution, 0)
	for rows.Next() {
		execution, err := pgstore.ScanExecution(rows)
		if err != nil {
			return nil, fmt.Errorf("scan Audit execution: %w", err)
		}
		result = append(result, execution)
	}
	return result, rows.Err()
}
