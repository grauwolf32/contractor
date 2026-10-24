package auditstoretest

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
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

// Items reads every item of the owner's Audit in Round and item order. It
// pages through the owner-fenced item pages that the public API serves.
func (s *Store) Items(ctx context.Context, ownerID, auditID string) ([]auditstore.Item, error) {
	params := auditstore.ListItemsParams{OwnerID: ownerID, AuditID: auditID, Limit: pagelimit.MaxRows}
	items := make([]auditstore.Item, 0)
	for {
		page, err := s.ListItemsPage(ctx, params)
		if err != nil {
			return nil, err
		}
		items = append(items, page...)
		if len(page) < params.Limit {
			return items, nil
		}
		last := page[len(page)-1]
		round, err := s.GetRound(ctx, auditID, last.RoundID)
		if err != nil {
			return nil, err
		}
		params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &round.Ordinal, &last.Ordinal, last.ItemID
	}
}

// listExecutionsSQL reads every execution of Audit $1 in creation order.
const listExecutionsSQL = `
SELECT execution_id, audit_id, round_id, role, workflow_role, role_attempt,
       manifest_ref, manifest_digest, submission_key, request_digest, run_id,
       state, terminal_outcome, terminal_run_generation, terminal_run_sequence,
       terminal_observed_at, run_provenance, run_deleted_at, created_at, updated_at,
       preparation_snapshot, preparation_outputs
  FROM audit_executions WHERE audit_id = $1
 ORDER BY created_at, execution_id`

// ListExecutions reads every execution of an Audit in creation order, so a
// test can inspect all of them. Production reads executions only through
// bounded windows, roles and selections; the store validates what it reads,
// and this read only decodes it.
func (s *Store) ListExecutions(ctx context.Context, auditID string) ([]auditstore.Execution, error) {
	rows, err := s.db.Query(ctx, listExecutionsSQL, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit executions: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.Execution, 0)
	for rows.Next() {
		var execution auditstore.Execution
		var manifest, provenance, preparation, outputs []byte
		var outcome *auditstore.TerminalOutcome
		var sequence *int64
		if err := rows.Scan(
			&execution.ExecutionID, &execution.AuditID, &execution.RoundID, &execution.Role,
			&execution.WorkflowRole, &execution.RoleAttempt, &manifest, &execution.Manifest.Digest,
			&execution.SubmissionKey, &execution.RequestDigest, &execution.RunID, &execution.State,
			&outcome, &execution.TerminalRunGeneration, &sequence, &execution.TerminalObservedAt,
			&provenance, &execution.RunDeletedAt, &execution.CreatedAt, &execution.UpdatedAt,
			&preparation, &outputs,
		); err != nil {
			return nil, fmt.Errorf("scan Audit execution: %w", err)
		}
		execution.TerminalOutcome = outcome
		if sequence != nil {
			value := uint64(*sequence)
			execution.TerminalRunSequence = &value
		}
		if err := json.Unmarshal(manifest, &execution.Manifest.Ref); err != nil {
			return nil, fmt.Errorf("decode Audit execution manifest: %w", err)
		}
		if provenance != nil {
			execution.RunProvenance = new(auditstore.RunProvenance)
			if err := json.Unmarshal(provenance, execution.RunProvenance); err != nil {
				return nil, fmt.Errorf("decode Audit execution Run provenance: %w", err)
			}
		}
		if preparation != nil {
			execution.Preparation = new(auditstore.PreparationSnapshot)
			if err := json.Unmarshal(preparation, execution.Preparation); err != nil {
				return nil, fmt.Errorf("decode Audit execution preparation: %w", err)
			}
		}
		if err := json.Unmarshal(outputs, &execution.PreparationOutputs); err != nil {
			return nil, fmt.Errorf("decode Audit execution preparation outputs: %w", err)
		}
		result = append(result, execution)
	}
	return result, rows.Err()
}
