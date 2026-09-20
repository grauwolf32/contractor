package evalstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type InventoryEntry struct {
	Execution *evaldomain.ExecutionRef `json:"execution"`
	Parent    *evaldomain.ExecutionRef `json:"parent"`
	Role      *string                  `json:"role"`
	Round     *int                     `json:"round"`
	State     string                   `json:"state"`
	Available bool                     `json:"available"`
	IntentID  string                   `json:"intentId,omitempty"`
	ProjectID *string                  `json:"projectId,omitempty"`
}
type Inventory struct {
	Revision int64
	Complete bool
	Entries  []InventoryEntry
	Gaps     []string
}

func ordinaryState(state string) string {
	switch state {
	case "succeeded", "completed":
		return "succeeded"
	case "failed":
		return "failed"
	case "cancelled":
		return "cancelled"
	case "initializing", "draft":
		return "accepted"
	case "running", "active", "paused", "waiting_review", "finalizing", "cancelling":
		return "running"
	}
	return "unknown"
}

// Inventory reads every authoritative association, independent of Audit item
// pagination or labels. The format bounds the collection to 1024 executions.
// An unresolved dispatch intent keeps its own identity without inventing a Run.
func (s *Store) Inventory(ctx context.Context, owner, id, member string) (Inventory, error) {
	out := Inventory{Entries: []InventoryEntry{}, Gaps: []string{}}
	var kind string
	var ref, state, projectID *string
	var available, closed bool
	err := s.db.QueryRow(ctx, `
SELECT p.revision, m.execution_kind, sub.execution_id, COALESCE(r.state, a.state),
    r.run_id IS NOT NULL OR a.audit_id IS NOT NULL,
    COALESCE(a.dispatch_state = 'closed', TRUE), COALESCE(r.project_id, a.project_id)
FROM eval_members m
JOIN eval_experiments e USING (experiment_id)
JOIN eval_member_projections p USING (experiment_id, member_id)
LEFT JOIN eval_submissions sub USING (experiment_id, member_id)
LEFT JOIN workflow_runs r ON m.execution_kind = 'run'
    AND r.run_id = sub.execution_id AND r.owner_id = e.owner_id
LEFT JOIN audits a ON m.execution_kind = 'audit'
    AND a.audit_id = sub.execution_id AND a.owner_id = e.owner_id
WHERE e.owner_id = $1 AND e.experiment_id = $2 AND m.member_id = $3
`, owner, id, member).Scan(&out.Revision, &kind, &ref, &state, &available, &closed, &projectID)
	if err != nil {
		return out, normalize(err)
	}
	if ref == nil {
		out.Gaps = append(out.Gaps, "Member execution is not confirmed.")
		return out, nil
	}
	parent := evaldomain.ExecutionRef{Kind: kind, ID: *ref}
	entry := InventoryEntry{Execution: &parent, Available: available, State: "unknown", ProjectID: projectID}
	if state != nil {
		entry.State = ordinaryState(*state)
	}
	out.Entries = append(out.Entries, entry)
	out.Complete = available && closed && (entry.State == "succeeded" || entry.State == "failed" || entry.State == "cancelled")
	if !available {
		out.Gaps = append(out.Gaps, "Parent execution was deleted or is unavailable.")
	} else if !out.Complete {
		out.Gaps = append(out.Gaps, "Parent execution inventory is still open.")
	}
	if kind != "audit" || !available {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `
SELECT x.execution_id,x.run_id,x.role,round.ordinal,x.state,x.terminal_outcome,r.state,r.run_id IS NOT NULL,r.project_id
FROM audit_executions x
JOIN audits a USING(audit_id)
LEFT JOIN audit_rounds round ON round.audit_id = x.audit_id
AND round.round_id = x.round_id
LEFT JOIN workflow_runs r ON r.run_id = x.run_id
AND r.owner_id = a.owner_id
WHERE x.audit_id = $1
    AND a.owner_id = $2
ORDER BY x.created_at,x.execution_id
LIMIT $3
`, *ref, owner, evaldomain.MaxInventoryExecutions+1)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var child InventoryEntry
		var runID, terminal, runState *string
		var phase string
		if err = rows.Scan(&child.IntentID, &runID, &child.Role, &child.Round, &phase, &terminal, &runState, &child.Available, &child.ProjectID); err != nil {
			return out, err
		}
		child.Parent = &parent
		child.State = "unknown"
		if runID != nil {
			child.Execution = &evaldomain.ExecutionRef{Kind: "run", ID: *runID}
		}
		if runState != nil {
			child.State = ordinaryState(*runState)
		}
		if phase != "collected" {
			out.Gaps = append(out.Gaps, "Child dispatch or collection is unresolved: "+child.IntentID)
			out.Complete = false
		}
		if runID != nil && !child.Available {
			out.Gaps = append(out.Gaps, "Owned child Run was deleted: "+child.IntentID)
			out.Complete = false
		}
		if runID == nil && (terminal == nil || *terminal != "submission-failed") {
			out.Gaps = append(out.Gaps, "Child Run identity is unresolved: "+child.IntentID)
			out.Complete = false
		}
		out.Entries = append(out.Entries, child)
		if len(out.Entries) > evaldomain.MaxInventoryExecutions {
			out.Entries = out.Entries[:evaldomain.MaxInventoryExecutions]
			out.Complete = false
			out.Gaps = append(out.Gaps, "Execution inventory exceeds the managed collection bound.")
			break
		}
	}
	out.Gaps = evaldomain.BoundedGaps(out.Gaps)
	return out, rows.Err()
}
