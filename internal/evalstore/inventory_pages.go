package evalstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type InventoryPage struct {
	Revision int64
	Complete bool
	Items    []InventoryEntry
	Gaps     []string
	HasMore  bool
	LastKey  string
}

func (s *Store) InventoryPage(ctx context.Context, owner, id, member, after string, limit int, revision *int64) (InventoryPage, error) {
	out := InventoryPage{Items: []InventoryEntry{}, Gaps: []string{}}
	var kind string
	var parentID, state *string
	var available, closed bool
	var unresolved, deleted int
	err := s.db.QueryRow(ctx, `
SELECT p.revision, m.execution_kind, sub.execution_id, COALESCE(r.state, a.state),
    r.run_id IS NOT NULL OR a.audit_id IS NOT NULL,
    COALESCE(a.dispatch_state = 'closed', TRUE),
    (SELECT count(*) FROM audit_executions x
        WHERE m.execution_kind = 'audit' AND x.audit_id = sub.execution_id
            AND (x.state <> 'collected'
                OR (x.run_id IS NULL AND x.terminal_outcome IS DISTINCT FROM 'submission-failed'))),
    (SELECT count(*) FROM audit_executions x
        WHERE m.execution_kind = 'audit' AND x.audit_id = sub.execution_id
            AND x.run_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM workflow_runs child
                WHERE child.run_id = x.run_id AND child.owner_id = e.owner_id))
FROM eval_members m
JOIN eval_experiments e USING (experiment_id)
JOIN eval_member_projections p USING (experiment_id, member_id)
LEFT JOIN eval_submissions sub USING (experiment_id, member_id)
LEFT JOIN workflow_runs r ON m.execution_kind = 'run'
    AND r.run_id = sub.execution_id AND r.owner_id = e.owner_id
LEFT JOIN audits a ON m.execution_kind = 'audit'
    AND a.audit_id = sub.execution_id AND a.owner_id = e.owner_id
WHERE e.owner_id = $1 AND e.experiment_id = $2 AND m.member_id = $3
`, owner, id, member).Scan(&out.Revision, &kind, &parentID, &state, &available, &closed, &unresolved, &deleted)
	if err != nil {
		return out, normalize(err)
	}
	if revision != nil && *revision != out.Revision {
		return out, evaldomain.Failure("eval_view_changed")
	}
	if parentID == nil {
		out.Gaps = append(out.Gaps, "Member execution is not confirmed.")
		return out, nil
	}
	parent := evaldomain.ExecutionRef{Kind: kind, ID: *parentID}
	parentState := "unknown"
	if state != nil {
		parentState = ordinaryState(*state)
	}
	out.Complete = available && closed && (parentState == "succeeded" || parentState == "failed" || parentState == "cancelled") && unresolved == 0 && deleted == 0
	if !available {
		out.Gaps = append(out.Gaps, "Parent execution was deleted or is unavailable.")
	}
	if !closed || state == nil || parentState == "running" || parentState == "accepted" {
		out.Gaps = append(out.Gaps, "Parent execution inventory is still open.")
	}
	if unresolved > 0 {
		out.Gaps = append(out.Gaps, "Child dispatch or collection remains unresolved.")
	}
	if deleted > 0 {
		out.Gaps = append(out.Gaps, "Owned child executions are unavailable.")
	}
	if after == "" {
		out.Items = append(out.Items, InventoryEntry{Execution: &parent, State: parentState, Available: available})
		out.LastKey = "0"
	}
	if kind != "audit" {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `
SELECT x.execution_id,x.run_id,x.role,round.ordinal,r.state,r.run_id IS NOT NULL,x.terminal_outcome
FROM audit_executions x
JOIN audits a USING(audit_id)
LEFT JOIN audit_rounds round ON round.audit_id = x.audit_id
AND round.round_id = x.round_id
LEFT JOIN workflow_runs r ON r.run_id = x.run_id
AND r.owner_id = a.owner_id
WHERE a.owner_id = $1
    AND x.audit_id = $2
    AND '1' || x.execution_id > $3
ORDER BY x.execution_id
LIMIT $4
`, owner, *parentID, after, limit-len(out.Items)+1)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var child InventoryEntry
		var run, state, outcome *string
		if err = rows.Scan(&child.IntentID, &run, &child.Role, &child.Round, &state, &child.Available, &outcome); err != nil {
			return out, err
		}
		if len(out.Items) == limit {
			out.HasMore = true
			break
		}
		child.Parent = &parent
		child.State = "unknown"
		if run != nil {
			child.Execution = &evaldomain.ExecutionRef{Kind: "run", ID: *run}
		}
		if state != nil {
			child.State = ordinaryState(*state)
		}
		if run == nil && outcome != nil && *outcome == "submission-failed" {
			child.State = "not_submitted"
		}
		out.Items = append(out.Items, child)
		out.LastKey = "1" + child.IntentID
	}
	return out, rows.Err()
}
