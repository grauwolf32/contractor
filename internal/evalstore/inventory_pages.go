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
	afterChild, err := inventoryChildAfter(after)
	if err != nil {
		return out, err
	}
	var kind string
	var parentID, state *string
	var available, closed bool
	var unresolved, deleted int
	err = s.db.QueryRow(ctx, inventoryPageMemberSQL, owner, id, member).Scan(&out.Revision, &kind, &parentID, &state, &available, &closed, &unresolved, &deleted)
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
	rows, err := s.db.Query(ctx, inventoryPageAuditExecutionsSQL, owner, *parentID, afterChild, limit-len(out.Items)+1)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var child InventoryEntry
		var run, state, outcome *string
		if err = rows.Scan(&child.IntentID, &run, &child.Workflow, &child.Iteration, &state, &child.Available, &outcome); err != nil {
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

// inventoryChildAfter returns the execution ID after which a page's child
// entries continue. A page key is "0" for the parent entry and "1" followed
// by the execution ID for a child entry, so the parent sorts first; an empty
// key starts at the parent. Before the first child the result is empty,
// which every execution ID follows. Any other key is invalid.
func inventoryChildAfter(after string) (string, error) {
	switch {
	case after == "" || after == "0":
		return "", nil
	case len(after) > 1 && after[0] == '1':
		return after[1:], nil
	}
	return "", evaldomain.Failure("eval_invalid")
}
