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
	err := s.db.QueryRow(ctx, inventoryMemberSQL, owner, id, member).Scan(&out.Revision, &kind, &ref, &state, &available, &closed, &projectID)
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
	rows, err := s.db.Query(ctx, inventoryAuditExecutionsSQL, *ref, owner, evaldomain.MaxInventoryExecutions+1)
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
