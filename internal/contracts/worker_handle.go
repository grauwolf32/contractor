package contracts

import (
	"encoding/json"
	"time"
)

// WorkerHandle addresses the Worker of one prepared allocation over A2A.
type WorkerHandle struct {
	AllocationID string `json:"allocationId"`
	// RuntimeAgentID is Server-owned routing/authentication metadata. It is
	// never accepted from or emitted to the Runtime private wire response.
	RuntimeAgentID   string           `json:"-"`
	AgentTemplateRef AgentTemplateRef `json:"agentTemplateRef"`
	WorkerRuntimeRef WorkerRuntimeRef `json:"workerRuntimeRef"`
	AgentCard        map[string]any   `json:"agentCard"`
	LeaseExpiresAt   time.Time        `json:"leaseExpiresAt"`
}

// Clone detaches the Agent Card before a handle crosses a Planner adapter
// boundary. The card is JSON data, so a JSON round trip copies it deeply; a
// card holding a non-JSON value is unusable by A2A and is omitted, so an
// invalid in-process input cannot bypass the copy by failing serialization.
func (h WorkerHandle) Clone() WorkerHandle {
	result := h
	// Decoding into a non-nil map reuses it, so the copy starts without one.
	result.AgentCard = nil
	if encoded, err := json.Marshal(h.AgentCard); err == nil {
		_ = json.Unmarshal(encoded, &result.AgentCard)
	}
	return result
}
