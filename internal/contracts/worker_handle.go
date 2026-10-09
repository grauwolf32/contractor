package contracts

import "time"

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
