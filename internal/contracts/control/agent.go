package control

// Runtime Agent heartbeats and the reconciliation actions answering them.
// Registration lives in registration.go.

import "github.com/grauwolf32/contractor/internal/contracts"

type AgentObservedState string

const (
	AgentIdle      AgentObservedState = "idle"
	AgentAllocated AgentObservedState = "allocated"
	AgentDraining  AgentObservedState = "draining"
	AgentFenced    AgentObservedState = "fenced"
)

type ReconciliationAction string

const (
	ActionContinue   ReconciliationAction = "continue"
	ActionDrain      ReconciliationAction = "drain"
	ActionRelease    ReconciliationAction = "release"
	ActionReregister ReconciliationAction = "reregister"
)

type ToolsetCapability struct {
	Ref   string   `json:"ref"`
	Tools []string `json:"tools"`
}

type AgentHeartbeat struct {
	APIVersion    string             `json:"apiVersion"`
	InstanceID    string             `json:"instanceId"`
	HeartbeatSeq  uint64             `json:"heartbeatSeq"`
	EchoedAckSeq  uint64             `json:"echoedAckSeq"`
	ObservedState AgentObservedState `json:"observedState"`
	AllocationID  *string            `json:"allocationId,omitempty"`
}

func (h AgentHeartbeat) Validate() error {
	if err := contracts.ValidateAPIVersion(h.APIVersion); err != nil {
		return err
	}
	if err := contracts.ValidateOpaqueID("instanceId", h.InstanceID); err != nil {
		return err
	}
	if h.HeartbeatSeq == 0 {
		return contracts.Invalidf("heartbeatSeq must be positive")
	}
	return validateObservedAllocation(h.ObservedState, h.AllocationID)
}

type HeartbeatResponse struct {
	APIVersion   string               `json:"apiVersion"`
	AckSeq       uint64               `json:"ackSeq"`
	Action       ReconciliationAction `json:"action"`
	AllocationID *string              `json:"allocationId,omitempty"`
}

func (r HeartbeatResponse) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if r.AckSeq == 0 {
		return contracts.Invalidf("ackSeq must be positive")
	}
	switch r.Action {
	case ActionContinue, ActionReregister:
	case ActionDrain:
		if r.AllocationID == nil {
			return contracts.Invalidf("allocationId is required for action %s", r.Action)
		}
	case ActionRelease:
	default:
		return contracts.Invalidf("unknown heartbeat action %q", r.Action)
	}
	if r.AllocationID != nil {
		return contracts.ValidateOpaqueID("allocationId", *r.AllocationID)
	}
	return nil
}

func validateCapabilities(runtimes []string, toolsets []ToolsetCapability, sandboxes []string) error {
	if len(runtimes) == 0 || len(sandboxes) == 0 {
		return contracts.Invalidf("supportedRuntimes and supportedSandboxProfiles must not be empty")
	}
	seen := make(map[string]struct{})
	for _, value := range runtimes {
		if err := contracts.ValidateSelector("supportedRuntimes", value); err != nil {
			return err
		}
		if _, exists := seen[value]; exists {
			return contracts.Invalidf("duplicate runtime capability %q", value)
		}
		seen[value] = struct{}{}
	}
	seen = make(map[string]struct{})
	for _, value := range sandboxes {
		if err := contracts.ValidateSelector("supportedSandboxProfiles", value); err != nil {
			return err
		}
		if _, exists := seen[value]; exists {
			return contracts.Invalidf("duplicate sandbox capability %q", value)
		}
		seen[value] = struct{}{}
	}
	seen = make(map[string]struct{})
	for _, capability := range toolsets {
		if err := contracts.ValidateSelector("supportedToolsets.ref", capability.Ref); err != nil {
			return err
		}
		if _, exists := seen[capability.Ref]; exists {
			return contracts.Invalidf("duplicate toolset capability %q", capability.Ref)
		}
		seen[capability.Ref] = struct{}{}
		if len(capability.Tools) == 0 {
			return contracts.Invalidf("toolset capability %q has no tools", capability.Ref)
		}
		toolNames := make(map[string]struct{})
		for _, tool := range capability.Tools {
			if err := contracts.ValidateOpaqueID("tool name", tool); err != nil {
				return err
			}
			if _, exists := toolNames[tool]; exists {
				return contracts.Invalidf("duplicate tool %q in capability %q", tool, capability.Ref)
			}
			toolNames[tool] = struct{}{}
		}
	}
	return nil
}

func validateObservedAllocation(state AgentObservedState, allocationID *string) error {
	switch state {
	case AgentIdle:
		if allocationID != nil {
			return contracts.Invalidf("idle agent must not report allocationId")
		}
		return nil
	case AgentFenced:
		if allocationID == nil {
			return nil
		}
		return contracts.ValidateOpaqueID("allocationId", *allocationID)
	case AgentAllocated, AgentDraining:
		if allocationID == nil {
			return contracts.Invalidf("%s agent must report allocationId", state)
		}
		return contracts.ValidateOpaqueID("allocationId", *allocationID)
	default:
		return contracts.Invalidf("unknown observedState %q", state)
	}
}
