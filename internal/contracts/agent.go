package contracts

import (
	"bytes"
	"encoding/json"
)

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
	Ref           string   `json:"ref"`
	Tools         []string `json:"tools"`
	ToolDiscovery string   `json:"toolDiscovery,omitempty"`
}

func (c *ToolsetCapability) UnmarshalJSON(raw []byte) error {
	type wire ToolsetCapability
	var value wire
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return invalidf("invalid toolset capability")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return invalidf("invalid toolset capability")
	}
	if _, present := fields["toolDiscovery"]; present && value.ToolDiscovery != "allocation" {
		return invalidf("invalid tool discovery mode")
	}
	*c = ToolsetCapability(value)
	return nil
}

// Validate describes installation readiness, never remote endpoint readiness.
// Allocation discovery deliberately advertises no invented tool inventory.
func (c ToolsetCapability) Validate() error {
	if err := validateSelector("supportedToolsets.ref", c.Ref); err != nil {
		return err
	}
	if c.Tools == nil {
		return invalidf("toolset capability tools must be an array")
	}
	switch c.ToolDiscovery {
	case "":
		if len(c.Tools) == 0 {
			return invalidf("static toolset capability has no tools")
		}
	case "allocation":
		if len(c.Tools) != 0 {
			return invalidf("allocation toolset capability must have empty tools")
		}
	default:
		return invalidf("unknown tool discovery mode")
	}
	seen := make(map[string]struct{})
	for _, tool := range c.Tools {
		if err := validateOpaqueID("tool name", tool); err != nil {
			return err
		}
		if _, exists := seen[tool]; exists {
			return invalidf("duplicate tool in capability")
		}
		seen[tool] = struct{}{}
	}
	return nil
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
	if err := validateAPIVersion(h.APIVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("instanceId", h.InstanceID); err != nil {
		return err
	}
	if h.HeartbeatSeq == 0 {
		return invalidf("heartbeatSeq must be positive")
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
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if r.AckSeq == 0 {
		return invalidf("ackSeq must be positive")
	}
	switch r.Action {
	case ActionContinue, ActionReregister:
	case ActionDrain:
		if r.AllocationID == nil {
			return invalidf("allocationId is required for action %s", r.Action)
		}
	case ActionRelease:
	default:
		return invalidf("unknown heartbeat action %q", r.Action)
	}
	if r.AllocationID != nil {
		return validateOpaqueID("allocationId", *r.AllocationID)
	}
	return nil
}

func validateCapabilities(runtimes []string, toolsets []ToolsetCapability, sandboxes []string) error {
	if len(runtimes) == 0 || len(sandboxes) == 0 {
		return invalidf("supportedRuntimes and supportedSandboxProfiles must not be empty")
	}
	seen := make(map[string]struct{})
	for _, value := range runtimes {
		if err := validateSelector("supportedRuntimes", value); err != nil {
			return err
		}
		if _, exists := seen[value]; exists {
			return invalidf("duplicate runtime capability %q", value)
		}
		seen[value] = struct{}{}
	}
	seen = make(map[string]struct{})
	for _, value := range sandboxes {
		if err := validateSelector("supportedSandboxProfiles", value); err != nil {
			return err
		}
		if _, exists := seen[value]; exists {
			return invalidf("duplicate sandbox capability %q", value)
		}
		seen[value] = struct{}{}
	}
	seen = make(map[string]struct{})
	for _, capability := range toolsets {
		if err := capability.Validate(); err != nil {
			return err
		}
		if _, exists := seen[capability.Ref]; exists {
			return invalidf("duplicate toolset capability %q", capability.Ref)
		}
		seen[capability.Ref] = struct{}{}
	}
	return nil
}

func validateObservedAllocation(state AgentObservedState, allocationID *string) error {
	switch state {
	case AgentIdle:
		if allocationID != nil {
			return invalidf("idle agent must not report allocationId")
		}
		return nil
	case AgentFenced:
		if allocationID == nil {
			return nil
		}
		return validateOpaqueID("allocationId", *allocationID)
	case AgentAllocated, AgentDraining:
		if allocationID == nil {
			return invalidf("%s agent must report allocationId", state)
		}
		return validateOpaqueID("allocationId", *allocationID)
	default:
		return invalidf("unknown observedState %q", state)
	}
}
