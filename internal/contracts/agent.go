package contracts

import "time"

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

type AgentRegistration struct {
	APIVersion               string              `json:"apiVersion"`
	InstanceID               string              `json:"instanceId"`
	StartedAt                time.Time           `json:"startedAt"`
	ControlURL               string              `json:"controlUrl"`
	A2AURL                   string              `json:"a2aUrl"`
	SupportedRuntimes        []string            `json:"supportedRuntimes"`
	SupportedToolsets        []ToolsetCapability `json:"supportedToolsets"`
	SupportedSandboxProfiles []string            `json:"supportedSandboxProfiles"`
	ObservedState            AgentObservedState  `json:"observedState"`
	AllocationID             *string             `json:"allocationId,omitempty"`
}

type AgentRegistrationResponse struct {
	APIVersion               string `json:"apiVersion"`
	HeartbeatIntervalSeconds int    `json:"heartbeatIntervalSeconds"`
	ConfirmedLeaseSeconds    int    `json:"confirmedLeaseSeconds"`
}

func (r AgentRegistrationResponse) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if r.HeartbeatIntervalSeconds <= 0 || r.ConfirmedLeaseSeconds <= 0 {
		return invalidf("heartbeat interval and confirmed lease must be positive")
	}
	if r.ConfirmedLeaseSeconds <= r.HeartbeatIntervalSeconds {
		return invalidf("confirmed lease must exceed heartbeat interval")
	}
	return nil
}

func (r AgentRegistration) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("instanceId", r.InstanceID); err != nil {
		return err
	}
	if r.StartedAt.IsZero() {
		return invalidf("startedAt must not be zero")
	}
	if err := validateURL("controlUrl", r.ControlURL); err != nil {
		return err
	}
	if err := validateURL("a2aUrl", r.A2AURL); err != nil {
		return err
	}
	if err := validateCapabilities(r.SupportedRuntimes, r.SupportedToolsets, r.SupportedSandboxProfiles); err != nil {
		return err
	}
	return validateObservedAllocation(r.ObservedState, r.AllocationID)
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
	case ActionDrain, ActionRelease:
		if r.AllocationID == nil {
			return invalidf("allocationId is required for action %s", r.Action)
		}
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
		if err := validateSelector("supportedToolsets.ref", capability.Ref); err != nil {
			return err
		}
		if _, exists := seen[capability.Ref]; exists {
			return invalidf("duplicate toolset capability %q", capability.Ref)
		}
		seen[capability.Ref] = struct{}{}
		if len(capability.Tools) == 0 {
			return invalidf("toolset capability %q has no tools", capability.Ref)
		}
		toolNames := make(map[string]struct{})
		for _, tool := range capability.Tools {
			if err := validateOpaqueID("tool name", tool); err != nil {
				return err
			}
			if _, exists := toolNames[tool]; exists {
				return invalidf("duplicate tool %q in capability %q", tool, capability.Ref)
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
			return invalidf("idle agent must not report allocationId")
		}
		return nil
	case AgentAllocated, AgentDraining, AgentFenced:
		if allocationID == nil {
			return invalidf("%s agent must report allocationId", state)
		}
		return validateOpaqueID("allocationId", *allocationID)
	default:
		return invalidf("unknown observedState %q", state)
	}
}
