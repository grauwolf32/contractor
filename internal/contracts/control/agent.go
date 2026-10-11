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
	if err := contracts.ValidateResourceID("instanceId", h.InstanceID); err != nil {
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

// Runtime capability bounds of one registration. The Operations read model
// shows registrations through the same rules, and the shared cases in
// api/testdata/v1alpha1/runtime-capability-cases.json hold the private
// schemas, the public OpenAPI and the Runtime to them.
const (
	// MaxRuntimeCapabilityRefs bounds supportedRuntimes, supportedToolsets and
	// supportedSandboxProfiles, each.
	MaxRuntimeCapabilityRefs = 128
	// MaxRuntimeCapabilityRefLength bounds one capability ref.
	MaxRuntimeCapabilityRefLength = 256
	// MaxToolsetCapabilityTools bounds the tools of one Toolset capability.
	MaxToolsetCapabilityTools = 256
)

// ValidRuntimeCapabilityRef reports whether value is an exact
// <id>@<version> capability ref of at most MaxRuntimeCapabilityRefLength.
func ValidRuntimeCapabilityRef(value string) bool {
	return len(value) <= MaxRuntimeCapabilityRefLength && contracts.ValidSelector(value)
}

// ValidateToolsetCapability requires a valid capability ref and 1 to
// MaxToolsetCapabilityTools unique tool names in the shared name grammar
// (contracts.ValidName).
func ValidateToolsetCapability(ref string, tools []string) error {
	if !ValidRuntimeCapabilityRef(ref) {
		return contracts.Invalidf("supportedToolsets.ref must be an exact selector of at most %d characters", MaxRuntimeCapabilityRefLength)
	}
	if len(tools) == 0 || len(tools) > MaxToolsetCapabilityTools {
		return contracts.Invalidf("toolset capability %q must have 1 to %d tools", ref, MaxToolsetCapabilityTools)
	}
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if !contracts.ValidName(tool) {
			return contracts.Invalidf("toolset capability %q has an invalid tool name", ref)
		}
		if _, exists := seen[tool]; exists {
			return contracts.Invalidf("duplicate tool %q in capability %q", tool, ref)
		}
		seen[tool] = struct{}{}
	}
	return nil
}

func validateCapabilities(runtimes []string, toolsets []ToolsetCapability, sandboxes []string) error {
	if len(runtimes) == 0 || len(sandboxes) == 0 {
		return contracts.Invalidf("supportedRuntimes and supportedSandboxProfiles must not be empty")
	}
	if len(runtimes) > MaxRuntimeCapabilityRefs || len(toolsets) > MaxRuntimeCapabilityRefs ||
		len(sandboxes) > MaxRuntimeCapabilityRefs {
		return contracts.Invalidf("Runtime capability collection exceeds its bound")
	}
	if err := validateCapabilityRefs("supportedRuntimes", "runtime", runtimes); err != nil {
		return err
	}
	if err := validateCapabilityRefs("supportedSandboxProfiles", "sandbox", sandboxes); err != nil {
		return err
	}
	seen := make(map[string]struct{}, len(toolsets))
	for _, capability := range toolsets {
		if err := ValidateToolsetCapability(capability.Ref, capability.Tools); err != nil {
			return err
		}
		if _, exists := seen[capability.Ref]; exists {
			return contracts.Invalidf("duplicate toolset capability %q", capability.Ref)
		}
		seen[capability.Ref] = struct{}{}
	}
	return nil
}

func validateCapabilityRefs(field, kind string, values []string) error {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !ValidRuntimeCapabilityRef(value) {
			return contracts.Invalidf("%s must hold exact selectors of at most %d characters", field, MaxRuntimeCapabilityRefLength)
		}
		if _, exists := seen[value]; exists {
			return contracts.Invalidf("duplicate %s capability %q", kind, value)
		}
		seen[value] = struct{}{}
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
		return contracts.ValidateResourceID("allocationId", *allocationID)
	case AgentAllocated, AgentDraining:
		if allocationID == nil {
			return contracts.Invalidf("%s agent must report allocationId", state)
		}
		return contracts.ValidateResourceID("allocationId", *allocationID)
	default:
		return contracts.Invalidf("unknown observedState %q", state)
	}
}
