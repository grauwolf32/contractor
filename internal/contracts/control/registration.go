package control

// Runtime Agent registration: the registration DTO, its response, and the
// normalization/fingerprint projection used by registration retries.
// Mirrors the Python runtime's contracts/registration.py. Heartbeats live
// in agent.go.

import (
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

type AgentRegistration struct {
	Capabilities *contracts.RuntimeCompletionCapabilities `json:"capabilities,omitempty" strictjson:"nullable"`
	APIVersion   string                                   `json:"apiVersion" strictjson:"required"`

	InstanceID                          string                               `json:"instanceId" strictjson:"required"`
	SoftwareVersion                     string                               `json:"softwareVersion" strictjson:"required"`
	StartedAt                           time.Time                            `json:"startedAt" strictjson:"required"`
	ControlURL                          string                               `json:"controlUrl" strictjson:"required"`
	A2AURL                              string                               `json:"a2aUrl" strictjson:"required"`
	InitialLabels                       []string                             `json:"initialLabels" strictjson:"required"`
	PlacementLabels                     []string                             `json:"placementLabels,omitempty"`
	SupportedRuntimes                   []string                             `json:"supportedRuntimes" strictjson:"required"`
	SupportedToolsets                   []ToolsetCapability                  `json:"supportedToolsets" strictjson:"required"`
	SupportedSandboxProfiles            []string                             `json:"supportedSandboxProfiles" strictjson:"required"`
	SupportedRuntimeAdapters            []runtimesettings.RuntimeAdapterRef  `json:"supportedRuntimeAdapters" strictjson:"required"`
	SupportedPerformanceMetricsVersions reporting.PerformanceMetricsVersions `json:"supportedPerformanceMetricsVersions,omitempty"`
	WorkspaceCapabilities               *contracts.WorkspaceCapabilities     `json:"workspaceCapabilities,omitempty" strictjson:"nullable"`
	ObservedState                       AgentObservedState                   `json:"observedState" strictjson:"required"`
	AllocationID                        *string                              `json:"allocationId,omitempty"`
}

func (r AgentRegistration) Validate() error {
	if r.Capabilities != nil {
		if err := r.Capabilities.Validate(); err != nil {
			return err
		}
	}
	if len(r.SupportedPerformanceMetricsVersions) > 1 || (len(r.SupportedPerformanceMetricsVersions) == 1 && r.SupportedPerformanceMetricsVersions[0] != 1) {
		return contracts.Invalidf("unsupported performance metrics capability")
	}
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := contracts.ValidateResourceID("instanceId", r.InstanceID); err != nil {
		return err
	}
	if len(r.SoftwareVersion) == 0 || len(r.SoftwareVersion) > 128 ||
		!contracts.ValidVersion(r.SoftwareVersion) {
		return contracts.Invalidf("softwareVersion must be a bounded version string")
	}
	if r.StartedAt.IsZero() {
		return contracts.Invalidf("startedAt must not be zero")
	}
	if err := contracts.ValidateURL("controlUrl", r.ControlURL); err != nil {
		return err
	}
	if err := contracts.ValidateURL("a2aUrl", r.A2AURL); err != nil {
		return err
	}
	if err := validateCapabilities(r.SupportedRuntimes, r.SupportedToolsets, r.SupportedSandboxProfiles); err != nil {
		return err
	}
	if err := validateSortedLabels("initialLabels", r.InitialLabels, 32, false); err != nil {
		return err
	}
	if err := contracts.ValidatePlacementLabels(r.PlacementLabels); err != nil {
		return err
	}
	if err := runtimesettings.ValidateRuntimeAdapterRefs(r.SupportedRuntimeAdapters); err != nil {
		return err
	}
	if r.WorkspaceCapabilities != nil {
		if err := r.WorkspaceCapabilities.Validate(); err != nil {
			return err
		}
	}
	return validateObservedAllocation(r.ObservedState, r.AllocationID)
}

type AgentRegistrationResponse struct {
	APIVersion string `json:"apiVersion" strictjson:"required"`

	RuntimeAgentID           string   `json:"runtimeAgentId" strictjson:"required"`
	Labels                   []string `json:"labels" strictjson:"required"`
	LabelRevision            uint64   `json:"labelRevision" strictjson:"required"`
	HeartbeatIntervalSeconds int      `json:"heartbeatIntervalSeconds" strictjson:"required"`
	ConfirmedLeaseSeconds    int      `json:"confirmedLeaseSeconds" strictjson:"required"`
}

func (r AgentRegistrationResponse) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if !contracts.ValidRuntimeAgentID(r.RuntimeAgentID) {
		return contracts.Invalidf("runtimeAgentId must be a lowercase SHA-256 SPKI fingerprint")
	}
	if err := validateSortedLabels("labels", r.Labels, 32, false); err != nil {
		return err
	}
	if r.LabelRevision == 0 {
		return contracts.Invalidf("labelRevision must be positive")
	}
	if r.HeartbeatIntervalSeconds <= 0 || r.ConfirmedLeaseSeconds <= r.HeartbeatIntervalSeconds {
		return contracts.Invalidf("confirmed lease must exceed a positive heartbeat interval")
	}
	return nil
}

// NormalizeAgentRegistration produces the immutable ordering used for
// registration retries and fingerprints. Decode still requires sorted wire
// sets so normalization cannot hide a malformed peer.
func NormalizeAgentRegistration(source AgentRegistration) AgentRegistration {
	result := source
	if source.Capabilities != nil {
		value := *source.Capabilities
		value.CompletionContracts = append([]string{}, source.Capabilities.CompletionContracts...)
		result.Capabilities = &value
	}
	result.InitialLabels = append([]string{}, source.InitialLabels...)
	result.PlacementLabels = append([]string{}, source.PlacementLabels...)
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]runtimesettings.RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	result.SupportedPerformanceMetricsVersions = append([]int(nil), source.SupportedPerformanceMetricsVersions...)
	if source.WorkspaceCapabilities != nil {
		capabilities := *source.WorkspaceCapabilities
		capabilities.Modes = append([]contracts.WorkspaceMode{}, source.WorkspaceCapabilities.Modes...)
		result.WorkspaceCapabilities = &capabilities
	}
	result.SupportedToolsets = make([]ToolsetCapability, len(source.SupportedToolsets))
	for index, capability := range source.SupportedToolsets {
		result.SupportedToolsets[index] = capability
		result.SupportedToolsets[index].Tools = append([]string(nil), capability.Tools...)
		sort.Strings(result.SupportedToolsets[index].Tools)
	}
	sort.Strings(result.InitialLabels)
	sort.Strings(result.PlacementLabels)
	sort.Strings(result.SupportedRuntimes)
	sort.Strings(result.SupportedSandboxProfiles)
	sort.Slice(result.SupportedRuntimeAdapters, func(i, j int) bool {
		return result.SupportedRuntimeAdapters[i] < result.SupportedRuntimeAdapters[j]
	})
	sort.Slice(result.SupportedToolsets, func(i, j int) bool {
		return result.SupportedToolsets[i].Ref < result.SupportedToolsets[j].Ref
	})
	return result
}

// AgentRegistrationFingerprint excludes mutable observation and startup
// configuration-label seeds. Placement labels are immutable and remain in the
// fingerprint. It is not the TLS principal fingerprint.
func AgentRegistrationFingerprint(source AgentRegistration) (string, error) {
	normalized := NormalizeAgentRegistration(source)
	normalized.InitialLabels = nil
	normalized.ObservedState = ""
	normalized.AllocationID = nil
	canonical, err := strictjson.Canonical(normalized)
	if err != nil {
		return "", err
	}
	return contentdigest.Bytes(canonical), nil
}

// RuntimeAdapterCapabilityProjection returns a detached sorted safe view for
// Operations. It contains refs only, never settings or probe diagnostics.
func RuntimeAdapterCapabilityProjection(source AgentRegistration) []string {
	result := make([]string, len(source.SupportedRuntimeAdapters))
	for index, ref := range source.SupportedRuntimeAdapters {
		result[index] = string(ref)
	}
	sort.Strings(result)
	return result
}

func validateSortedLabels(field string, values []string, maximum int, allowDefault bool) error {
	if values == nil || len(values) > maximum {
		return contracts.Invalidf("%s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value) == 0 || len(value) > 63 || !contracts.ValidIdentifier(value) ||
			(!allowDefault && value == "default") {
			return contracts.Invalidf("%s contains an invalid label", field)
		}
		if value <= previous {
			return contracts.Invalidf("%s must be sorted and unique", field)
		}
		previous = value
	}
	return nil
}
