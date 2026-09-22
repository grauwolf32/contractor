package contracts

// Runtime Agent registration: the registration DTO, its response, and the
// normalization/fingerprint projection used by registration retries.
// Mirrors the Python runtime's contracts/registration.py. Heartbeats live
// in agent.go.

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"sort"
	"time"
)

var runtimeAgentIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

type AgentRegistration struct {
	Capabilities *RuntimeCompletionCapabilities `json:"capabilities,omitempty"`
	APIVersion   string                         `json:"apiVersion"`

	InstanceID                          string                     `json:"instanceId"`
	SoftwareVersion                     string                     `json:"softwareVersion"`
	StartedAt                           time.Time                  `json:"startedAt"`
	ControlURL                          string                     `json:"controlUrl"`
	A2AURL                              string                     `json:"a2aUrl"`
	InitialLabels                       []string                   `json:"initialLabels"`
	SupportedRuntimes                   []string                   `json:"supportedRuntimes"`
	SupportedToolsets                   []ToolsetCapability        `json:"supportedToolsets"`
	SupportedSandboxProfiles            []string                   `json:"supportedSandboxProfiles"`
	SupportedRuntimeAdapters            []RuntimeAdapterRef        `json:"supportedRuntimeAdapters"`
	SupportedPerformanceMetricsVersions PerformanceMetricsVersions `json:"supportedPerformanceMetricsVersions,omitempty"`
	WorkspaceCapabilities               *WorkspaceCapabilities     `json:"workspaceCapabilities,omitempty"`
	ObservedState                       AgentObservedState         `json:"observedState"`
	AllocationID                        *string                    `json:"allocationId,omitempty"`
}

func (r AgentRegistration) Validate() error {
	if r.Capabilities != nil {
		if err := r.Capabilities.Validate(); err != nil {
			return err
		}
	}
	if len(r.SupportedPerformanceMetricsVersions) > 1 || (len(r.SupportedPerformanceMetricsVersions) == 1 && r.SupportedPerformanceMetricsVersions[0] != 1) {
		return invalidf("unsupported performance metrics capability")
	}
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("instanceId", r.InstanceID); err != nil {
		return err
	}
	if len(r.SoftwareVersion) == 0 || len(r.SoftwareVersion) > 128 ||
		!versionPattern.MatchString(r.SoftwareVersion) {
		return invalidf("softwareVersion must be a bounded version string")
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
	if len(r.SupportedRuntimes) > 128 || len(r.SupportedToolsets) > 128 ||
		len(r.SupportedSandboxProfiles) > 128 {
		return invalidf("Runtime capability collection exceeds its bound")
	}
	if err := validateCapabilities(r.SupportedRuntimes, r.SupportedToolsets, r.SupportedSandboxProfiles); err != nil {
		return err
	}
	if err := validateSortedLabels("initialLabels", r.InitialLabels, 32, false); err != nil {
		return err
	}
	if err := validateSortedRuntimeAdapters(r.SupportedRuntimeAdapters); err != nil {
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
	APIVersion string `json:"apiVersion"`

	RuntimeAgentID           string   `json:"runtimeAgentId"`
	Labels                   []string `json:"labels"`
	LabelRevision            uint64   `json:"labelRevision"`
	HeartbeatIntervalSeconds int      `json:"heartbeatIntervalSeconds"`
	ConfirmedLeaseSeconds    int      `json:"confirmedLeaseSeconds"`
}

func (r AgentRegistrationResponse) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if !runtimeAgentIDPattern.MatchString(r.RuntimeAgentID) {
		return invalidf("runtimeAgentId must be a lowercase SHA-256 SPKI fingerprint")
	}
	if err := validateSortedLabels("labels", r.Labels, 32, false); err != nil {
		return err
	}
	if r.LabelRevision == 0 {
		return invalidf("labelRevision must be positive")
	}
	if r.HeartbeatIntervalSeconds <= 0 || r.ConfirmedLeaseSeconds <= r.HeartbeatIntervalSeconds {
		return invalidf("confirmed lease must exceed a positive heartbeat interval")
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
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	result.SupportedPerformanceMetricsVersions = append([]int(nil), source.SupportedPerformanceMetricsVersions...)
	if source.WorkspaceCapabilities != nil {
		capabilities := *source.WorkspaceCapabilities
		capabilities.Modes = append([]WorkspaceMode{}, source.WorkspaceCapabilities.Modes...)
		result.WorkspaceCapabilities = &capabilities
	}
	result.SupportedToolsets = make([]ToolsetCapability, len(source.SupportedToolsets))
	for index, capability := range source.SupportedToolsets {
		result.SupportedToolsets[index] = capability
		result.SupportedToolsets[index].Tools = append([]string(nil), capability.Tools...)
		sort.Strings(result.SupportedToolsets[index].Tools)
	}
	sort.Strings(result.InitialLabels)
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
// labels. It is not the TLS principal fingerprint.
func AgentRegistrationFingerprint(source AgentRegistration) (string, error) {
	normalized := NormalizeAgentRegistration(source)
	normalized.InitialLabels = nil
	normalized.ObservedState = ""
	normalized.AllocationID = nil
	canonical, err := MarshalPrivateCanonical(normalized)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
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
