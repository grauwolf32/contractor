package contracts

import (
	"bytes"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/ucarion/jcs"
	"golang.org/x/text/unicode/norm"
)

const (
	PrivateProtocolVersionV2 = 2

	RuntimeAdapterOTLPHTTP  RuntimeAdapterRef = "otlp-http@1"
	RuntimeAdapterHTTPProxy RuntimeAdapterRef = "http-proxy@1"

	RuntimeCredentialOTLPHeaders RuntimeCredentialKind = "otlp-headers@1"
	RuntimeCredentialProxyBasic  RuntimeCredentialKind = "http-proxy-basic@1"
	RuntimeCredentialProxyBearer RuntimeCredentialKind = "http-proxy-bearer@1"
)

const (
	PrivateProtocolErrorVersion   PrivateProtocolErrorClass = "version"
	PrivateProtocolErrorDuplicate PrivateProtocolErrorClass = "duplicate_key"
	PrivateProtocolErrorSchema    PrivateProtocolErrorClass = "schema"
	PrivateProtocolErrorInvariant PrivateProtocolErrorClass = "invariant"
)

const (
	ProxyTargetLLMGateway     HTTPProxyTarget = "llm-gateway"
	ProxyTargetToolHTTP       HTTPProxyTarget = "tool-http"
	ProxyTargetToolSubprocess HTTPProxyTarget = "tool-subprocess"
)

var (
	runtimeAgentIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
	headerNamePattern     = regexp.MustCompile(`^[!#$%&'*+\-.^_` + "`" + `|~0-9A-Za-z]+$`)
	privateVersionError   = errors.New("private protocol version mismatch")
)

var runtimeAdapterErrorCodes = map[string]struct{}{
	"close_failed": {}, "delivery_failed": {}, "flush_failed": {},
	"flush_timeout": {}, "queue_overflow": {}, "request_failed": {},
}

var forbiddenRuntimeHeaderNames = map[string]struct{}{
	"connection": {}, "content-length": {}, "host": {}, "keep-alive": {},
	"proxy-authenticate": {}, "proxy-authorization": {}, "proxy-connection": {},
	"te": {}, "trailer": {}, "transfer-encoding": {}, "upgrade": {},
}

// PrivateProtocolError is deliberately detail-free: malformed private input
// can contain credentials and must not be copied into errors or logs.
type PrivateProtocolErrorClass string

type PrivateProtocolError struct {
	Class PrivateProtocolErrorClass
}

func (e *PrivateProtocolError) Error() string {
	return "private protocol v2 " + string(e.Class) + " error"
}

func (e *PrivateProtocolError) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, e.Error())
}

type RuntimeAdapterRef string

func (r RuntimeAdapterRef) Validate() error {
	switch r {
	case RuntimeAdapterOTLPHTTP, RuntimeAdapterHTTPProxy:
		return nil
	default:
		return invalidf("unknown RuntimeAdapter ref")
	}
}

type RuntimeCredentialKind string

func (k RuntimeCredentialKind) Validate() error {
	switch k {
	case RuntimeCredentialOTLPHeaders, RuntimeCredentialProxyBasic, RuntimeCredentialProxyBearer:
		return nil
	default:
		return invalidf("unknown Runtime credential kind")
	}
}

type AgentRegistrationV2 struct {
	APIVersion               string                   `json:"apiVersion"`
	PrivateProtocolVersion   int                      `json:"privateProtocolVersion"`
	InstanceID               string                   `json:"instanceId"`
	SoftwareVersion          string                   `json:"softwareVersion"`
	StartedAt                time.Time                `json:"startedAt"`
	ControlURL               string                   `json:"controlUrl"`
	A2AURL                   string                   `json:"a2aUrl"`
	InitialLabels            []string                 `json:"initialLabels"`
	SupportedRuntimes        []string                 `json:"supportedRuntimes"`
	SupportedToolsets        []ToolsetCapability      `json:"supportedToolsets"`
	SupportedSandboxProfiles []string                 `json:"supportedSandboxProfiles"`
	SupportedRuntimeAdapters []RuntimeAdapterRef      `json:"supportedRuntimeAdapters"`
	WorkspaceCapabilities    *WorkspaceCapabilitiesV2 `json:"workspaceCapabilities,omitempty"`
	ObservedState            AgentObservedState       `json:"observedState"`
	AllocationID             *string                  `json:"allocationId,omitempty"`
}

func (r AgentRegistrationV2) Validate() error {
	if err := validatePrivateProtocolVersion(r.PrivateProtocolVersion); err != nil {
		return err
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

type WorkspaceModeV2 string

const (
	WorkspaceModeDirect  WorkspaceModeV2 = "direct"
	WorkspaceModeOverlay WorkspaceModeV2 = "overlay"
)

func (m WorkspaceModeV2) Validate() error {
	switch m {
	case WorkspaceModeDirect, WorkspaceModeOverlay:
		return nil
	default:
		return invalidf("workspace mode is invalid")
	}
}

type WorkspaceStorageV2 string

const (
	WorkspaceStorageLocal  WorkspaceStorageV2 = "local"
	WorkspaceStorageMemory WorkspaceStorageV2 = "memory"
)

func (s WorkspaceStorageV2) Validate() error {
	switch s {
	case WorkspaceStorageLocal, WorkspaceStorageMemory:
		return nil
	default:
		return invalidf("workspace storage is invalid")
	}
}

type WorkspaceLimitsV2 struct {
	MaxFiles            int   `json:"maxFiles"`
	MaxExpandedBytes    int64 `json:"maxExpandedBytes"`
	MaxManagedTextBytes int64 `json:"maxManagedTextBytes"`
	MaxFileBytes        int64 `json:"maxFileBytes"`
}

func (l WorkspaceLimitsV2) Validate() error {
	if l.MaxFiles <= 0 || l.MaxExpandedBytes <= 0 || l.MaxManagedTextBytes <= 0 || l.MaxFileBytes <= 0 ||
		l.MaxFileBytes > l.MaxExpandedBytes || l.MaxManagedTextBytes > l.MaxExpandedBytes {
		return invalidf("workspace limits are invalid")
	}
	return nil
}

type WorkspaceCapabilitiesV2 struct {
	Storage WorkspaceStorageV2 `json:"storage"`
	Modes   []WorkspaceModeV2  `json:"modes"`
	Limits  WorkspaceLimitsV2  `json:"limits"`
}

func (c WorkspaceCapabilitiesV2) Validate() error {
	if err := c.Storage.Validate(); err != nil {
		return err
	}
	if len(c.Modes) == 0 || len(c.Modes) > 2 {
		return invalidf("workspace capability modes must be a non-empty bounded array")
	}
	previous := ""
	for _, mode := range c.Modes {
		if err := mode.Validate(); err != nil {
			return err
		}
		if string(mode) <= previous {
			return invalidf("workspace capability modes must be sorted and unique")
		}
		previous = string(mode)
	}
	return c.Limits.Validate()
}

type AllocationWorkspaceSourceV2 struct {
	Artifact ArtifactRef `json:"artifact"`
	Target   string      `json:"target"`
}

type AllocationWorkspaceStateV2 struct {
	Artifact ArtifactRef `json:"artifact"`
}

type AllocationWorkspaceExportV2 struct {
	State string `json:"state"`
	Diff  string `json:"diff"`
}

type AllocationWorkspaceSpecV2 struct {
	Mode    WorkspaceModeV2               `json:"mode"`
	Sources []AllocationWorkspaceSourceV2 `json:"sources"`
	State   *AllocationWorkspaceStateV2   `json:"state,omitempty"`
	Export  *AllocationWorkspaceExportV2  `json:"export,omitempty"`
}

// CloneAllocationWorkspaceSpecV2 returns a detached copy suitable for crossing
// ownership boundaries between Scheduler, Control Plane and Runtime clients.
func CloneAllocationWorkspaceSpecV2(source *AllocationWorkspaceSpecV2) *AllocationWorkspaceSpecV2 {
	if source == nil {
		return nil
	}
	result := *source
	result.Sources = make([]AllocationWorkspaceSourceV2, len(source.Sources))
	for index, item := range source.Sources {
		result.Sources[index] = item
		result.Sources[index].Artifact = cloneWorkspaceArtifactRef(item.Artifact)
	}
	if source.State != nil {
		state := *source.State
		state.Artifact = cloneWorkspaceArtifactRef(source.State.Artifact)
		result.State = &state
	}
	if source.Export != nil {
		export := *source.Export
		result.Export = &export
	}
	return &result
}

func cloneWorkspaceArtifactRef(source ArtifactRef) ArtifactRef {
	result := source
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return result
}

func (s AllocationWorkspaceSpecV2) Validate() error {
	if err := s.Mode.Validate(); err != nil {
		return err
	}
	if len(s.Sources) == 0 || len(s.Sources) > 32 {
		return invalidf("workspace sources must be a non-empty bounded array")
	}
	targets := make([]string, 0, len(s.Sources))
	for _, source := range s.Sources {
		if err := source.Artifact.ValidateExact(); err != nil {
			return err
		}
		if err := validateWorkspaceTarget(source.Target); err != nil {
			return err
		}
		targets = append(targets, source.Target)
	}
	for index, target := range targets {
		for otherIndex, other := range targets {
			if index == otherIndex {
				continue
			}
			if target == other || target == "" || strings.HasPrefix(other, target+"/") {
				return invalidf("workspace source targets must be unique and non-overlapping")
			}
		}
	}
	if s.State != nil {
		if err := s.State.Artifact.ValidateExact(); err != nil {
			return err
		}
	}
	if s.Export != nil {
		if s.Mode != WorkspaceModeOverlay || s.Export.State == s.Export.Diff ||
			!idPattern.MatchString(s.Export.State) || !idPattern.MatchString(s.Export.Diff) {
			return invalidf("workspace export slots are invalid")
		}
	}
	return nil
}

func validateWorkspaceTarget(value string) error {
	if value == "" {
		return nil
	}
	if value != norm.NFC.String(value) || len([]byte(value)) > 1024 || strings.HasPrefix(value, "/") ||
		strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://") {
		return invalidf("workspace target is invalid")
	}
	parts := strings.Split(value, "/")
	if len(parts) > 32 {
		return invalidf("workspace target is invalid")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, "\r\n\t") {
			return invalidf("workspace target is invalid")
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return invalidf("workspace target is invalid")
			}
		}
	}
	first := parts[0]
	if len(first) >= 2 && ((first[0] >= 'A' && first[0] <= 'Z') || (first[0] >= 'a' && first[0] <= 'z')) && first[1] == ':' {
		return invalidf("workspace target is invalid")
	}
	return nil
}

type AgentRegistrationResponseV2 struct {
	APIVersion               string   `json:"apiVersion"`
	PrivateProtocolVersion   int      `json:"privateProtocolVersion"`
	RuntimeAgentID           string   `json:"runtimeAgentId"`
	Labels                   []string `json:"labels"`
	LabelRevision            uint64   `json:"labelRevision"`
	HeartbeatIntervalSeconds int      `json:"heartbeatIntervalSeconds"`
	ConfirmedLeaseSeconds    int      `json:"confirmedLeaseSeconds"`
}

func (r AgentRegistrationResponseV2) Validate() error {
	if err := validatePrivateProtocolVersion(r.PrivateProtocolVersion); err != nil {
		return err
	}
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

type TelemetrySettingsV2 struct {
	Adapter             RuntimeAdapterRef       `json:"adapter"`
	Endpoint            string                  `json:"endpoint"`
	Headers             map[string]SecretString `json:"headers"`
	CaptureContent      bool                    `json:"captureContent"`
	FlushTimeoutSeconds int                     `json:"flushTimeoutSeconds"`
}

func (s TelemetrySettingsV2) Validate() error {
	if s.Adapter != RuntimeAdapterOTLPHTTP {
		return invalidf("telemetry adapter must be otlp-http@1")
	}
	if err := validateRuntimeEndpoint("telemetry.endpoint", s.Endpoint); err != nil {
		return err
	}
	if s.Headers == nil || len(s.Headers) > 32 {
		return invalidf("telemetry headers must be a non-null map with at most 32 entries")
	}
	total := 0
	for name, value := range s.Headers {
		lower := strings.ToLower(name)
		if len(name) == 0 || len(name) > 64 || !headerNamePattern.MatchString(name) ||
			strings.ContainsAny(name, "\r\n") {
			return invalidf("telemetry header name is invalid")
		}
		if _, forbidden := forbiddenRuntimeHeaderNames[lower]; forbidden {
			return invalidf("telemetry header is forbidden")
		}
		secret := value.Reveal()
		if len(secret) == 0 || len(secret) > 4096 || strings.ContainsAny(secret, "\r\n") {
			return invalidf("telemetry header value is invalid")
		}
		total += len(secret)
	}
	if total > 16*1024 {
		return invalidf("telemetry header values exceed 16 KiB")
	}
	if s.CaptureContent {
		return invalidf("telemetry captureContent must be false")
	}
	if s.FlushTimeoutSeconds < 1 || s.FlushTimeoutSeconds > 10 {
		return invalidf("telemetry flushTimeoutSeconds must be from 1 through 10")
	}
	return nil
}

type HTTPProxyBasicAuthV2 struct {
	Username SecretString `json:"username"`
	Password SecretString `json:"password"`
}

type HTTPProxyTarget string

type HTTPProxySettingsV2 struct {
	Adapter     RuntimeAdapterRef     `json:"adapter"`
	ProxyURL    string                `json:"proxyUrl"`
	BasicAuth   *HTTPProxyBasicAuthV2 `json:"basicAuth,omitempty"`
	BearerToken *SecretString         `json:"bearerToken,omitempty"`
	CABundlePEM *string               `json:"caBundlePem,omitempty"`
	Targets     []HTTPProxyTarget     `json:"targets"`
}

func (s HTTPProxySettingsV2) Validate() error {
	if s.Adapter != RuntimeAdapterHTTPProxy {
		return invalidf("HTTP proxy adapter must be http-proxy@1")
	}
	if err := validateRuntimeEndpoint("httpProxy.proxyUrl", s.ProxyURL); err != nil {
		return err
	}
	if s.BasicAuth != nil && s.BearerToken != nil {
		return invalidf("HTTP proxy basicAuth and bearerToken are mutually exclusive")
	}
	if s.BasicAuth != nil {
		username, password := s.BasicAuth.Username.Reveal(), s.BasicAuth.Password.Reveal()
		if len(username) < 1 || len(username) > 256 || len(password) < 1 || len(password) > 8192 {
			return invalidf("HTTP proxy basicAuth is outside its size bound")
		}
	}
	if s.BearerToken != nil {
		value := s.BearerToken.Reveal()
		if len(value) < 1 || len(value) > 8192 {
			return invalidf("HTTP proxy bearerToken is outside its size bound")
		}
	}
	if s.CABundlePEM != nil {
		if err := validateCABundle(*s.CABundlePEM); err != nil {
			return err
		}
	}
	if len(s.Targets) == 0 || len(s.Targets) > 3 {
		return invalidf("HTTP proxy targets must be a non-empty subset")
	}
	previous := ""
	for _, target := range s.Targets {
		switch target {
		case ProxyTargetLLMGateway, ProxyTargetToolHTTP, ProxyTargetToolSubprocess:
		default:
			return invalidf("HTTP proxy target is invalid")
		}
		if string(target) <= previous {
			return invalidf("HTTP proxy targets must be sorted and unique")
		}
		previous = string(target)
	}
	return nil
}

type RuntimeSettingsV2 struct {
	LLMGatewayURL         string               `json:"llmGatewayUrl"`
	LLMGatewayToken       *SecretString        `json:"llmGatewayToken,omitempty"`
	ArtifactAPIURL        string               `json:"artifactApiUrl"`
	Telemetry             *TelemetrySettingsV2 `json:"telemetry,omitempty"`
	HTTPProxy             *HTTPProxySettingsV2 `json:"httpProxy,omitempty"`
	RequestTimeoutSeconds int                  `json:"requestTimeoutSeconds"`
}

// WorkerExecutionSettingsV2 is the in-process secret-bearing value delivered
// only after allocation provenance is durable.
type WorkerExecutionSettingsV2 struct {
	ModelPolicy                     ResolvedModelPolicy
	RuntimeSettings                 RuntimeSettingsV2
	ResolvedRuntimeConfigProvenance ResolvedRuntimeConfigProvenanceV2
}

func (s RuntimeSettingsV2) Validate() error {
	if err := validateRuntimeEndpoint("runtimeSettings.llmGatewayUrl", s.LLMGatewayURL); err != nil {
		return err
	}
	if err := validateURL("runtimeSettings.artifactApiUrl", s.ArtifactAPIURL); err != nil || len(s.ArtifactAPIURL) > 2048 {
		return invalidf("runtimeSettings.artifactApiUrl is invalid")
	}
	if s.RequestTimeoutSeconds <= 0 {
		return invalidf("runtimeSettings.requestTimeoutSeconds must be positive")
	}
	if s.Telemetry != nil {
		if err := s.Telemetry.Validate(); err != nil {
			return err
		}
	}
	if s.HTTPProxy != nil {
		if err := s.HTTPProxy.Validate(); err != nil {
			return err
		}
	}
	return nil
}

type RuntimeConfigRefV2 struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

func (r RuntimeConfigRefV2) Validate() error {
	if len(r.Name) == 0 || len(r.Name) > 63 || !idPattern.MatchString(r.Name) {
		return invalidf("RuntimeConfig ref name is invalid")
	}
	if len(r.Version) == 0 || len(r.Version) > 128 {
		return invalidf("RuntimeConfig ref version is invalid")
	}
	if err := validateSelector("RuntimeConfig ref", r.Name+"@"+r.Version); err != nil {
		return err
	}
	return validateDigest("RuntimeConfig ref digest", r.Digest)
}

type RuntimeLabelBindingProvenanceV2 struct {
	Label           string             `json:"label"`
	BindingRevision uint64             `json:"bindingRevision"`
	Config          RuntimeConfigRefV2 `json:"config"`
}

type RuntimeCredentialRefV2 struct {
	CredentialID string                `json:"credentialId"`
	Kind         RuntimeCredentialKind `json:"kind"`
}

type ResolvedRuntimeConfigProvenanceV2 struct {
	Default               RuntimeLabelBindingProvenanceV2   `json:"default"`
	RunLabels             []RuntimeLabelBindingProvenanceV2 `json:"runLabels"`
	AgentLabels           []RuntimeLabelBindingProvenanceV2 `json:"agentLabels"`
	RuntimeAdapters       []RuntimeAdapterRef               `json:"runtimeAdapters"`
	LLMGatewayConfig      *LLMGatewayConfigRef              `json:"llmGatewayConfig,omitempty"`
	LLMCredential         *LLMCredentialRef                 `json:"llmCredential,omitempty"`
	RuntimeCredentialRefs []RuntimeCredentialRefV2          `json:"runtimeCredentialRefs"`
}

func (p ResolvedRuntimeConfigProvenanceV2) Validate() error {
	if p.Default.Label != "default" || p.Default.BindingRevision == 0 {
		return invalidf("provenance default binding is invalid")
	}
	if err := p.Default.Config.Validate(); err != nil {
		return err
	}
	if err := validateProvenanceBindings("runLabels", p.RunLabels); err != nil {
		return err
	}
	if err := validateProvenanceBindings("agentLabels", p.AgentLabels); err != nil {
		return err
	}
	if err := validateSortedRuntimeAdapters(p.RuntimeAdapters); err != nil {
		return err
	}
	if p.LLMGatewayConfig != nil {
		if err := p.LLMGatewayConfig.ValidateRef(); err != nil {
			return err
		}
	}
	if p.LLMCredential != nil {
		if p.LLMGatewayConfig == nil {
			return invalidf("LLM credential provenance requires a Gateway config ref")
		}
		if err := p.LLMCredential.Validate(); err != nil {
			return err
		}
	}
	if p.RuntimeCredentialRefs == nil || len(p.RuntimeCredentialRefs) > 64 {
		return invalidf("Runtime credential provenance refs must be a non-null bounded array")
	}
	previous := ""
	for _, ref := range p.RuntimeCredentialRefs {
		if len(ref.CredentialID) == 0 || len(ref.CredentialID) > 128 || !idPattern.MatchString(ref.CredentialID) {
			return invalidf("Runtime credential provenance ID is invalid")
		}
		if err := ref.Kind.Validate(); err != nil {
			return err
		}
		key := string(ref.Kind) + "\x00" + ref.CredentialID
		if key <= previous {
			return invalidf("Runtime credential provenance refs must be sorted and unique")
		}
		previous = key
	}
	return nil
}

type AllocationSpecV2 struct {
	APIVersion                      string                            `json:"apiVersion"`
	AllocationID                    string                            `json:"allocationId"`
	RunID                           string                            `json:"runId"`
	StageExecutionID                string                            `json:"stageExecutionId"`
	LogicalAgentName                string                            `json:"logicalAgentName"`
	Namespace                       string                            `json:"namespace"`
	LeaseExpiresAt                  time.Time                         `json:"leaseExpiresAt"`
	AgentTemplate                   ResolvedAgentTemplate             `json:"agentTemplate"`
	ResolvedSkills                  []ResolvedSkill                   `json:"resolvedSkills"`
	ModelPolicy                     ResolvedModelPolicy               `json:"modelPolicy"`
	RuntimeSettings                 RuntimeSettingsV2                 `json:"runtimeSettings"`
	ResolvedRuntimeConfigProvenance ResolvedRuntimeConfigProvenanceV2 `json:"resolvedRuntimeConfigProvenance"`
	Workspace                       *AllocationWorkspaceSpecV2        `json:"workspace,omitempty"`
}

func (s AllocationSpecV2) Validate() error {
	if err := validateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	for field, value := range map[string]string{
		"allocationId": s.AllocationID, "runId": s.RunID,
		"stageExecutionId": s.StageExecutionID, "logicalAgentName": s.LogicalAgentName,
		"namespace": s.Namespace,
	} {
		if err := validateOpaqueID(field, value); err != nil {
			return err
		}
	}
	if strings.Contains(s.Namespace, "/") || s.LeaseExpiresAt.IsZero() {
		return invalidf("allocation namespace or lease is invalid")
	}
	if err := validateResolvedAgentTemplate(s.AgentTemplate); err != nil {
		return err
	}
	if err := ValidateResolvedSkills(s.AgentTemplate, s.ResolvedSkills); err != nil {
		return err
	}
	if err := validateWorkerModelPolicy(s.ModelPolicy, len(s.AgentTemplate.Toolsets) > 0 || len(s.AgentTemplate.Skills) > 0); err != nil {
		return err
	}
	if err := s.RuntimeSettings.Validate(); err != nil {
		return err
	}
	if s.Workspace != nil {
		if err := s.Workspace.Validate(); err != nil {
			return err
		}
	}
	return s.ResolvedRuntimeConfigProvenance.Validate()
}

type PrepareAllocationRequestV2 struct {
	APIVersion string           `json:"apiVersion"`
	Spec       AllocationSpecV2 `json:"spec"`
}

func (r PrepareAllocationRequestV2) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Spec.Validate()
}

type RuntimeAdapterMetricsV2 struct {
	Operations       uint64  `json:"operations"`
	FailedOperations uint64  `json:"failedOperations"`
	FlushAttempted   *bool   `json:"flushAttempted,omitempty"`
	FlushSucceeded   *bool   `json:"flushSucceeded,omitempty"`
	LastErrorCode    *string `json:"lastErrorCode,omitempty"`
}

func (m RuntimeAdapterMetricsV2) Validate() error {
	if m.FailedOperations > m.Operations {
		return invalidf("Runtime adapter failures exceed operations")
	}
	if (m.FlushAttempted == nil) != (m.FlushSucceeded == nil) {
		return invalidf("Runtime adapter flush fields must be present together")
	}
	if m.FlushAttempted != nil && !*m.FlushAttempted && *m.FlushSucceeded {
		return invalidf("Runtime adapter flush cannot succeed when not attempted")
	}
	if m.LastErrorCode != nil {
		if _, allowed := runtimeAdapterErrorCodes[*m.LastErrorCode]; !allowed {
			return invalidf("Runtime adapter lastErrorCode is invalid")
		}
	}
	return nil
}

type RuntimeReportV2 struct {
	Complete   bool                                          `json:"complete"`
	DurationMS *int64                                        `json:"durationMs,omitempty"`
	StopReason *string                                       `json:"stopReason,omitempty"`
	Adapters   map[RuntimeAdapterRef]RuntimeAdapterMetricsV2 `json:"adapters"`
}

func (r RuntimeReportV2) Validate() error {
	if r.DurationMS != nil && *r.DurationMS < 0 {
		return invalidf("runtime duration must be non-negative")
	}
	if r.StopReason != nil && strings.TrimSpace(*r.StopReason) == "" {
		return invalidf("runtime stopReason must not be empty")
	}
	if r.Adapters == nil || len(r.Adapters) > 64 {
		return invalidf("runtime adapter metrics map is invalid")
	}
	for ref, metrics := range r.Adapters {
		if err := ref.Validate(); err != nil {
			return err
		}
		if err := metrics.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// NormalizeAgentRegistrationV2 produces the immutable ordering used for
// registration retries and fingerprints. Decode still requires sorted wire
// sets so normalization cannot hide a malformed peer.
func NormalizeAgentRegistrationV2(source AgentRegistrationV2) AgentRegistrationV2 {
	result := source
	result.InitialLabels = append([]string{}, source.InitialLabels...)
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	if source.WorkspaceCapabilities != nil {
		capabilities := *source.WorkspaceCapabilities
		capabilities.Modes = append([]WorkspaceModeV2{}, source.WorkspaceCapabilities.Modes...)
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

// AgentRegistrationFingerprintV2 excludes mutable observation and startup
// labels. It is not the TLS principal fingerprint.
func AgentRegistrationFingerprintV2(source AgentRegistrationV2) (string, error) {
	normalized := NormalizeAgentRegistrationV2(source)
	normalized.InitialLabels = nil
	normalized.ObservedState = ""
	normalized.AllocationID = nil
	canonical, err := MarshalPrivateV2Canonical(normalized)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

// RuntimeAdapterCapabilityProjectionV2 returns a detached sorted safe view for
// Operations. It contains refs only, never settings or probe diagnostics.
func RuntimeAdapterCapabilityProjectionV2(source AgentRegistrationV2) []string {
	result := make([]string, len(source.SupportedRuntimeAdapters))
	for index, ref := range source.SupportedRuntimeAdapters {
		result[index] = string(ref)
	}
	sort.Strings(result)
	return result
}

// DecodePrivateV2Strict rejects duplicate keys, unknown fields and trailing
// JSON before returning a semantically validated private-v2 DTO.
func DecodePrivateV2Strict[T Validatable](data []byte) (T, error) {
	var value T
	if err := rejectDuplicateJSONKeys(data); err != nil {
		class := PrivateProtocolErrorSchema
		if errors.Is(err, errDuplicateJSONKey) {
			class = PrivateProtocolErrorDuplicate
		}
		return value, &PrivateProtocolError{Class: class}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, &PrivateProtocolError{Class: PrivateProtocolErrorSchema}
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return value, &PrivateProtocolError{Class: PrivateProtocolErrorSchema}
	}
	if err := value.Validate(); err != nil {
		class := PrivateProtocolErrorInvariant
		if errors.Is(err, privateVersionError) {
			class = PrivateProtocolErrorVersion
		}
		return value, &PrivateProtocolError{Class: class}
	}
	return value, nil
}

// MarshalPrivateV2Canonical uses RFC 8785 JCS for cross-language fixtures and
// fingerprints. It remains a private-wire encoder and therefore includes
// SecretString values; callers must never log its result.
func MarshalPrivateV2Canonical(value any) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode private protocol value: %w", err)
	}
	var jsonValue any
	if err := json.Unmarshal(encoded, &jsonValue); err != nil {
		return nil, fmt.Errorf("normalize private protocol value: %w", err)
	}
	canonical, err := jcs.Format(jsonValue)
	if err != nil {
		return nil, fmt.Errorf("canonicalize private protocol value: %w", err)
	}
	return []byte(canonical), nil
}

var errDuplicateJSONKey = errors.New("duplicate JSON key")

func rejectDuplicateJSONKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := scanJSONValue(decoder); err != nil {
		return err
	}
	if token, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err != nil {
			return err
		}
		return fmt.Errorf("unexpected trailing JSON token %v", token)
	}
	return nil
}

func scanJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, composite := token.(json.Delim)
	if !composite {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("JSON object key is not a string")
			}
			if _, duplicate := seen[key]; duplicate {
				return errDuplicateJSONKey
			}
			seen[key] = struct{}{}
			if err := scanJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim('}') {
			return errors.New("invalid JSON object")
		}
	case '[':
		for decoder.More() {
			if err := scanJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim(']') {
			return errors.New("invalid JSON array")
		}
	default:
		return errors.New("invalid JSON delimiter")
	}
	return nil
}

func validatePrivateProtocolVersion(value int) error {
	if value != PrivateProtocolVersionV2 {
		return fmt.Errorf("%w: expected version 2", privateVersionError)
	}
	return nil
}

func validateSortedLabels(field string, values []string, maximum int, allowDefault bool) error {
	if values == nil || len(values) > maximum {
		return invalidf("%s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value) == 0 || len(value) > 63 || !idPattern.MatchString(value) ||
			(!allowDefault && value == "default") {
			return invalidf("%s contains an invalid label", field)
		}
		if value <= previous {
			return invalidf("%s must be sorted and unique", field)
		}
		previous = value
	}
	return nil
}

func validateSortedRuntimeAdapters(values []RuntimeAdapterRef) error {
	if values == nil || len(values) > 64 {
		return invalidf("RuntimeAdapter refs must be a non-null bounded array")
	}
	previous := ""
	for _, value := range values {
		if err := value.Validate(); err != nil {
			return err
		}
		if string(value) <= previous {
			return invalidf("RuntimeAdapter refs must be sorted and unique")
		}
		previous = string(value)
	}
	return nil
}

func validateRuntimeEndpoint(field, value string) error {
	if len(value) == 0 || len([]byte(value)) > 2048 || value != strings.TrimSpace(value) {
		return invalidf("%s is invalid", field)
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.Fragment != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return invalidf("%s must be an absolute HTTP(S) URL without userinfo, query, or fragment", field)
	}
	return nil
}

func validateCABundle(value string) error {
	if len(value) == 0 || len([]byte(value)) > 64*1024 || strings.Contains(value, "PRIVATE KEY") {
		return invalidf("HTTP proxy CA bundle is invalid")
	}
	rest := []byte(value)
	count := 0
	for len(bytes.TrimSpace(rest)) > 0 {
		block, remaining := pem.Decode(rest)
		if block == nil || block.Type != "CERTIFICATE" {
			return invalidf("HTTP proxy CA bundle is invalid")
		}
		if _, err := x509.ParseCertificate(block.Bytes); err != nil {
			return invalidf("HTTP proxy CA bundle is invalid")
		}
		count++
		if count > 8 {
			return invalidf("HTTP proxy CA bundle contains too many certificates")
		}
		rest = remaining
	}
	if count == 0 {
		return invalidf("HTTP proxy CA bundle must contain a certificate")
	}
	return nil
}

func validateProvenanceBindings(field string, values []RuntimeLabelBindingProvenanceV2) error {
	if values == nil || len(values) > 32 {
		return invalidf("provenance %s must be a non-null bounded array", field)
	}
	previous := ""
	for _, value := range values {
		if len(value.Label) == 0 || len(value.Label) > 63 || value.Label == "default" ||
			!idPattern.MatchString(value.Label) || value.Label <= previous || value.BindingRevision == 0 {
			return invalidf("provenance %s contains invalid or unsorted bindings", field)
		}
		if err := value.Config.Validate(); err != nil {
			return err
		}
		previous = value.Label
	}
	return nil
}
