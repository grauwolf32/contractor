package contracts

// Secret-bearing Runtime settings delivered with an allocation: telemetry,
// HTTP proxy, Caido, HTTP origin target, and the resolved RuntimeConfig
// provenance they are pinned by.
// Mirrors the Python runtime's contracts/settings.py.

import (
	"regexp"
	"strings"
)

var headerNamePattern = regexp.MustCompile(`^[!#$%&'*+\-.^_` + "`" + `|~0-9A-Za-z]+$`)

var forbiddenRuntimeHeaderNames = map[string]struct{}{
	"connection": {}, "content-length": {}, "host": {}, "keep-alive": {},
	"proxy-authenticate": {}, "proxy-authorization": {}, "proxy-connection": {},
	"te": {}, "trailer": {}, "transfer-encoding": {}, "upgrade": {},
}

const (
	ProxyTargetLLMGateway     HTTPProxyTarget = "llm-gateway"
	ProxyTargetToolHTTP       HTTPProxyTarget = "tool-http"
	ProxyTargetToolSubprocess HTTPProxyTarget = "tool-subprocess"
)

type TelemetrySettings struct {
	Adapter             RuntimeAdapterRef        `json:"adapter"`
	Endpoint            string                   `json:"endpoint"`
	Headers             map[string]SecretString  `json:"headers"`
	CaptureContent      bool                     `json:"captureContent"`
	FlushTimeoutSeconds int                      `json:"flushTimeoutSeconds"`
	Export              *TelemetryExportSettings `json:"export,omitempty"`
}

func (s TelemetrySettings) Validate() error {
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
	if s.FlushTimeoutSeconds < 1 || s.FlushTimeoutSeconds > 10 {
		return invalidf("telemetry flushTimeoutSeconds must be from 1 through 10")
	}
	if s.Export != nil {
		return s.Export.Validate()
	}
	return nil
}

type HTTPProxyBasicAuth struct {
	Username SecretString `json:"username"`
	Password SecretString `json:"password"`
}

type HTTPProxyTarget string

type HTTPProxySettings struct {
	Adapter     RuntimeAdapterRef   `json:"adapter"`
	ProxyURL    string              `json:"proxyUrl"`
	BasicAuth   *HTTPProxyBasicAuth `json:"basicAuth,omitempty"`
	BearerToken *SecretString       `json:"bearerToken,omitempty"`
	CABundlePEM *string             `json:"caBundlePem,omitempty"`
	Targets     []HTTPProxyTarget   `json:"targets"`
}

// MaxCaidoRequestTimeoutSeconds bounds the Caido GraphQL request contract.
const MaxCaidoRequestTimeoutSeconds = 120

type CaidoSettings struct {
	Adapter               RuntimeAdapterRef `json:"adapter"`
	Endpoint              string            `json:"endpoint"`
	BearerToken           *SecretString     `json:"bearerToken,omitempty"`
	CABundlePEM           *string           `json:"caBundlePem,omitempty"`
	RequestTimeoutSeconds int               `json:"requestTimeoutSeconds"`
}

func (s CaidoSettings) Validate() error {
	if s.Adapter != RuntimeAdapterCaidoGraphQL {
		return invalidf("Caido adapter must be caido-graphql@1")
	}
	if err := validateRuntimeEndpoint("caido.endpoint", s.Endpoint); err != nil {
		return err
	}
	if s.BearerToken != nil {
		value := s.BearerToken.Reveal()
		if len(value) < 1 || len([]byte(value)) > 8192 {
			return invalidf("Caido bearerToken is outside its size bound")
		}
	}
	if s.CABundlePEM != nil {
		if err := validateCABundle("Caido", *s.CABundlePEM); err != nil {
			return err
		}
	}
	if s.RequestTimeoutSeconds < 1 || s.RequestTimeoutSeconds > MaxCaidoRequestTimeoutSeconds {
		return invalidf("Caido requestTimeoutSeconds must be from 1 through 120")
	}
	return nil
}

func (s HTTPProxySettings) Validate() error {
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
		if err := validateCABundle("HTTP proxy", *s.CABundlePEM); err != nil {
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

type RuntimeSettings struct {
	LLMRecovery     bool          `json:"llmRecovery,omitempty"`
	LLMGatewayURL   string        `json:"llmGatewayUrl,omitempty"`
	LLMGatewayToken *SecretString `json:"llmGatewayToken,omitempty"`
	// LLMGatewayFailureSignatures carries the selected Gateway's declared set;
	// absent means the protocol default, exactly as on the Gateway body.
	LLMGatewayFailureSignatures *GatewayFailureSignatures `json:"llmGatewayFailureSignatures,omitempty"`
	ArtifactAPIURL              string                    `json:"artifactApiUrl"`
	Telemetry                   *TelemetrySettings        `json:"telemetry,omitempty"`
	HTTPProxy                   *HTTPProxySettings        `json:"httpProxy,omitempty"`
	Caido                       *CaidoSettings            `json:"caido,omitempty"`
	HTTPOriginTarget            *HTTPOriginTargetSettings `json:"httpOriginTarget,omitempty"`
	RequestTimeoutSeconds       int                       `json:"requestTimeoutSeconds"`
}

func (s RuntimeSettings) Validate() error {
	if s.LLMGatewayURL != "" {
		if err := validateRuntimeEndpoint("runtimeSettings.llmGatewayUrl", s.LLMGatewayURL); err != nil {
			return err
		}
	} else if s.LLMGatewayToken != nil || s.LLMGatewayFailureSignatures != nil {
		return invalidf("LLM token and failure signatures require a Gateway URL")
	}
	if s.LLMGatewayFailureSignatures != nil {
		if err := s.LLMGatewayFailureSignatures.Validate(); err != nil {
			return err
		}
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
	if s.Caido != nil {
		if err := s.Caido.Validate(); err != nil {
			return err
		}
	}
	if s.HTTPOriginTarget != nil {
		if err := s.HTTPOriginTarget.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// HTTPOriginTargetRef is the immutable, non-secret target provenance pinned
// by a Project Run. URL may contain an application path, while Authorization
// is scoped by Runtime to its exact normalized origin.
type HTTPOriginTargetRef struct {
	URL        string                `json:"url"`
	Credential *RuntimeCredentialRef `json:"credential,omitempty"`
}

func (t HTTPOriginTargetRef) Validate() error {
	if err := validateRuntimeEndpoint("httpOriginTarget.url", t.URL); err != nil {
		return err
	}
	if t.Credential == nil {
		return nil
	}
	if len(t.Credential.CredentialID) == 0 || len(t.Credential.CredentialID) > 128 ||
		!idPattern.MatchString(t.Credential.CredentialID) {
		return invalidf("HTTP origin target credential ID is invalid")
	}
	if t.Credential.Kind != RuntimeCredentialOriginBasic && t.Credential.Kind != RuntimeCredentialOriginBearer {
		return invalidf("HTTP origin target credential kind is invalid")
	}
	return nil
}

// HTTPOriginTargetSettings exists only in the allocation transport. Its
// optional secret members are mutually exclusive and are never persisted.
type HTTPOriginTargetSettings struct {
	URL         string              `json:"url"`
	BasicAuth   *HTTPProxyBasicAuth `json:"basicAuth,omitempty"`
	BearerToken *SecretString       `json:"bearerToken,omitempty"`
}

func (s HTTPOriginTargetSettings) Validate() error {
	if err := validateRuntimeEndpoint("httpOriginTarget.url", s.URL); err != nil {
		return err
	}
	if s.BasicAuth != nil && s.BearerToken != nil {
		return invalidf("HTTP origin target basicAuth and bearerToken are mutually exclusive")
	}
	if s.BasicAuth != nil {
		username, password := s.BasicAuth.Username.Reveal(), s.BasicAuth.Password.Reveal()
		if len(username) < 1 || len(username) > 256 || strings.Contains(username, ":") ||
			len(password) < 1 || len(password) > 8192 {
			return invalidf("HTTP origin target basicAuth is outside its size bound")
		}
	}
	if s.BearerToken != nil {
		value := s.BearerToken.Reveal()
		if len(value) < 1 || len(value) > 8192 {
			return invalidf("HTTP origin target bearerToken is outside its size bound")
		}
	}
	return nil
}

// WorkerExecutionSettings is the in-process secret-bearing value delivered
// only after allocation provenance is durable.
type WorkerExecutionSettings struct {
	ModelPolicy                     ResolvedModelPolicy
	RuntimeSettings                 RuntimeSettings
	ResolvedRuntimeConfigProvenance ResolvedRuntimeConfigProvenance
}

type RuntimeConfigRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

func (r RuntimeConfigRef) Validate() error {
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

type RuntimeLabelBindingProvenance struct {
	Label           string           `json:"label"`
	BindingRevision uint64           `json:"bindingRevision"`
	Config          RuntimeConfigRef `json:"config"`
}

type RuntimeCredentialRef struct {
	CredentialID string                `json:"credentialId"`
	Kind         RuntimeCredentialKind `json:"kind"`
}

type ResolvedRuntimeConfigProvenance struct {
	Default               RuntimeLabelBindingProvenance   `json:"default"`
	RunLabels             []RuntimeLabelBindingProvenance `json:"runLabels"`
	AgentLabels           []RuntimeLabelBindingProvenance `json:"agentLabels"`
	RuntimeAdapters       []RuntimeAdapterRef             `json:"runtimeAdapters"`
	LLMGatewayConfig      *LLMGatewayConfigRef            `json:"llmGatewayConfig,omitempty"`
	LLMCredential         *LLMCredentialRef               `json:"llmCredential,omitempty"`
	RuntimeCredentialRefs []RuntimeCredentialRef          `json:"runtimeCredentialRefs"`
}

func (p ResolvedRuntimeConfigProvenance) Validate() error {
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
