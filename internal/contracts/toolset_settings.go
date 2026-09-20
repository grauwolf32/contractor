package contracts

import (
	"bytes"
	"encoding/json"
	"strings"
)

const (
	MCPToolsetRef                    = "mcp@1"
	MaximumConfiguredToolsets        = 32
	DefaultMCPConnectTimeoutSeconds  = 10
	DefaultMCPRequestTimeoutSeconds  = 60
	MaximumAllocationDiscoveredTools = 64
)

func ValidateAllocationToolNames(names []string) error {
	if len(names) == 0 || len(names) > MaximumAllocationDiscoveredTools {
		return invalidf("invalid allocation tool selection size")
	}
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if !toolArgumentName.MatchString(name) {
			return invalidf("invalid allocation tool name")
		}
		if _, exists := seen[name]; exists {
			return invalidf("duplicate allocation tool name")
		}
		seen[name] = struct{}{}
	}
	return nil
}

// ToolsetRuntimeConfiguration is a typed, materialized allocation value. New
// exact refs must register a decoder here and in the corresponding Runtime
// factory. Arbitrary JSON is never passed through to a tool implementation.
type ToolsetRuntimeConfiguration interface {
	Validate() error
	CloneToolsetSettings() ToolsetRuntimeConfiguration
}

type ToolsetRuntimeSettings map[string]ToolsetRuntimeConfiguration

func (s *ToolsetRuntimeSettings) UnmarshalJSON(raw []byte) error {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return invalidf("toolsets must be an object")
	}
	var values map[string]json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		return invalidf("toolsets must be an object")
	}
	result := make(ToolsetRuntimeSettings, len(values))
	for ref, value := range values {
		switch ref {
		case MCPToolsetRef:
			parsed, err := DecodeStrict[MCPToolsetSettings](value)
			if err != nil {
				return invalidf("invalid MCP toolset settings")
			}
			result[ref] = parsed
		default:
			return invalidf("unregistered toolset configuration")
		}
	}
	if err := result.Validate(); err != nil {
		return err
	}
	*s = result
	return nil
}

func (s ToolsetRuntimeSettings) Validate() error {
	if len(s) > MaximumConfiguredToolsets {
		return invalidf("too many toolset configurations")
	}
	for ref, value := range s {
		if ref != MCPToolsetRef {
			return invalidf("unregistered toolset configuration")
		}
		if _, ok := value.(MCPToolsetSettings); !ok {
			return invalidf("toolset configuration type mismatch")
		}
		if err := value.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (s ToolsetRuntimeSettings) Clone() ToolsetRuntimeSettings {
	if s == nil {
		return nil
	}
	result := make(ToolsetRuntimeSettings, len(s))
	for ref, value := range s {
		result[ref] = value.CloneToolsetSettings()
	}
	return result
}

// MCPToolsetSettings exists only on the private allocation wire. Credentials
// remain references everywhere else; these headers use the existing redaction
// primitive and must never be persisted as configuration/provenance.
type MCPToolsetSettings struct {
	Endpoint              string                  `json:"endpoint"`
	Headers               map[string]SecretString `json:"headers,omitempty"`
	CABundlePEM           *string                 `json:"caBundlePem,omitempty"`
	ConnectTimeoutSeconds int                     `json:"connectTimeoutSeconds"`
	RequestTimeoutSeconds int                     `json:"requestTimeoutSeconds"`
}

func (s MCPToolsetSettings) Validate() error {
	if err := ValidateMCPEndpoint(s.Endpoint); err != nil {
		return err
	}
	if s.ConnectTimeoutSeconds < 1 || s.ConnectTimeoutSeconds > 60 ||
		s.RequestTimeoutSeconds < 1 || s.RequestTimeoutSeconds > 600 {
		return invalidf("MCP timeouts are outside their bounds")
	}
	if s.CABundlePEM != nil {
		if err := validateCABundle("MCP", *s.CABundlePEM); err != nil {
			return err
		}
	}
	return ValidateMCPHeaders(s.Headers)
}

func ValidateMCPEndpoint(endpoint string) error {
	if strings.ContainsAny(endpoint, "?#") {
		return invalidf("invalid MCP endpoint")
	}
	return validateRuntimeEndpoint("MCP endpoint", endpoint)
}

func ValidateMCPHeaders(headers map[string]SecretString) error {
	if len(headers) > 32 {
		return invalidf("too many MCP headers")
	}
	seen := make(map[string]struct{}, len(headers))
	total := 0
	for name, secret := range headers {
		lower := strings.ToLower(name)
		if len(name) < 1 || len(name) > 64 || !headerNamePattern.MatchString(name) {
			return invalidf("invalid MCP header name")
		}
		if _, exists := seen[lower]; exists {
			return invalidf("duplicate MCP header name")
		}
		seen[lower] = struct{}{}
		if _, forbidden := forbiddenRuntimeHeaderNames[lower]; forbidden {
			return invalidf("reserved MCP header")
		}
		switch lower {
		case "accept", "content-type", "mcp-session-id", "mcp-protocol-version", "last-event-id":
			return invalidf("reserved MCP header")
		}
		value := secret.Reveal()
		if len(value) < 1 || len(value) > 4096 || strings.ContainsAny(value, "\r\n") {
			return invalidf("invalid MCP header value")
		}
		total += len(name) + len(value)
	}
	if total > 16*1024 {
		return invalidf("MCP headers exceed size bound")
	}
	return nil
}

func (s MCPToolsetSettings) CloneToolsetSettings() ToolsetRuntimeConfiguration {
	result := s
	if s.Headers != nil {
		result.Headers = make(map[string]SecretString, len(s.Headers))
		for name, value := range s.Headers {
			result.Headers[name] = value
		}
	}
	if s.CABundlePEM != nil {
		value := *s.CABundlePEM
		result.CABundlePEM = &value
	}
	return result
}
