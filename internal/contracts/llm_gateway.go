package contracts

import (
	"net"
	"net/url"
	"strings"
)

const (
	OpenAICompatibleProtocol  = "openai-compatible@1"
	LiteLLMVirtualKeysManager = "litellm-virtual-keys@1"
)

type LLMGatewayConfigRef struct {
	GatewayID string `json:"gatewayId"`
	Version   string `json:"version"`
	Digest    string `json:"digest"`
}

type LLMGatewayCredentialManager struct {
	Implementation string `json:"implementation"`
	ManagementURL  string `json:"managementUrl"`
}

// ResolvedLLMGatewayConfig is a complete immutable non-secret Gateway body.
// Credentials are selected beside this value and never embedded in it.
type ResolvedLLMGatewayConfig struct {
	Ref               LLMGatewayConfigRef          `json:"ref"`
	Protocol          string                       `json:"protocol"`
	URL               string                       `json:"url"`
	CredentialManager *LLMGatewayCredentialManager `json:"credentialManager,omitempty"`
}

func (c ResolvedLLMGatewayConfig) Validate() error {
	if err := validateSelector("llmGatewayConfigRef", c.Ref.GatewayID+"@"+c.Ref.Version); err != nil {
		return err
	}
	if err := validateDigest("llmGatewayConfigRef.digest", c.Ref.Digest); err != nil {
		return err
	}
	if c.Protocol != OpenAICompatibleProtocol {
		return invalidf("llmGatewayConfig.protocol must be %q", OpenAICompatibleProtocol)
	}
	if err := validateInferenceGatewayURL(c.URL); err != nil {
		return err
	}
	if c.CredentialManager == nil {
		return nil
	}
	if c.CredentialManager.Implementation != LiteLLMVirtualKeysManager {
		return invalidf(
			"llmGatewayConfig.credentialManager.implementation must be %q",
			LiteLLMVirtualKeysManager,
		)
	}
	return validateCanonicalManagementURL(c.CredentialManager.ManagementURL)
}

func validateInferenceGatewayURL(raw string) error {
	parsed, err := parseGatewayURL(raw)
	if err != nil {
		return invalidf("llmGatewayConfig.url must be an absolute HTTP(S) URL without userinfo, query, or fragment")
	}
	if parsed.Path == "" || !strings.HasPrefix(parsed.Path, "/") {
		return invalidf("llmGatewayConfig.url must include an explicit absolute path")
	}
	return nil
}

func validateCanonicalManagementURL(raw string) error {
	parsed, err := parseGatewayURL(raw)
	if err != nil || parsed.Path != "" {
		return invalidf("llmGatewayConfig credential management URL must be a canonical HTTP(S) origin")
	}
	if parsed.Scheme == "https" {
		return nil
	}
	ip := net.ParseIP(parsed.Hostname())
	if ip == nil || !ip.IsLoopback() {
		return invalidf("HTTP credential management URL is allowed only for a loopback IP origin")
	}
	return nil
}

func parseGatewayURL(raw string) (*url.URL, error) {
	if raw == "" || raw != strings.TrimSpace(raw) || strings.Contains(raw, "#") {
		return nil, invalidf("Gateway URL is invalid")
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.IsAbs() == false || parsed.Opaque != "" || parsed.Host == "" ||
		parsed.Hostname() == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, invalidf("Gateway URL is invalid")
	}
	return parsed, nil
}
