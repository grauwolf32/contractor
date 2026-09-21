package contracts

import (
	"net"
	"net/url"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
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

func (r LLMGatewayConfigRef) ValidateRef() error {
	if err := validateSelector("llmGatewayConfigRef", r.GatewayID+"@"+r.Version); err != nil {
		return err
	}
	return validateDigest("llmGatewayConfigRef.digest", r.Digest)
}

// LLMCredentialRef is deliberately non-secret. The token is resolved only at
// the final model-client or allocation-preparation boundary.
type LLMCredentialRef struct {
	CredentialID string `json:"credentialId"`
}

func (r LLMCredentialRef) Validate() error {
	return validateSelector("llmCredentialRef", r.CredentialID+"@1")
}

type LLMGatewayCredentialManager struct {
	Implementation string `json:"implementation"`
	ManagementURL  string `json:"managementUrl"`
}

// GatewayFailureSignature names one exact provider response that means the
// model is temporarily unavailable although the HTTP status alone would read as
// a permanent request rejection. Only exact matches are supported so that an
// arbitrary 4xx carrying request-specific text can never become a transient
// availability failure.
type GatewayFailureSignature struct {
	Status int `json:"status"`
	// MessageEquals matches the whole error message (or bare error string).
	MessageEquals string `json:"messageEquals,omitempty"`
	// LiteLLMWrapped matches the same upstream message inside LiteLLM's
	// observed BadRequest wrapper, with or without its model-group suffix.
	LiteLLMWrapped string `json:"litellmWrapped,omitempty"`
}

// GatewayFailureSignatures is the provider-specific part of failure
// classification; status-based rules belong to the protocol and stay in code.
type GatewayFailureSignatures struct {
	ModelUnavailable []GatewayFailureSignature `json:"modelUnavailable,omitempty"`
	// PermanentCodes are provider error codes that fail the invocation without
	// blocking the route, whatever HTTP status or retry hint accompanies them.
	PermanentCodes []string `json:"permanentCodes,omitempty"`
}

const (
	MaximumGatewayFailureSignatures    = 32
	MaximumGatewayFailureSignatureText = 512
)

var (
	// gatewayFailureSignatureStatuses are the statuses a provider is known to
	// misuse for availability; retryable statuses are classified without help.
	gatewayFailureSignatureStatuses = map[int]struct{}{400: {}, 404: {}, 409: {}, 422: {}}
	gatewayFailureCodePattern       = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
)

// DefaultGatewayFailureSignatures is the openai-compatible@1 baseline used when
// a Gateway declares nothing: the LM Studio unload responses observed through
// LiteLLM during the 2026-09-20 outage.
func DefaultGatewayFailureSignatures() GatewayFailureSignatures {
	return GatewayFailureSignatures{
		ModelUnavailable: []GatewayFailureSignature{
			{Status: 400, MessageEquals: "Model is unloaded."},
			{Status: 400, MessageEquals: "Model unloaded by user or API request."},
			{Status: 400, LiteLLMWrapped: "Model is unloaded."},
			{Status: 400, LiteLLMWrapped: "Model unloaded by user or API request."},
		},
		PermanentCodes: []string{"insufficient_quota", "budget_exceeded", "context_length_exceeded"},
	}
}

func (s GatewayFailureSignatures) Validate() error {
	if len(s.ModelUnavailable) > MaximumGatewayFailureSignatures || len(s.PermanentCodes) > MaximumGatewayFailureSignatures {
		return invalidf("llmGatewayConfig.failureSignatures lists at most %d entries per kind", MaximumGatewayFailureSignatures)
	}
	seen := make(map[GatewayFailureSignature]struct{}, len(s.ModelUnavailable))
	for _, signature := range s.ModelUnavailable {
		if err := signature.validate(); err != nil {
			return err
		}
		if _, duplicate := seen[signature]; duplicate {
			return invalidf("llmGatewayConfig.failureSignatures.modelUnavailable repeats a signature")
		}
		seen[signature] = struct{}{}
	}
	codes := make(map[string]struct{}, len(s.PermanentCodes))
	for _, code := range s.PermanentCodes {
		if !gatewayFailureCodePattern.MatchString(code) {
			return invalidf("llmGatewayConfig.failureSignatures.permanentCodes must be snake_case provider codes")
		}
		if _, duplicate := codes[code]; duplicate {
			return invalidf("llmGatewayConfig.failureSignatures.permanentCodes repeats %q", code)
		}
		codes[code] = struct{}{}
	}
	return nil
}

func (s GatewayFailureSignature) validate() error {
	if _, ok := gatewayFailureSignatureStatuses[s.Status]; !ok {
		return invalidf("llmGatewayConfig.failureSignatures.modelUnavailable status must be 400, 404, 409, or 422")
	}
	if (s.MessageEquals == "") == (s.LiteLLMWrapped == "") {
		return invalidf("llmGatewayConfig.failureSignatures.modelUnavailable needs exactly one of messageEquals or litellmWrapped")
	}
	text := s.MessageEquals + s.LiteLLMWrapped
	if len(text) > MaximumGatewayFailureSignatureText || strings.TrimSpace(text) != text ||
		!utf8.ValidString(text) || strings.ContainsFunc(text, unicode.IsControl) {
		return invalidf("llmGatewayConfig.failureSignatures.modelUnavailable text must be trimmed UTF-8 of at most %d bytes without control characters", MaximumGatewayFailureSignatureText)
	}
	return nil
}

// ResolvedLLMGatewayConfig is a complete immutable non-secret Gateway body.
// Credentials are selected beside this value and never embedded in it.
type ResolvedLLMGatewayConfig struct {
	Ref               LLMGatewayConfigRef          `json:"ref"`
	Protocol          string                       `json:"protocol"`
	URL               string                       `json:"url"`
	CredentialManager *LLMGatewayCredentialManager `json:"credentialManager,omitempty"`
	// FailureSignatures is present only when the Gateway declares its own set,
	// which keeps the digest of existing declarations stable; readers use
	// EffectiveFailureSignatures.
	FailureSignatures *GatewayFailureSignatures `json:"failureSignatures,omitempty"`
}

// EffectiveFailureSignatures returns the declared set or the protocol default.
func (c ResolvedLLMGatewayConfig) EffectiveFailureSignatures() GatewayFailureSignatures {
	if c.FailureSignatures != nil {
		return cloneGatewayFailureSignatures(*c.FailureSignatures)
	}
	return DefaultGatewayFailureSignatures()
}

func cloneGatewayFailureSignatures(source GatewayFailureSignatures) GatewayFailureSignatures {
	return GatewayFailureSignatures{
		ModelUnavailable: append([]GatewayFailureSignature(nil), source.ModelUnavailable...),
		PermanentCodes:   append([]string(nil), source.PermanentCodes...),
	}
}

func (c ResolvedLLMGatewayConfig) Validate() error {
	if err := c.Ref.ValidateRef(); err != nil {
		return err
	}
	if c.Protocol != OpenAICompatibleProtocol {
		return invalidf("llmGatewayConfig.protocol must be %q", OpenAICompatibleProtocol)
	}
	if err := validateInferenceGatewayURL(c.URL); err != nil {
		return err
	}
	if c.FailureSignatures != nil {
		if err := c.FailureSignatures.Validate(); err != nil {
			return err
		}
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
