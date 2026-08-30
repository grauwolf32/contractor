package config

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func resolveLLMGatewayConfig(
	selector Selector, spec *llmGatewayConfigSpecSource,
) (contracts.ResolvedLLMGatewayConfig, error) {
	if spec == nil {
		return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf("spec is required")
	}
	if spec.Protocol != contracts.OpenAICompatibleProtocol {
		return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf(
			"spec.protocol must be %q", contracts.OpenAICompatibleProtocol,
		)
	}
	inferenceURL, err := normalizeInferenceGatewayURL(spec.URL)
	if err != nil {
		return contracts.ResolvedLLMGatewayConfig{}, err
	}
	gateway := contracts.ResolvedLLMGatewayConfig{
		Ref: contracts.LLMGatewayConfigRef{
			GatewayID: selector.ID, Version: selector.Version,
		},
		Protocol: spec.Protocol,
		URL:      inferenceURL,
	}
	if spec.CredentialManager != nil {
		if spec.CredentialManager.Implementation != contracts.LiteLLMVirtualKeysManager {
			return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf(
				"spec.credentialManager.implementation must be %q",
				contracts.LiteLLMVirtualKeysManager,
			)
		}
		managementURL, normalizeErr := normalizeManagementGatewayURL(
			spec.CredentialManager.ManagementURL,
		)
		if normalizeErr != nil {
			return contracts.ResolvedLLMGatewayConfig{}, normalizeErr
		}
		gateway.CredentialManager = &contracts.LLMGatewayCredentialManager{
			Implementation: spec.CredentialManager.Implementation,
			ManagementURL:  managementURL,
		}
	}
	digest, err := llmGatewayConfigDigest(selector, gateway)
	if err != nil {
		return contracts.ResolvedLLMGatewayConfig{}, err
	}
	gateway.Ref.Digest = digest
	if err := gateway.Validate(); err != nil {
		return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf("resolved Gateway is invalid: %w", err)
	}
	return gateway, nil
}

func normalizeInferenceGatewayURL(raw string) (string, error) {
	parsed, err := parseConfiguredGatewayURL(raw)
	if err != nil {
		return "", fmt.Errorf("spec.url must be an absolute HTTP(S) URL without userinfo, query, or fragment")
	}
	if parsed.Path == "" || !strings.HasPrefix(parsed.Path, "/") {
		return "", fmt.Errorf("spec.url must include an explicit absolute inference path")
	}
	return parsed.String(), nil
}

func normalizeManagementGatewayURL(raw string) (string, error) {
	parsed, err := parseConfiguredGatewayURL(raw)
	if err != nil || parsed.Path != "" && parsed.Path != "/" {
		return "", fmt.Errorf("spec.credentialManager.managementUrl must be an HTTP(S) origin with a root path")
	}
	if parsed.Scheme == "http" {
		ip := net.ParseIP(parsed.Hostname())
		if ip == nil || !ip.IsLoopback() {
			return "", fmt.Errorf("spec.credentialManager.managementUrl permits HTTP only for a loopback IP origin")
		}
	}
	return (&url.URL{Scheme: parsed.Scheme, Host: parsed.Host}).String(), nil
}

func parseConfiguredGatewayURL(raw string) (*url.URL, error) {
	if raw == "" || raw != strings.TrimSpace(raw) || strings.Contains(raw, "#") {
		return nil, fmt.Errorf("Gateway URL is invalid")
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Opaque != "" || parsed.Host == "" ||
		parsed.Hostname() == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, fmt.Errorf("Gateway URL is invalid")
	}
	return parsed, nil
}
