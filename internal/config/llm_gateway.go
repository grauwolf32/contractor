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
	if spec.FailureSignatures != nil {
		gateway.FailureSignatures = failureSignaturesFromSource(*spec.FailureSignatures)
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

func failureSignaturesFromSource(source llmGatewayFailureSignaturesSource) *contracts.GatewayFailureSignatures {
	result := &contracts.GatewayFailureSignatures{
		ModelUnavailable: make([]contracts.GatewayFailureSignature, 0, len(source.ModelUnavailable)),
		PermanentCodes:   append([]string(nil), source.PermanentCodes...),
	}
	for _, signature := range source.ModelUnavailable {
		result.ModelUnavailable = append(result.ModelUnavailable, contracts.GatewayFailureSignature{
			Status: signature.Status, MessageEquals: signature.MessageEquals, LiteLLMWrapped: signature.LiteLLMWrapped,
		})
	}
	return result
}

func failureSignaturesToSource(signatures contracts.GatewayFailureSignatures) *llmGatewayFailureSignaturesSource {
	result := &llmGatewayFailureSignaturesSource{
		ModelUnavailable: make([]llmGatewayFailureSignatureSource, 0, len(signatures.ModelUnavailable)),
		PermanentCodes:   append([]string(nil), signatures.PermanentCodes...),
	}
	for _, signature := range signatures.ModelUnavailable {
		result.ModelUnavailable = append(result.ModelUnavailable, llmGatewayFailureSignatureSource{
			Status: signature.Status, MessageEquals: signature.MessageEquals, LiteLLMWrapped: signature.LiteLLMWrapped,
		})
	}
	return result
}

// failureSignaturesDocument is the canonical JSON shape shared by the digest
// and the public resource body; omitted fields never appear.
func failureSignaturesDocument(signatures contracts.GatewayFailureSignatures) map[string]any {
	result := map[string]any{}
	if len(signatures.ModelUnavailable) != 0 {
		entries := make([]any, 0, len(signatures.ModelUnavailable))
		for _, signature := range signatures.ModelUnavailable {
			entry := map[string]any{"status": signature.Status}
			if signature.MessageEquals != "" {
				entry["messageEquals"] = signature.MessageEquals
			}
			if signature.LiteLLMWrapped != "" {
				entry["litellmWrapped"] = signature.LiteLLMWrapped
			}
			entries = append(entries, entry)
		}
		result["modelUnavailable"] = entries
	}
	if len(signatures.PermanentCodes) != 0 {
		codes := make([]any, 0, len(signatures.PermanentCodes))
		for _, code := range signatures.PermanentCodes {
			codes = append(codes, code)
		}
		result["permanentCodes"] = codes
	}
	return result
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
