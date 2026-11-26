package config

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

func modelPolicyDigest(selector resolved.Selector, policy contracts.ResolvedModelPolicy) (string, error) {
	spec := map[string]any{
		"model": policy.Model,
	}
	addModelPolicyOptions(spec, policy)
	if policy.Temperature != nil {
		spec["temperature"] = *policy.Temperature
	}
	manifest := map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       modelPolicyKind,
		"metadata": map[string]any{
			"name":    selector.ID,
			"version": selector.Version,
		},
		"spec": spec,
	}
	return contentdigest.CanonicalJSON(manifest)
}

func llmGatewayConfigDigest(
	selector resolved.Selector, gateway llmgateway.ResolvedLLMGatewayConfig,
) (string, error) {
	spec := map[string]any{
		"protocol": gateway.Protocol,
		"url":      gateway.URL,
	}
	if gateway.CredentialManager != nil {
		spec["credentialManager"] = map[string]any{
			"implementation": gateway.CredentialManager.Implementation,
			"managementUrl":  gateway.CredentialManager.ManagementURL,
		}
	}
	if gateway.FailureSignatures != nil {
		spec["failureSignatures"] = failureSignaturesDocument(*gateway.FailureSignatures)
	}
	return contentdigest.CanonicalJSON(map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       llmGatewayConfigKind,
		"metadata": map[string]any{
			"name": selector.ID, "version": selector.Version,
		},
		"spec": spec,
	})
}

func agentTemplateDigest(selector resolved.Selector, template contracts.ResolvedAgentTemplate) (string, error) {
	template.Ref.TemplateID, template.Ref.Version = selector.ID, selector.Version
	return template.ContentDigest()
}

func addModelPolicyOptions(target map[string]any, policy contracts.ResolvedModelPolicy) {
	if policy.SupportsImages != nil {
		target["supportsImages"] = *policy.SupportsImages
	}
	for name, value := range map[string]int{
		"contextWindowTokens": policy.ContextWindowTokens,
		"maxOutputTokens":     policy.MaxOutputTokens,
		"maxModelCalls":       policy.MaxModelCalls,
		"maxToolCalls":        policy.MaxToolCalls,
		"maxWorkerCalls":      policy.MaxWorkerCalls,
		"maxTotalTokens":      policy.MaxTotalTokens,
	} {
		if value != 0 {
			target[name] = value
		}
	}
}
