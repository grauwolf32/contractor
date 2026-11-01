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
	addModelPolicyLimits(spec, policy)
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
	toolsets := make([]any, 0, len(template.Toolsets))
	for _, selection := range template.Toolsets {
		tools := make([]any, len(selection.Tools))
		for index, tool := range selection.Tools {
			tools[index] = tool
		}
		toolsets = append(toolsets, map[string]any{
			"ref": map[string]any{
				"toolsetId": selection.Ref.ToolsetID,
				"version":   selection.Ref.Version,
			},
			"tools": tools,
		})
	}
	spec := map[string]any{
		"description": template.Description,
		"runtime": map[string]any{
			"runtimeId": template.Runtime.RuntimeID,
			"version":   template.Runtime.Version,
		},
		"instructions": map[string]any{
			"ref":    template.Instructions.Ref,
			"digest": template.Instructions.Digest,
		},
		"toolsets": toolsets,
		"sandboxProfile": map[string]any{
			"sandboxProfileId": template.SandboxProfile.SandboxProfileID,
			"version":          template.SandboxProfile.Version,
		},
	}
	if template.IsToolWorker() {
		delete(spec, "instructions")
		spec["execution"] = template.Execution
	}
	if template.Sitemap != nil {
		spec["sitemap"] = template.Sitemap
	}
	if len(template.Skills) > 0 {
		skills := make([]any, len(template.Skills))
		for index, skill := range template.Skills {
			skills[index] = map[string]any{"namespace": skill.Namespace, "name": skill.Name}
		}
		spec["skills"] = skills
	}
	if template.Summarizer != nil {
		summarizer := map[string]any{}
		if template.Summarizer.CumulativeBudget != nil {
			summarizer["cumulativeBudget"] = *template.Summarizer.CumulativeBudget
		}
		summarizer["contextWindowRatio"] = template.Summarizer.ContextWindowRatio
		summarizer["includeReasoning"] = template.Summarizer.IncludeReasoning
		if instructions := template.Summarizer.Instructions; instructions != nil {
			summarizer["instructions"] = map[string]any{
				"ref": instructions.Ref, "digest": instructions.Digest,
			}
		}
		spec["summarizer"] = summarizer
	}

	manifest := map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       agentTemplateKind,
		"metadata": map[string]any{
			"name":    selector.ID,
			"version": selector.Version,
		},
		"spec": spec,
	}
	return contentdigest.CanonicalJSON(manifest)
}

func addModelPolicyLimits(target map[string]any, policy contracts.ResolvedModelPolicy) {
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
