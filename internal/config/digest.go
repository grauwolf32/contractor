package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/ucarion/jcs"
)

func modelPolicyDigest(selector Selector, policy contracts.ResolvedModelPolicy) (string, error) {
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
	return digestJCS(manifest)
}

func llmGatewayConfigDigest(
	selector Selector, gateway contracts.ResolvedLLMGatewayConfig,
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
	return digestJCS(map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       llmGatewayConfigKind,
		"metadata": map[string]any{
			"name": selector.ID, "version": selector.Version,
		},
		"spec": spec,
	})
}

func executionConfigDigest(selector Selector, patch StageExecutionConfigPatch) (string, error) {
	return digestJCS(map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       executionConfigKind,
		"metadata": map[string]any{
			"name": selector.ID, "version": selector.Version,
		},
		"spec": patch.canonicalValue(),
	})
}

func agentTemplateDigest(selector Selector, template contracts.ResolvedAgentTemplate) (string, error) {
	modelPolicy := map[string]any{
		"ref": map[string]any{
			"policyId": template.ModelPolicy.Ref.PolicyID,
			"version":  template.ModelPolicy.Ref.Version,
			"digest":   template.ModelPolicy.Ref.Digest,
		},
		"model": template.ModelPolicy.Model,
	}
	addModelPolicyLimits(modelPolicy, template.ModelPolicy)
	if template.ModelPolicy.Temperature != nil {
		modelPolicy["temperature"] = *template.ModelPolicy.Temperature
	}

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

	manifest := map[string]any{
		"apiVersion": contracts.APIVersion,
		"kind":       agentTemplateKind,
		"metadata": map[string]any{
			"name":    selector.ID,
			"version": selector.Version,
		},
		"spec": map[string]any{
			"description": template.Description,
			"runtime": map[string]any{
				"runtimeId": template.Runtime.RuntimeID,
				"version":   template.Runtime.Version,
			},
			"instructions": map[string]any{
				"ref":    template.Instructions.Ref,
				"digest": template.Instructions.Digest,
			},
			"modelPolicy": modelPolicy,
			"toolsets":    toolsets,
			"sandboxProfile": map[string]any{
				"sandboxProfileId": template.SandboxProfile.SandboxProfileID,
				"version":          template.SandboxProfile.Version,
			},
		},
	}
	return digestJCS(manifest)
}

func addModelPolicyLimits(target map[string]any, policy contracts.ResolvedModelPolicy) {
	for name, value := range map[string]int{
		"maxOutputTokens": policy.MaxOutputTokens,
		"maxModelCalls":   policy.MaxModelCalls,
		"maxToolCalls":    policy.MaxToolCalls,
		"maxWorkerCalls":  policy.MaxWorkerCalls,
		"maxTotalTokens":  policy.MaxTotalTokens,
	} {
		if value != 0 {
			target[name] = value
		}
	}
}

// digestJCS converts Go's typed values through encoding/json so the JCS
// implementation receives exactly the I-JSON data model it supports.
func digestJCS(manifest any) (string, error) {
	encoded, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("encode digest manifest: %w", err)
	}
	var jsonValue any
	if err := json.Unmarshal(encoded, &jsonValue); err != nil {
		return "", fmt.Errorf("normalize digest manifest: %w", err)
	}
	canonical, err := jcs.Format(jsonValue)
	if err != nil {
		return "", fmt.Errorf("canonicalize digest manifest: %w", err)
	}
	sum := sha256.Sum256([]byte(canonical))
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func digestBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}
