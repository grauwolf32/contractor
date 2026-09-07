package config

import (
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type ConfigurationKind string

const (
	ConfigurationAgentTemplates   ConfigurationKind = "agent-templates"
	ConfigurationExecutionConfigs ConfigurationKind = "execution-configs"
	ConfigurationModelPolicies    ConfigurationKind = "model-policies"
	ConfigurationLLMGateways      ConfigurationKind = "llm-gateways"
)

type ConfigurationSource string

const (
	ConfigurationSourceOperator ConfigurationSource = "operator"
	ConfigurationSourceManaged  ConfigurationSource = "managed"
)

var (
	ErrInvalidConfigurationKind = errors.New("invalid configuration kind")
	ErrConfigurationNotFound    = errors.New("configuration not found")
	ErrInvalidPublication       = errors.New("invalid configuration publication")
	ErrPublicationConflict      = errors.New("configuration publication conflict")
)

type ConfigurationRef struct {
	Kind    ConfigurationKind `json:"kind"`
	Name    string            `json:"name"`
	Version string            `json:"version"`
	Digest  string            `json:"digest"`
}

// ConfigurationResource is a safe public projection. Body is always one of
// the closed, secret-free shapes constructed in this package.
type ConfigurationResource struct {
	Ref    ConfigurationRef    `json:"ref"`
	Body   any                 `json:"body"`
	Source ConfigurationSource `json:"source"`
}

func ParseConfigurationKind(raw string) (ConfigurationKind, error) {
	kind := ConfigurationKind(raw)
	switch kind {
	case ConfigurationAgentTemplates, ConfigurationExecutionConfigs,
		ConfigurationModelPolicies, ConfigurationLLMGateways:
		return kind, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidConfigurationKind, raw)
	}
}

func (s *Snapshot) Configurations(kind ConfigurationKind) ([]ConfigurationResource, error) {
	if _, err := ParseConfigurationKind(string(kind)); err != nil {
		return nil, err
	}
	selectors := s.configurationSelectors(kind)
	result := make([]ConfigurationResource, 0, len(selectors))
	for _, selector := range selectors {
		resource, err := s.Configuration(kind, selector)
		if err != nil {
			return nil, err
		}
		result = append(result, resource)
	}
	return result, nil
}

func (s *Snapshot) Configuration(
	kind ConfigurationKind, raw string,
) (ConfigurationResource, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return ConfigurationResource{}, fmt.Errorf("%w: invalid selector", ErrInvalidPublication)
	}
	source := s.configurationSource(kind, selector.String())
	switch kind {
	case ConfigurationModelPolicies:
		value, ok := s.policies[selector.String()]
		if !ok {
			return ConfigurationResource{}, ErrConfigurationNotFound
		}
		return ConfigurationResource{
			Ref: ConfigurationRef{
				Kind: kind, Name: selector.ID, Version: selector.Version, Digest: value.Ref.Digest,
			},
			Body: modelPolicyResourceBody(value), Source: source,
		}, nil
	case ConfigurationLLMGateways:
		value, ok := s.gateways[selector.String()]
		if !ok {
			return ConfigurationResource{}, ErrConfigurationNotFound
		}
		return ConfigurationResource{
			Ref: ConfigurationRef{
				Kind: kind, Name: selector.ID, Version: selector.Version, Digest: value.Ref.Digest,
			},
			Body: llmGatewayResourceBody(value), Source: source,
		}, nil
	case ConfigurationAgentTemplates:
		value, ok := s.templates[selector.String()]
		if !ok {
			return ConfigurationResource{}, ErrConfigurationNotFound
		}
		return ConfigurationResource{
			Ref: ConfigurationRef{
				Kind: kind, Name: selector.ID, Version: selector.Version, Digest: value.Ref.Digest,
			},
			Body: agentTemplateResourceBody(value), Source: source,
		}, nil
	case ConfigurationExecutionConfigs:
		value, ok := s.executionConfigs[selector.String()]
		if !ok {
			return ConfigurationResource{}, ErrConfigurationNotFound
		}
		return ConfigurationResource{
			Ref: ConfigurationRef{
				Kind: kind, Name: selector.ID, Version: selector.Version, Digest: value.Ref.Digest,
			},
			Body: executionConfigResourceBody(value), Source: source,
		}, nil
	default:
		return ConfigurationResource{}, ErrInvalidConfigurationKind
	}
}

func (s *Snapshot) configurationSelectors(kind ConfigurationKind) []string {
	var result []string
	switch kind {
	case ConfigurationModelPolicies:
		result = mapKeys(s.policies)
	case ConfigurationLLMGateways:
		result = mapKeys(s.gateways)
	case ConfigurationAgentTemplates:
		result = mapKeys(s.templates)
	case ConfigurationExecutionConfigs:
		result = mapKeys(s.executionConfigs)
	}
	sort.Strings(result)
	return result
}

func mapKeys[T any](values map[string]T) []string {
	result := make([]string, 0, len(values))
	for key := range values {
		result = append(result, key)
	}
	return result
}

func (s *Snapshot) configurationSource(
	kind ConfigurationKind, selector string,
) ConfigurationSource {
	source, ok := s.sources[configurationSourceKey(kind, selector)]
	if !ok {
		return ConfigurationSourceOperator
	}
	return source
}

func configurationSourceKey(kind ConfigurationKind, identity string) string {
	return string(kind) + "\x00" + identity
}

func modelPolicyResourceBody(policy contracts.ResolvedModelPolicy) map[string]any {
	result := map[string]any{"model": policy.Model}
	addModelPolicyLimits(result, policy)
	if policy.Temperature != nil {
		result["temperature"] = *policy.Temperature
	}
	return result
}

func llmGatewayResourceBody(gateway contracts.ResolvedLLMGatewayConfig) map[string]any {
	result := map[string]any{"protocol": gateway.Protocol, "url": gateway.URL}
	if gateway.CredentialManager != nil {
		result["credentialManager"] = map[string]any{
			"implementation": gateway.CredentialManager.Implementation,
			"managementUrl":  gateway.CredentialManager.ManagementURL,
		}
	}
	return result
}

func agentTemplateResourceBody(template contracts.ResolvedAgentTemplate) map[string]any {
	toolsets := make([]any, 0, len(template.Toolsets))
	for _, selection := range template.Toolsets {
		toolsets = append(toolsets, map[string]any{
			"ref":   selection.Ref.ToolsetID + "@" + selection.Ref.Version,
			"tools": append([]string(nil), selection.Tools...),
		})
	}
	result := map[string]any{
		"description": template.Description,
		"runtime":     template.Runtime.RuntimeID + "@" + template.Runtime.Version,
		"instructions": map[string]any{
			"ref": template.Instructions.Ref, "digest": template.Instructions.Digest,
		},
		"modelPolicy": template.ModelPolicy.Ref,
		"toolsets":    toolsets,
		"sandboxProfile": template.SandboxProfile.SandboxProfileID + "@" +
			template.SandboxProfile.Version,
	}
	if len(template.Skills) > 0 {
		result["skills"] = append([]contracts.ArtifactRef(nil), template.Skills...)
	}
	if template.Summarizer != nil {
		summarizer := map[string]any{
			"modelPolicy":        template.Summarizer.ModelPolicy.Ref,
			"contextWindowRatio": template.Summarizer.ContextWindowRatio,
		}
		if template.Summarizer.CumulativeBudget != nil {
			summarizer["cumulativeBudget"] = *template.Summarizer.CumulativeBudget
		}
		result["summarizer"] = summarizer
		if instructions := template.Summarizer.Instructions; instructions != nil {
			summarizer["instructions"] = map[string]any{
				"ref": instructions.Ref, "digest": instructions.Digest,
			}
		}
	}
	return result
}

func executionConfigResourceBody(profile ResolvedExecutionConfigProfile) map[string]any {
	result := make(map[string]any, 2)
	if profile.Override.Planner != nil {
		result["planner"] = executionSelectionResourceBody(*profile.Override.Planner)
	}
	if len(profile.Override.Agents) != 0 {
		agents := make(map[string]any, len(profile.Override.Agents))
		for name, selection := range profile.Override.Agents {
			agents[name] = executionSelectionResourceBody(selection)
		}
		result["agents"] = agents
	}
	return result
}

func executionSelectionResourceBody(selection ResolvedExecutionSelectionOverride) map[string]any {
	result := make(map[string]any, 3)
	if selection.ModelPolicy != nil {
		result["modelPolicy"] = selection.ModelPolicy.Ref
	}
	if selection.LLMGateway != nil {
		result["llmGateway"] = selection.LLMGateway.Ref
	}
	if selection.Credential != nil {
		if selection.Credential.Clear {
			result["credential"] = nil
		} else {
			result["credential"] = selection.Credential.Ref
		}
	}
	return result
}
