package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

// plannerContextManagementSource is the authored closed union of a Stage's
// plannerContextManagement (WFL-R90): strategy none alone, or compact@1 with
// its compactor policy, instructions and optional thresholds.
type plannerContextManagementSource struct {
	Strategy           string                 `yaml:"strategy"`
	ModelPolicy        string                 `yaml:"modelPolicy,omitempty"`
	Instructions       *instructionsRefSource `yaml:"instructions,omitempty"`
	ContextWindowRatio *float64               `yaml:"contextWindowRatio,omitempty"`
	KeepRecentGroups   *int                   `yaml:"keepRecentGroups,omitempty"`
	MaxCompactions     *int                   `yaml:"maxCompactions,omitempty"`
}

// resolvePlannerContextManagement normalizes a Stage's authored
// plannerContextManagement. An omitted member and strategy none both resolve
// to nil; compact@1 resolves to its complete branch with every default
// explicit and to the exact compactor ModelPolicy it names, which the caller
// pins as the base Planner compactor selection. Explicit null members,
// unknown or duplicate fields and mixed branches are rejected.
func (l *loader) resolvePlannerContextManagement(
	node yaml.Node,
	operation bool,
) (*resolved.PlannerContextManagement, *contracts.ResolvedModelPolicy, error) {
	if node.Kind == 0 {
		return nil, nil, nil
	}
	if operation {
		return nil, nil, fmt.Errorf("Server operation forbids plannerContextManagement")
	}
	if node.Kind != yaml.MappingNode {
		return nil, nil, fmt.Errorf("must be a mapping")
	}
	for index := 0; index+1 < len(node.Content); index += 2 {
		if value := node.Content[index+1]; value.Kind == yaml.ScalarNode && value.ShortTag() == "!!null" {
			return nil, nil, fmt.Errorf("%s must not be null", node.Content[index].Value)
		}
	}
	var source plannerContextManagementSource
	if err := node.Load(&source, yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return nil, nil, err
	}
	switch source.Strategy {
	case resolved.PlannerContextNone:
		if source.ModelPolicy != "" || source.Instructions != nil || source.ContextWindowRatio != nil ||
			source.KeepRecentGroups != nil || source.MaxCompactions != nil {
			return nil, nil, fmt.Errorf("strategy %s accepts no other field", resolved.PlannerContextNone)
		}
		return nil, nil, nil
	case resolved.PlannerContextCompact:
	default:
		return nil, nil, fmt.Errorf(
			"strategy must be %s or %s", resolved.PlannerContextNone, resolved.PlannerContextCompact,
		)
	}
	if source.ModelPolicy == "" {
		return nil, nil, fmt.Errorf("strategy %s requires modelPolicy", resolved.PlannerContextCompact)
	}
	policy, err := l.compactorPolicy(source.ModelPolicy)
	if err != nil {
		return nil, nil, err
	}
	instructions, err := l.resolveInstructions(source.Instructions)
	if err != nil {
		return nil, nil, fmt.Errorf("instructions: %w", err)
	}
	result := &resolved.PlannerContextManagement{
		Strategy:           resolved.PlannerContextCompact,
		Instructions:       instructions,
		ContextWindowRatio: resolved.DefaultPlannerContextWindowRatio,
		KeepRecentGroups:   resolved.DefaultPlannerKeepRecentGroups,
		MaxCompactions:     resolved.DefaultPlannerMaxCompactions,
	}
	if source.ContextWindowRatio != nil {
		result.ContextWindowRatio = *source.ContextWindowRatio
	}
	if source.KeepRecentGroups != nil {
		result.KeepRecentGroups = *source.KeepRecentGroups
	}
	if source.MaxCompactions != nil {
		result.MaxCompactions = *source.MaxCompactions
	}
	if err := result.Validate(); err != nil {
		return nil, nil, err
	}
	return result, &policy, nil
}
