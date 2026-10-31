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
	return l.resolvePlannerContextPolicy(node, operation, true)
}

// Summary continuations select their compactor through the existing escalation
// executionConfig.planner.compactorModelPolicy; ordinary Stages also accept the
// policy in their context block as the lowest-precedence execution selection.
func (l *loader) resolvePlannerContextPolicy(
	node yaml.Node, operation, withModelPolicy bool,
) (*resolved.PlannerContextManagement, *contracts.ResolvedModelPolicy, error) {
	if node.Kind == 0 {
		return nil, nil, nil
	}
	resolvedNode := yamlAliasTarget(&node)
	if resolvedNode == nil {
		return nil, nil, fmt.Errorf("invalid YAML alias")
	}
	node = *resolvedNode
	if operation {
		return nil, nil, fmt.Errorf("Server operation forbids plannerContextManagement")
	}
	if node.Kind != yaml.MappingNode {
		return nil, nil, fmt.Errorf("must be a mapping")
	}
	var members map[string]yaml.Node
	if err := node.Load(&members, yaml.WithUniqueKeys()); err != nil {
		return nil, nil, err
	}
	for name, member := range members {
		if value := yamlAliasTarget(&member); value != nil && value.Kind == yaml.ScalarNode && value.ShortTag() == "!!null" {
			return nil, nil, fmt.Errorf("%s must not be null", name)
		}
	}
	var source plannerContextManagementSource
	if err := node.Load(&source, yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return nil, nil, err
	}
	switch source.Strategy {
	case resolved.PlannerContextNone:
		if len(members) != 1 {
			return nil, nil, fmt.Errorf("strategy %s accepts no other field", resolved.PlannerContextNone)
		}
		return nil, nil, nil
	case resolved.PlannerContextCompact:
	default:
		return nil, nil, fmt.Errorf(
			"strategy must be %s or %s", resolved.PlannerContextNone, resolved.PlannerContextCompact,
		)
	}
	var policy *contracts.ResolvedModelPolicy
	if withModelPolicy {
		if source.ModelPolicy == "" {
			return nil, nil, fmt.Errorf("strategy %s requires modelPolicy", resolved.PlannerContextCompact)
		}
		selected, err := l.compactorPolicy(source.ModelPolicy)
		if err != nil {
			return nil, nil, err
		}
		policy = &selected
	} else if _, present := members["modelPolicy"]; present {
		return nil, nil, fmt.Errorf("summary compactor policy belongs in executionConfig.planner.compactorModelPolicy")
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
	return result, policy, nil
}
