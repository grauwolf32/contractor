package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"go.yaml.in/yaml/v4"
)

// plannerContextManagementSource is the authored closed union of a Stage's
// plannerContextManagement (WFL-R90): strategy none alone, or compact@1 with
// instructions and optional thresholds. ExecutionConfig owns its model policy.
type plannerContextManagementSource struct {
	Strategy           string                 `yaml:"strategy"`
	Instructions       *instructionsRefSource `yaml:"instructions,omitempty"`
	ContextWindowRatio *float64               `yaml:"contextWindowRatio,omitempty"`
	KeepRecentGroups   *int                   `yaml:"keepRecentGroups,omitempty"`
	MaxCompactions     *int                   `yaml:"maxCompactions,omitempty"`
}

// resolvePlannerContextManagement normalizes behavior for both ordinary Stages
// and summary continuations. ExecutionConfig supplies their compactor policies.
// Unknown, null and duplicate members and mixed branches are rejected.
func (l *loader) resolvePlannerContextManagement(
	node yaml.Node,
	operation bool,
) (*resolved.PlannerContextManagement, error) {
	if node.Kind == 0 {
		return nil, nil
	}
	resolvedNode := yamlAliasTarget(&node)
	if resolvedNode == nil {
		return nil, fmt.Errorf("invalid YAML alias")
	}
	node = *resolvedNode
	if operation {
		return nil, fmt.Errorf("Server operation forbids plannerContextManagement")
	}
	if node.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("must be a mapping")
	}
	var members map[string]yaml.Node
	if err := node.Load(&members, yaml.WithUniqueKeys()); err != nil {
		return nil, err
	}
	for name, member := range members {
		if value := yamlAliasTarget(&member); value != nil && value.Kind == yaml.ScalarNode && value.ShortTag() == "!!null" {
			return nil, fmt.Errorf("%s must not be null", name)
		}
	}
	var source plannerContextManagementSource
	if err := node.Load(&source, yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return nil, err
	}
	switch source.Strategy {
	case resolved.PlannerContextNone:
		if len(members) != 1 {
			return nil, fmt.Errorf("strategy %s accepts no other field", resolved.PlannerContextNone)
		}
		return nil, nil
	case resolved.PlannerContextCompact:
	default:
		return nil, fmt.Errorf(
			"strategy must be %s or %s", resolved.PlannerContextNone, resolved.PlannerContextCompact,
		)
	}
	instructions, err := l.resolveInstructions(source.Instructions)
	if err != nil {
		return nil, fmt.Errorf("instructions: %w", err)
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
		return nil, err
	}
	return result, nil
}
