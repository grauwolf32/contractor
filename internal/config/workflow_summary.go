package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"go.yaml.in/yaml/v4"
)

func (s *stageTransitionsSource) UnmarshalYAML(node *yaml.Node) error {
	type plain stageTransitionsSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	s.summarizedPresent = yamlMappingHasKey(node, "summarized")
	return nil
}

func (s *escalateSource) UnmarshalYAML(node *yaml.Node) error {
	type plain escalateSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	s.plannerPresent = yamlMappingHasKey(node, "planner")
	s.deadlinePresent = yamlMappingHasKey(node, "deadlineSeconds")
	return nil
}

func resolveEscalationPlanner(field string, source *escalateSource) (*resolved.PlannerRef, int, error) {
	if field != "on.summarized" {
		if source.plannerPresent || source.deadlinePresent {
			return nil, 0, fmt.Errorf("%s: Planner strategy/deadline overrides require on.summarized", field)
		}
		return nil, 0, nil
	}
	if source.Planner == nil || source.DeadlineSeconds == nil {
		return nil, 0, fmt.Errorf("on.summarized.escalate requires planner and deadlineSeconds")
	}
	selector, err := resolved.ParseSelector(*source.Planner)
	if err != nil {
		return nil, 0, fmt.Errorf("on.summarized.escalate.planner: %w", err)
	}
	return &resolved.PlannerRef{PlannerID: selector.ID, Version: selector.Version}, *source.DeadlineSeconds, nil
}
