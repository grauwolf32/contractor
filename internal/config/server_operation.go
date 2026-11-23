package config

import (
	"fmt"

	"go.yaml.in/yaml/v4"
)

func (s *stageSource) UnmarshalYAML(node *yaml.Node) error {
	type plain stageSource
	if err := node.Load((*plain)(s), yaml.WithKnownFields(), yaml.WithUniqueKeys()); err != nil {
		return err
	}
	if yamlMappingHasKey(node, "operation") {
		if s.Operation == nil || yamlMappingHasKey(node, "planner") || yamlMappingHasKey(node, "instructions") {
			return fmt.Errorf("operation requires an object and forbids planner and instructions")
		}
	}
	return nil
}
