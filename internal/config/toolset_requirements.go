package config

import (
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const MaximumToolsetConsumers = 256

type ToolsetConsumer struct {
	Stage string `json:"stage"`
	Agent string `json:"agent"`
}

type ToolsetConfigurationRequirement struct {
	Ref                   string            `json:"ref"`
	ConfigurationRequired bool              `json:"configurationRequired"`
	ConfigurationSchema   string            `json:"configurationSchema"`
	ToolDiscovery         string            `json:"toolDiscovery"`
	WorkerRuntimes        []string          `json:"workerRuntimes"`
	Tools                 []string          `json:"tools"`
	Consumers             []ToolsetConsumer `json:"consumers"`
}

// WorkflowToolsetRequirements includes every stage, even stages which do not
// execute until after a transition. Only code-owned configurable refs contribute.
func WorkflowToolsetRequirements(workflow ResolvedWorkflow) ([]ToolsetConfigurationRequirement, error) {
	descriptors := sandboxPlacementDescriptors.Toolsets
	byRef := map[string]*ToolsetConfigurationRequirement{}
	tools := map[string]map[string]struct{}{}
	for stageName, stage := range workflow.Stages {
		for agentName, binding := range stage.Agents {
			for _, selection := range binding.Template.Toolsets {
				ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
				descriptor, ok := descriptors[ref]
				if !ok || descriptor.Configuration == nil {
					continue
				}
				requirement := byRef[ref]
				if requirement == nil {
					mode := descriptor.ToolDiscovery
					if mode == "" {
						mode = "static"
					}
					requirement = &ToolsetConfigurationRequirement{Ref: ref,
						ConfigurationRequired: descriptor.Configuration.Required,
						ConfigurationSchema:   descriptor.Configuration.Schema, ToolDiscovery: mode,
						WorkerRuntimes: append([]string{}, descriptor.SupportedWorkerRuntimes...)}
					byRef[ref], tools[ref] = requirement, map[string]struct{}{}
				}
				requirement.Consumers = append(requirement.Consumers, ToolsetConsumer{Stage: stageName, Agent: agentName})
				if len(requirement.Consumers) > MaximumToolsetConsumers || len(byRef) > contracts.MaximumConfiguredToolsets {
					return nil, fmt.Errorf("Workflow toolset requirements exceed supported limits")
				}
				for _, name := range selection.Tools {
					tools[ref][name] = struct{}{}
				}
			}
		}
	}
	result := make([]ToolsetConfigurationRequirement, 0, len(byRef))
	for ref, requirement := range byRef {
		for name := range tools[ref] {
			requirement.Tools = append(requirement.Tools, name)
		}
		sort.Strings(requirement.Tools)
		sort.Slice(requirement.Consumers, func(i, j int) bool {
			left, right := requirement.Consumers[i], requirement.Consumers[j]
			if left.Stage != right.Stage {
				return left.Stage < right.Stage
			}
			return left.Agent < right.Agent
		})
		result = append(result, *requirement)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Ref < result[j].Ref })
	return result, nil
}

// RequiredToolsetConfiguration is shared by allocation and Run admission.
func RequiredToolsetConfiguration(ref string) bool {
	descriptor := sandboxPlacementDescriptors.Toolsets[ref]
	return descriptor.Configuration != nil && descriptor.Configuration.Required
}
