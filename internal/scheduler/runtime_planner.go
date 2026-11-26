package scheduler

import (
	"fmt"
	"maps"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// executionStage adds the built-in Planner executor only to the allocation
// view. It never changes the immutable Workflow/Stage snapshot or Worker set.
func executionStage(stage resolved.ResolvedStage) resolved.ResolvedStage {
	if stage.Planner.Selector() != resolved.StreamlinePlanner {
		return stage
	}
	name := "planner_executor"
	for n := 1; ; n++ {
		free := true
		for logicalName, binding := range stage.Agents {
			if logicalName == name || binding.Namespace == name {
				free = false
				break
			}
		}
		if free {
			break
		}
		name = fmt.Sprintf("planner_executor_%d", n)
	}
	instruction := "Execute the Server-authorized Planner invocation through its execution host."
	template := contracts.ResolvedAgentTemplate{
		Ref:            contracts.AgentTemplateRef{TemplateID: "planner-executor", Version: "1"},
		Description:    "Allocation-owned Planner model executor",
		Runtime:        contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"},
		Instructions:   contracts.ResolvedInstructions{Ref: "builtin/planner-executor", Text: instruction, Digest: contentdigest.Bytes([]byte(instruction))},
		Toolsets:       []contracts.ToolsetSelection{},
		SandboxProfile: contracts.SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"},
	}
	digest, err := template.ContentDigest()
	if err != nil {
		panic("built-in Planner template cannot be digested")
	}
	template.Ref.Digest = digest
	stage.Agents = maps.Clone(stage.Agents)
	stage.Agents[name] = resolved.ResolvedAgentBinding{Namespace: name, Template: template}
	stage.ExecutionConfig.Agents = maps.Clone(stage.ExecutionConfig.Agents)
	if stage.ExecutionConfig.Planner != nil {
		selection := *stage.ExecutionConfig.Planner
		// The Planner invokes its auxiliary model inside its own allocation;
		// this is not Worker context management.
		selection.CompactorModelPolicy = nil
		stage.ExecutionConfig.Agents[name] = selection
	}
	return stage
}
