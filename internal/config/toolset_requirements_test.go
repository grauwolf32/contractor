package config

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestWorkflowToolsetRequirementsIncludesLateStagesAndBoundsConsumers(t *testing.T) {
	agent := func(toolset, tool string) ResolvedAgentBinding {
		return ResolvedAgentBinding{Template: contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: toolset, Version: "1"}, Tools: []string{tool}}}}}
	}
	workflow := ResolvedWorkflow{Stages: map[string]ResolvedStage{
		"start": {Agents: map[string]ResolvedAgentBinding{"native": agent("run-artifacts", "read_artifact")}},
		"late":  {Agents: map[string]ResolvedAgentBinding{"finder": agent("mcp", "search"), "reader": agent("mcp", "fetch")}},
	}}
	requirements, err := WorkflowToolsetRequirements(workflow)
	if err != nil || len(requirements) != 1 {
		t.Fatalf("requirements: %v", err)
	}
	r := requirements[0]
	if r.Ref != "mcp@1" || !r.ConfigurationRequired || !reflect.DeepEqual(r.Tools, []string{"fetch", "search"}) || !reflect.DeepEqual(r.Consumers, []ToolsetConsumer{{Stage: "late", Agent: "finder"}, {Stage: "late", Agent: "reader"}}) {
		t.Fatalf("requirements differ: %+v", r)
	}
	for i := 0; i < MaximumToolsetConsumers; i++ {
		workflow.Stages["late"].Agents[fmt.Sprintf("reader-%d", i)] = agent("mcp", "fetch")
	}
	if _, err := WorkflowToolsetRequirements(workflow); err == nil {
		t.Fatal("unbounded consumers accepted")
	}
}
